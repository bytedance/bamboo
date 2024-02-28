import argparse
import json
import math
import os
import shutil
from datetime import datetime
from random import shuffle
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.bamboo_et import BambooET
from utils.loader import BambooLoaderFactory
from utils.dispersion import Dispersion
from utils.get_batch import get_batch, split_data
from utils.log_helper import create_logger


def get_parser(config_path: Optional[str] = None):
    parser = argparse.ArgumentParser(description='Args for bamboo simulation.')
    # General arguments
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--job_name', default='default', type=str)
    parser.add_argument('--model_name', type=str, default='ET')
    parser.add_argument('--datapath', default='/mnt/bn/ai4s-hl/bamboo/pyscf_data/data/')
    parser.add_argument('--datafile_complete_training', default='train_pyscf_svpd_b3lyp_complete_fsi_data_01312024')
    parser.add_argument('--datafile_complete_val', default='val_pyscf_svpd_b3lyp_complete_fsi_data_01312024')
    parser.add_argument('--datafile_clean_training', default='train_pyscf_svpd_b3lyp_clean_loose_fsi_data_01312024')
    parser.add_argument('--datafile_clean_val', default='val_pyscf_svpd_b3lyp_clean_loose_fsi_data_01312024')
    parser.add_argument('--max_queue_size', default=100, type=int)
    parser.add_argument('--jobpath', default='/mnt/bn/ai4s-hl/bamboo/bamboo_training/test_pretrain')
    parser.add_argument('--fp32', action="store_true")
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--force_retrain', action='store_true')
    # Training arguments
    parser.add_argument('--train_bs', default=128, type=int)
    parser.add_argument('--val_bs', default=128, type=int)
    parser.add_argument('--num_epoch', default=750, type=int)
    parser.add_argument('--num_pretrain_epoch', default=600, type=float)
    parser.add_argument('--num_warm_up_epoch', default=10, type=int)
    parser.add_argument('--optimizer', default='adamax', type=str)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--scheduler_gamma', default=0.99, type=float)
    parser.add_argument('--loss_charge_ratio', default=10.0, type=float)
    parser.add_argument('--loss_dipole_ratio', default=10.0, type=float)
    parser.add_argument('--loss_quadrupole_ratio', default=0.0, type=float)
    parser.add_argument('--loss_energy_ratio', default=0.01, type=float)
    parser.add_argument('--loss_forces_ratio', default=0.3, type=float)
    parser.add_argument('--loss_virial_ratio', default=0.01, type=float)
    parser.add_argument('--charge_ub', default=2.0, type=float)
    parser.add_argument('--nnfij_penalty', default=1e-4, type=float)
    parser.add_argument('--h_force_extra_ratio', default=0.3, type=float)
    parser.add_argument('--decoul_dr_force_norm_ratio', default=3e2, type=float)
    # Model arguments
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_rbf', type=int, default=32)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--act_fn', type=str, default='SiLU')
    parser.add_argument('--rcut', type=float, default=5.0)
    parser.add_argument('--coul_damping_beta', type=float, default=18.7)
    parser.add_argument('--coul_damping_r0', type=float, default=2.2)
    parser.add_argument('--disp_damping_beta', type=float, default=23.0)
    parser.add_argument('--disp_cutoff', type=float, default=10.0)
    parser.add_argument('--disp_training', type=str, default=Dispersion.NO.name)
    parser.add_argument('--disp_simulation', type=str, default=Dispersion.D3CSO.name)
    parser.add_argument('--energy_mlp_layers', type=int, default=2)
    parser.add_argument('--charge_mlp_layers', type=int, default=2)
    
    args = parser.parse_args()

    if isinstance(config_path, str) and os.path.isfile(config_path):
        config_file_for_update = config_path
    elif os.path.isfile(args.config):
        config_file_for_update = args.config
    else:
        return args

    with open(config_file_for_update, 'r') as f:
        config_args = json.load(f)
    for key, value in config_args.items():
        if value is None:
            continue
        setattr(args, key, value)
    return args


def init_method_file(job_name: str):
    return f"/tmp/init_torch_ddp_{job_name}"


class BambooTrainer():
    """
    Basic trainer for bamboo
    """
    def __init__(self, args, rank=0, world_size=1):
        self.args = args
        self.rank = rank
        self.world_size = world_size

        # Init log dir
        if not os.path.exists(self.args.jobpath):
            os.makedirs(self.args.jobpath, exist_ok=True)
        job_path = os.path.join(self.args.jobpath, self.args.job_name).lower()
        if not os.path.exists(job_path):
            os.makedirs(job_path, exist_ok=True)
        train_log_path = os.path.join(job_path, 'train_logs')
        if not os.path.exists(train_log_path):
            os.makedirs(train_log_path, exist_ok=True)
        
        if self.me:
            log_file = os.path.join(train_log_path, f"train_{datetime.now().strftime('%m%d%H%M')}.log")
            self.logger = create_logger(name="TRAIN", log_file=log_file)
        else:
            self.logger = create_logger(name="TRAIN", screen=False)

        self.logger.info(f"Initializing.")
        self.logger.info(f"World size: {self.world_size}, Rank: {self.rank}")      
        for k_args, v_args in vars(args).items():
            self.logger.info(f'{k_args} = {v_args}')

        # Init device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank}")
            self.logger.info(f'device = cuda:{self.rank}')
        else:
            raise RuntimeError("Cannot find CUDA device.")

        # Init dtype
        if self.args.fp32:
            torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.float64)
        self.dtype = torch.get_default_dtype()

        # Init random seed
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

        # Init loss ratios
        self.loss_ratios = dict()
        self.loss_ratios['energy'] = self.args.loss_energy_ratio
        self.loss_ratios['forces'] = self.args.loss_forces_ratio
        self.loss_ratios['virial'] = self.args.loss_virial_ratio
        self.loss_ratios['charge'] = self.args.loss_charge_ratio
        self.loss_ratios['dipole'] = self.args.loss_dipole_ratio
#        self.loss_ratios['quadrupole'] = self.args.loss_quadrupole_ratio
        self.loss_unit = {
            'energy': 'kcal/mol',
            'forces': 'kcal/mol/Ang',
            'virial': 'kcal/mol',
            'charge': 'a.u.',
            'dipole': 'Debye',
#            'quadrupole': 'Debye*Ang'
        }
        self.nnfij_penalty_ratio = self.args.nnfij_penalty
        self.h_force_extra_ratio = self.args.h_force_extra_ratio
        self.decoul_dr_force_norm_ratio = self.args.decoul_dr_force_norm_ratio

        # Init dataset
        train_file_pretrain = os.path.join(self.args.datapath, self.args.datafile_complete_training)
        val_file_pretrain = os.path.join(self.args.datapath, self.args.datafile_complete_val)
        if self.args.num_epoch > self.args.num_pretrain_epoch:
            train_file_clean = os.path.join(self.args.datapath, self.args.datafile_clean_training)
            val_file_clean = os.path.join(self.args.datapath, self.args.datafile_clean_val)

            self.train_file_clean = train_file_clean
            self.val_file_clean = val_file_clean
            self.enable_finetune = True
        else:
            self.enable_finetune = False

        self.train_file_pretrain = train_file_pretrain
        self.val_file_pretrain = val_file_pretrain

        # Init model
        act_fn_map = {'ELU': nn.ELU(), 'CELU': nn.CELU(), 'GELU': nn.GELU(), 'SiLU': nn.SiLU(), 'Mish': nn.Mish(), 'Softplus': nn.Softplus()}
        assert self.args.act_fn in act_fn_map.keys(), f'Activation function {self.args.act_fn} not supported. Only support {list(act_fn_map.keys())}'
        
        nn_params = {
            'dim': self.args.emb_dim,
            'num_rbf': self.args.num_rbf,
            'rcut': self.args.rcut,
            'charge_ub': self.args.charge_ub,
            'act_fn': act_fn_map[self.args.act_fn],
            'charge_mlp_layers': self.args.charge_mlp_layers,
            'energy_mlp_layers': self.args.energy_mlp_layers,
        }
        gnn_params = {
            'n_layers': self.args.num_layers,
            'num_heads': self.args.num_heads,
            'act_fn': act_fn_map[self.args.act_fn]
        }
        coul_disp_params = {
            'coul_damping_beta': self.args.coul_damping_beta,
            'coul_damping_r0': self.args.coul_damping_r0,
            'disp_damping_beta': self.args.disp_damping_beta,
            'disp_cutoff': self.args.disp_cutoff,
            'disp_training': Dispersion[self.args.disp_training].name,
            'disp_simulation': Dispersion[self.args.disp_simulation].name
        }
        model_name_upper = str(self.args.model_name).upper()
        if model_name_upper == 'EGNN':
            from models.bamboo_egnn import BambooEGNN
            train_model_tmp = BambooEGNN(device = self.device,
                                dtype = self.dtype,
                                coul_disp_params = coul_disp_params,
                                nn_params = nn_params,
                                gnn_params = gnn_params)
        elif model_name_upper == 'ET':
            from models.bamboo_et import BambooET
            train_model_tmp = BambooET(device = self.device,
                                dtype = self.dtype,
                                coul_disp_params = coul_disp_params,
                                nn_params = nn_params,
                                gnn_params = gnn_params)
        elif model_name_upper == 'VisNet':
            from models.bamboo_visnet import BambooVisNet
            train_model_tmp = BambooVisNet(device =self.device,
                                dtype = self.dtype,
                                coul_disp_params = coul_disp_params,
                                nn_params = nn_params,
                                gnn_params = gnn_params)
        else:
            raise NotImplementedError('Supported model: EGNN, ET, VisNet')

        if self.world_size >= 2:
            train_model_tmp.to(self.device)
            self.model = DDP(train_model_tmp, device_ids=[self.rank])
        else:
            self.model = train_model_tmp

        # Init optimizer and scheduler
        optimizer_name = str(self.args.optimizer).upper()
        
        if optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        elif optimizer_name == 'ADAM':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif optimizer_name == 'ADAMAX':
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError('Supported optimizer: SGD, Adam, Adamax')
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.scheduler_gamma)
        
        # Load checkpoint from state dict
        self.start_epoch = 0
        self.ckpt_path = os.path.join(job_path, 'checkpoints')
        self.statedict_path = os.path.join(job_path, 'state_dicts')
        if os.path.exists(self.ckpt_path) and self.args.force_retrain:
            shutil.rmtree(self.ckpt_path)
        if os.path.exists(self.statedict_path) and self.args.force_retrain:
            shutil.rmtree(self.statedict_path)
        if os.path.exists(self.statedict_path) and len(os.listdir(self.statedict_path)) > 0:
            for i in range(self.args.num_epoch, 0, -1):
                filename = os.path.join(self.statedict_path, f'epoch_{i}_sd.pt')
                if os.path.isfile(filename):
                    self.logger.info(f'Loading checkpoint from {filename}')
                    map_location_dict = {"cuda:0": f"cuda:{self.rank}"}
                    checkpoint = torch.load(filename, map_location=map_location_dict)
                    self.train_model.load_state_dict(checkpoint["model_state_dict"])
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    self.start_epoch = checkpoint['epoch']
                    self.model.eval()
                    break
        elif not os.path.exists(self.statedict_path):
            os.makedirs(self.statedict_path, exist_ok=True)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path, exist_ok=True)

    def print_losses(self, losses, prefix=None):
        if not self.me:
            return
        if prefix is None:
            message = ''
        else:
            message = prefix
        for k in self.loss_ratios.keys():
            message += f' {k} {losses[k]:.4f} {self.loss_unit[k]},'
        self.logger.info(message)        

    def prepare_data(self, data: Dict[str, torch.Tensor], batch_size: int):
        world_size = self.world_size
        rank = self.rank
        # Simple data split for both DDP and original mode.
        data_size = len(data['total_charge'])
        if world_size <= 1:
            # Not in DDP, calculate indexes directly.
            steps = (data_size-1) // batch_size + 1

            start = [i * batch_size for i in range(steps)]
            end = [(i+1) * batch_size for i in range(steps-1)] + [data_size]
            return steps, start, end
        
        target_data_size = data_size // world_size
        steps = ((target_data_size) - 1) // batch_size + 1

        data_size_list = [target_data_size] * world_size
        for i in range(target_data_size * world_size, data_size):
            list_index = world_size - 1 - (i % world_size)
            data_size_list[list_index] += 1
        
        assert sum(data_size_list) == data_size, "Some Label not included."
        start_index = sum(data_size_list[:rank])
        end_index = start_index + data_size_list[rank]
        
        split_data(data, start_index, end_index, self.device)

        data_size = len(data['total_charge'])
        start = [i * batch_size for i in range(steps)]
        end = [(i+1) * batch_size for i in range(steps-1)] + [data_size]

        return steps, start, end

    def reduce_indicator(self, mapper: Dict[str, float]):
        if not self.ddp_mode:
            return
        keys = list(mapper.keys())
        keys.sort()
        vals = [float(mapper[key]) for key in keys]
        vals_tensor = torch.tensor(vals, device=self.device, dtype=torch.float64)
        dist.reduce(tensor=vals_tensor, dst=0)
        for i, key in enumerate(keys):
            mapper[key] = float(vals_tensor[i].item())

    def train_through_epochs(self, train_pt_file, val_pt_file, start_epoch, end_epoch, message, validation_before_training_flag=True):
        def closure():
            self.optimizer.zero_grad()
            mse, mae, penalty = self.model(batch_data)
            data_length = len(batch_data['total_charge'])
            msnnfij = penalty['msnnfij']
            mse_h_force = penalty['mse_h_force']
            decoul_dr_force = penalty['decoul_dr_force']
            loss = 0.
            for k in mse.keys():
                train_rmse[k].append(mse[k].item() * (data_length))
                train_mae[k].append(mae[k].item() * (data_length))
                loss += self.loss_ratios[k] * mse[k]

            loss += msnnfij * self.nnfij_penalty_ratio
            loss += mse_h_force * self.loss_ratios['forces'] * self.h_force_extra_ratio
            loss += decoul_dr_force * self.decoul_dr_force_norm_ratio

            loss.backward()
            return loss

        if start_epoch > end_epoch:
            return

        train_data_creator = BambooLoaderFactory(
            data_folder=train_pt_file,
            world_size=self.world_size,
            rank=self.rank,
            batch_size=self.args.train_bs,
            device=self.device,
            max_size=self.args.max_queue_size
        )

        val_data_creator = BambooLoaderFactory(
            data_folder=val_pt_file,
            world_size=self.world_size,
            rank=self.rank,
            batch_size=self.args.val_bs,
            device=self.device,
            max_size=self.args.max_queue_size
        )

        # global_train_data_size = len(train_data["total_charge"])
        # train_steps, train_start, train_end = self.prepare_data(train_data, self.args.train_bs)

        best_rmse = {k: 0. for k in list(self.loss_ratios.keys())+['weighted']}

        # Validation before training
        val_rmse = {k: [] for k in self.loss_ratios.keys()}
        val_mae = {k: [] for k in self.loss_ratios.keys()}
        mean_abs_charge = []
        ms_nnfij = []

        val_data = val_data_creator.create_loader()
        val_num_batches = val_data.num_batches
        global_val_data_size = val_data.total_samples
        val_data.start()
        for step in range(val_num_batches):
            batch_data = val_data.get()
            data_length = len(batch_data["total_charge"])
            mse, mae, penalty = self.train_model.get_loss(batch_data)
            val_data.task_done()
            macharge = penalty['macharge']
            msnnfij = penalty['msnnfij']
            mean_abs_charge.append(macharge.item())
            ms_nnfij.append(msnnfij.item())
            for k in mse.keys():
                val_rmse[k].append(mse[k].item() * data_length)
                val_mae[k].append(mae[k].item() * data_length)
        for k in val_rmse.keys():
            val_rmse[k] = sum(val_rmse[k]) / global_val_data_size
            val_mae[k] = sum(val_mae[k]) / global_val_data_size
        val_rmse['weighted'] = sum([self.loss_ratios[k] * val_rmse[k] for k in self.loss_ratios.keys()])

        self.reduce_indicator(val_rmse)
        self.reduce_indicator(val_mae)
        for k in val_rmse:
            val_rmse[k] = math.sqrt(val_rmse[k])

        if self.ddp_mode:
            dist.barrier()

        for epoch in range(int(start_epoch), int(end_epoch)):
            # Training
            train_rmse = {k: [] for k in self.loss_ratios.keys()}
            train_mae = {k: [] for k in self.loss_ratios.keys()}
            self.logger.info("[Train Start]")
            counter = 0
            train_data = train_data_creator.create_loader()
            num_batches = train_data.num_batches
            train_data.start()
            global_train_data_size = train_data.total_samples
            for step in range(num_batches):
                batch_data = train_data.get()
                self.optimizer.step(closure)
                train_data.task_done()
                counter += 1

            self.logger.info(f"[Train End] Total: {global_train_data_size} clusters. Rank: {counter} batches.")
            
            if self.ddp_mode:
                dist.barrier()

            for k in train_rmse.keys():
                train_rmse[k] = sum(train_rmse[k]) / global_train_data_size
                train_mae[k] = sum(train_mae[k]) / global_train_data_size
            train_rmse['weighted'] = sum([self.loss_ratios[k] * train_rmse[k] for k in self.loss_ratios.keys()])

            self.reduce_indicator(train_rmse)
            self.reduce_indicator(train_mae)
            for k in train_rmse:
                train_rmse[k] = math.sqrt(train_rmse[k])

            if self.me:
                self.print_losses(train_rmse, prefix=f"Training {message} {self.args.model_name} epoch {epoch+1}, RMSE:")
                self.print_losses(train_mae, prefix=f"Training {message} {self.args.model_name} epoch {epoch+1}, MAE:")

            self.scheduler.step()

            # Validation
            val_rmse = {k: [] for k in self.loss_ratios.keys()}
            val_mae = {k: [] for k in self.loss_ratios.keys()}
            mean_abs_charge = []
            ms_nnfij = []

            self.logger.info(f"[Val Start]")
            val_data = val_data_creator.create_loader()
            val_num_batches = val_data.num_batches
            global_val_data_size = val_data.total_samples
            val_data.start()
            for step in range(val_num_batches):
                batch_data = val_data.get()
                data_length = len(batch_data["total_charge"])
                mse, mae, penalty = self.train_model.get_loss(batch_data)
                macharge = penalty['macharge']
                msnnfij = penalty['msnnfij']
                mean_abs_charge.append(macharge.item())
                ms_nnfij.append(msnnfij.item())
                for k in mse.keys():
                    val_rmse[k].append(mse[k].item() * data_length)
                    val_mae[k].append(mae[k].item() * data_length)
            
            self.logger.info(f"[Val End]")

            for k in val_rmse.keys():
                val_rmse[k] = sum(val_rmse[k]) / global_val_data_size
                val_mae[k] = sum(val_mae[k]) / global_val_data_size

            val_rmse['weighted'] = sum([self.loss_ratios[k] * val_rmse[k] for k in self.loss_ratios.keys()])
            
            self.reduce_indicator(val_rmse)
            self.reduce_indicator(val_mae)
            for k in val_rmse:
                val_rmse[k] = math.sqrt(val_rmse[k])

            if self.me:
                self.print_losses(val_rmse, prefix=f"Validation {message} {self.args.model_name} epoch {epoch+1}, RMSE:")
                self.print_losses(val_mae, prefix=f"Validation {message} {self.args.model_name} epoch {epoch+1}, MAE:")
                self.logger.info(f"mean absolute charge: {np.mean(mean_abs_charge)}")
                self.logger.info(f"mean squared nnfij: {np.mean(ms_nnfij)}")

                for k in best_rmse.keys():
                    if val_rmse[k] < best_rmse[k]:
                        best_rmse[k] = val_rmse[k]
                        self.logger.info(f'Found {message} best {k} RMSE {best_rmse[k]:.4f} at epoch {epoch+1}')

                # Save ckpts every epoch. Statedict is used for retrain; checkpoint is used for inference
                statedict_filename = os.path.join(self.statedict_path, f"epoch_{epoch+1}_sd.pt")
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.train_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                    }, statedict_filename)
                self.logger.info(f'Statedict saved at {statedict_filename}')

                ckpt_filename = os.path.join(self.ckpt_path, f"epoch_{epoch+1}_loss_{int(1000*val_rmse['weighted'])}.pt")
                module = torch.jit.script(self.train_model)
                module.eval()
                module.save(ckpt_filename)
                self.logger.info(f'Checkpoint saved at {ckpt_filename}')
        
        return best_rmse

    def train(self):
        if self.start_epoch == 0:
            start_message = f"Training started from scratch."
        else:
            start_message = f"Training resumed from epoch {self.start_epoch}."
        self.logger.info(start_message)

        if self.start_epoch < self.args.num_warm_up_epoch:
            start_warm_up_epoch = self.start_epoch
        else:
            start_warm_up_epoch = self.args.num_warm_up_epoch

        best_rmse = self.train_through_epochs(self.train_file_clean, self.val_file_clean, start_warm_up_epoch, self.args.num_warm_up_epoch, 'warm_up_clean_data')

        warm_up_message = f"Training {self.args.model_name} warm up finished."
        self.logger.info(warm_up_message)
        
        if self.start_epoch + self.args.num_warm_up_epoch < self.args.num_pretrain_epoch:
            start_pretrain_epoch = self.start_epoch + self.args.num_warm_up_epoch
        else:
            start_pretrain_epoch = self.args.num_pretrain_epoch

        best_rmse = self.train_through_epochs(self.train_file_pretrain, self.val_file_pretrain, start_pretrain_epoch, self.args.num_pretrain_epoch, 'complete_data')
        
        switch_message = f"Training {self.args.model_name} dataset switched."
        for k in self.loss_ratios.keys():
            switch_message += f" Best {k} RMSE {best_rmse[k]:4f}. "
        switch_message += f" Best weighted RMSE {best_rmse['weighted']:.4f}."
        self.logger.info(switch_message)

        if not self.enable_finetune:
            return

        if self.start_epoch >= self.args.num_pretrain_epoch:
            start_clean_epoch = self.start_epoch
        else:
            start_clean_epoch = self.args.num_pretrain_epoch

        best_rmse = self.train_through_epochs(self.train_file_clean, self.val_file_clean, start_clean_epoch, self.args.num_epoch, 'clean_data')

        finish_message = f"Training {self.args.model_name} finished."
        for k in self.loss_ratios.keys():
            finish_message += f" Best {k} RMSE {best_rmse[k]:4f}. "
        finish_message += f" Best weighted RMSE {best_rmse['weighted']:.4f}."
        self.logger.info(finish_message)

    @property
    def me(self) -> bool:
        # Only process 0 does most IO operation.
        return bool(0 == self.rank)

    @property
    def ddp_mode(self) -> bool:
        return bool(self.world_size >= 2)

    @property 
    def train_model(self) -> 'BambooET':
        if self.ddp_mode:
            return self.model.module  # Original model for DDP wrapper
        else:
            return self.model  # No wrapped model


def run(rank, world_size, args, init_method):
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_method}",
        rank=rank,
        world_size=world_size)
    
    bamboo_trainer = BambooTrainer(args, rank=rank, world_size=world_size)
    bamboo_trainer.train()
    
    dist.destroy_process_group()


def spwan_job(demo_fn, world_size: int, args: argparse.Namespace, init_method: str):
    mp.spawn(demo_fn,
             args=(world_size, args, init_method),
             nprocs=world_size,
             join=True)


def train_run(config_path: Optional[str] = None):
    args = get_parser(config_path)

    init_method = init_method_file(args.job_name)

    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        bamboo_trainer = BambooTrainer(args)
        bamboo_trainer.train()
    else:
        # Make sure init_method file is cleaned.
        if os.path.exists(init_method):
            os.remove(init_method)
        spwan_job(run, n_gpus, args, init_method)


if __name__ == "__main__":
    train_run()
    
