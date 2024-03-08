# -----  BAMBOO: Bytedance AI Molecular Booster -----
# Copyright 2022-2024 Bytedance Ltd. and/or its affiliates 

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import argparse
import json
import logging
import math
import os
from datetime import datetime
from random import shuffle
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from models.bamboo_get import BambooGET
from utils.batchify import batchify
from utils.log_helper import create_logger
from utils.path import DATA_PATH, TRAIN_PATH


def get_parser():
    parser = argparse.ArgumentParser(description='Arguments for bamboo model training')

    # general arguments
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--job_name', default='default', type=str)
    parser.add_argument('--data_training', default='train_data.pt')
    parser.add_argument('--data_validation', default='val_data.pt')
    parser.add_argument('--random_seed', default=42, type=int)

    # training arguments
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--num_epoch', default=750, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--scheduler_gamma', default=0.99, type=float)
    parser.add_argument('--loss_charge_ratio', default=10.0, type=float)
    parser.add_argument('--loss_dipole_ratio', default=10.0, type=float)
    parser.add_argument('--loss_energy_ratio', default=0.01, type=float)
    parser.add_argument('--loss_forces_ratio', default=0.3, type=float)
    parser.add_argument('--loss_virial_ratio', default=0.01, type=float)
    parser.add_argument('--charge_ub', default=2.0, type=float)
    parser.add_argument('--qeq_force_regularizer', default=300.0, type=float)

    # model arguments
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_rbf', type=int, default=32)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--rcut', type=float, default=5.0)
    parser.add_argument('--coul_damping_beta', type=float, default=18.7)
    parser.add_argument('--coul_damping_r0', type=float, default=2.2)
    parser.add_argument('--disp_damping_beta', type=float, default=23.0)
    parser.add_argument('--disp_cutoff', type=float, default=10.0)
    parser.add_argument('--energy_mlp_layers', type=int, default=2)
    parser.add_argument('--charge_mlp_layers', type=int, default=2)
    
    args = parser.parse_args()

    # if config file is provided, read args from config file
    if os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            config_args = json.load(f)
            for key, value in config_args.items():
                if value is None:
                    continue
                setattr(args, key, value)
    return args

class BambooTrainer():
    """
    Basic trainer for bamboo
    """
    def __init__(self, args):
        self.args = args

        # Init log and checkpoint directory
        job_path = os.path.join(TRAIN_PATH, self.args.job_name).lower()
        if not os.path.exists(job_path):
            os.makedirs(job_path, exist_ok=True)
        train_log_path = os.path.join(job_path, 'train_logs')
        if not os.path.exists(train_log_path):
            os.makedirs(train_log_path, exist_ok=True)     
        
        log_file = os.path.join(train_log_path, f"train_{datetime.now().strftime('%m%d%H%M')}.log")
        self.logger = create_logger(name="TRAIN", log_file=log_file)
        self.logger.info(f"Initializing.")
        for k_args, v_args in vars(args).items():
            self.logger.info(f'{k_args} = {v_args}')

        ckpt_path = os.path.join(job_path, 'checkpoints')   
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
        self.ckpt_path = ckpt_path

        # Init device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda")
            self.logger.info(f'device = cuda')
        else:
            raise RuntimeError("Cannot find CUDA device.")

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
        self.loss_unit = {
            'energy': 'kcal/mol',
            'forces': 'kcal/mol/Ang',
            'virial': 'kcal/mol',
            'charge': 'a.u.',
            'dipole': 'Debye',
        }
        self.qeq_force_regularizer = self.args.qeq_force_regularizer

        # Init dataset
        self.train_data = torch.load(os.path.join(DATA_PATH, self.args.data_training), map_location='cpu')
        self.val_data = torch.load(os.path.join(DATA_PATH, self.args.data_validation), map_location='cpu')

        # Init model        
        nn_params = {
            'dim': self.args.emb_dim,
            'num_rbf': self.args.num_rbf,
            'rcut': self.args.rcut,
            'charge_ub': self.args.charge_ub,
            'act_fn': nn.SiLU(),
            'charge_mlp_layers': self.args.charge_mlp_layers,
            'energy_mlp_layers': self.args.energy_mlp_layers,
        }
        gnn_params = {
            'n_layers': self.args.num_layers,
            'num_heads': self.args.num_heads,
            'act_fn': nn.SiLU(),
        }
        coul_disp_params = {
            'coul_damping_beta': self.args.coul_damping_beta,
            'coul_damping_r0': self.args.coul_damping_r0,
            'disp_damping_beta': self.args.disp_damping_beta,
            'disp_cutoff': self.args.disp_cutoff,
        }
        self.model = BambooGET(device = self.device,
                                coul_disp_params = coul_disp_params,
                                nn_params = nn_params,
                                gnn_params = gnn_params)

        # Init optimizer and scheduler
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.scheduler_gamma)

    def print_losses(self, losses: Dict[str, float], prefix: str):
        message = prefix
        for k in self.loss_ratios.keys():
            message += f' {k} {losses[k]:.4f} {self.loss_unit[k]},'
        self.logger.info(message)        

    def prepare_data(self, data: Dict[str, torch.Tensor], batch_size: int):
        # Simple data split 
        data_size = len(data['total_charge'])
        steps = (data_size-1) // batch_size + 1
        start = [i * batch_size for i in range(steps)]
        end = [(i+1) * batch_size for i in range(steps-1)] + [data_size]
        return steps, start, end, data_size

    def validate_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.logger.info(f"[Val Start] Epoch {epoch+1}.")
        val_rmse = {k: [] for k in self.loss_ratios.keys()}
        val_mae = {k: [] for k in self.loss_ratios.keys()}
        steps, start, end, data_size = self.prepare_data(self.val_data, self.args.val_batch_size)

        for step in range(steps):
            batch_data = batchify(self.val_data, start[step], end[step], device=self.device)
            data_length = len(batch_data["total_charge"])
            mse, mae, _ = self.model.get_loss(batch_data)
            for k in mse.keys():
                val_rmse[k].append(mse[k].item() * data_length)
                val_mae[k].append(mae[k].item() * data_length)

        for k in val_rmse.keys():
            val_rmse[k] = sum(val_rmse[k]) / data_size
            val_mae[k] = sum(val_mae[k]) / data_size
        val_rmse['weighted'] = sum([self.loss_ratios[k] * val_rmse[k] for k in self.loss_ratios.keys()])
        for k in val_rmse:
            val_rmse[k] = math.sqrt(val_rmse[k])

        self.logger.info(f"[Val End] Epoch {epoch+1}, Total: {data_size} clusters, {steps} batches.")
        self.print_losses(val_rmse, prefix=f"[Val RMSE] Epoch {epoch+1}, ")
        self.print_losses(val_mae, prefix=f"[Val MAE] Epoch {epoch+1}, ")
        return val_rmse

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        def closure():
            self.optimizer.zero_grad()
            mse, mae, penalty = self.model.get_loss(batch_data)
            data_length = len(batch_data['total_charge'])
            qeq_force = penalty['qeq_force']
            loss = 0.
            for k in mse.keys():
                train_rmse[k].append(mse[k].item() * data_length)
                train_mae[k].append(mae[k].item() * data_length)
                loss += self.loss_ratios[k] * mse[k]
            loss += qeq_force * self.qeq_force_regularizer
            loss.backward()
            return loss

        self.logger.info(f"[Train Start] Epoch {epoch+1}.")
        train_rmse = {k: [] for k in self.loss_ratios.keys()}
        train_mae = {k: [] for k in self.loss_ratios.keys()}
        steps, start, end, data_size = self.prepare_data(self.train_data, self.args.train_batch_size)
        steps_shuffle = list(range(steps))
        shuffle(steps_shuffle)

        for step in steps_shuffle:
            batch_data = batchify(self.train_data, start[step], end[step], device=self.device)
            self.optimizer.step(closure)

        for k in train_rmse.keys():
            train_rmse[k] = sum(train_rmse[k]) / data_size
            train_mae[k] = sum(train_mae[k]) / data_size
        train_rmse['weighted'] = sum([self.loss_ratios[k] * train_rmse[k] for k in self.loss_ratios.keys()])
        for k in train_rmse:
            train_rmse[k] = math.sqrt(train_rmse[k])

        self.logger.info(f"[Train End] Epoch {epoch+1}, Total: {data_size} clusters, {steps} batches.")
        self.print_losses(train_rmse, prefix=f"[Train RMSE] Epoch {epoch+1}, ")
        self.print_losses(train_mae, prefix=f"[Train MAE] Epoch {epoch+1}, ")

    def train(self, epochs: int):
        val_rmse = self.validate_one_epoch(epoch = -1)
        best_rmse = val_rmse

        for epoch in range(epochs):
            self.train_one_epoch(epoch = epoch)
            self.scheduler.step()
            val_rmse = self.validate_one_epoch(epoch = epoch)
            if val_rmse['weighted'] < best_rmse['weighted']:
                self.logger.info(f"Found best weighted RMSE {best_rmse['weighted']:.4f} at epoch {epoch+1}")

            # Save ckpts every epoch
            ckpt_filename = os.path.join(self.ckpt_path, f"epoch_{epoch+1}_loss_{int(1000*val_rmse['weighted'])}.pt")
            module = torch.jit.script(self.model)
            module.eval()
            module.save(ckpt_filename)
            self.logger.info(f'Epoch {epoch+1}, checkpoint saved at {ckpt_filename}')
        
        finish_message = f"Training finished."
        for k in self.loss_ratios.keys():
            finish_message += f" Best {k} RMSE {best_rmse[k]:4f}. "
        finish_message += f" Best weighted RMSE {best_rmse['weighted']:.4f}."
        self.logger.info(finish_message)

def main():
    args = get_parser()
    bamboo_trainer = BambooTrainer(args)
    bamboo_trainer.train(args.num_epoch)

if __name__ == "__main__":
    main()