import argparse
import json
import os
import random
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from models.bamboo_et import BambooET
from utils.get_batch import get_batch, split_data
from utils.log_helper import create_logger
from utils.rejit import convert


def get_parser(config_path: Optional[str] = None):
    # Create the parser
    parser = argparse.ArgumentParser(description="Parameters for the training script")

    # Required arguments
    parser.add_argument("--work_dir", type=str, default=".", help="Path to the work directory")
    parser.add_argument("--train_cluster", type=str, default="train_pyscf_svpd_b3lyp_more_data_clean_loose_double_10192023.pt", help="Training cluster")
    parser.add_argument("--val_cluster", type=str, default="val_pyscf_svpd_b3lyp_more_data_clean_loose_double_10192023.pt", help="Validation cluster")
    parser.add_argument("--data_path", type=str, default="/mnt/bn/ai4s-hl/bamboo/pyscf_data/data", help="Path to the data")

    parser.add_argument("--frame_folders", nargs="*", type=str, default=[], help="List of frame folders")
    parser.add_argument("--models", nargs="*", type=str, default=[], help="List of models to be ensembled.")
    parser.add_argument("--uncertainty_jobs", nargs="*", type=str, default=[], help="List of uncertainty jobs.")
    parser.add_argument("--ensemble_model", type=str, default=None, help="Specify the model to be ensembled.")

    # Optional arguments
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--scheduler_count", type=int, default=10, help="Scheduler count")
    
    parser.add_argument("--energy_ratio", type=float, default=0.3, help="Energy ratio")
    parser.add_argument("--force_ratio", type=float, default=1.0, help="Force ratio")
    parser.add_argument("--virial_ratio", type=float, default=0.1, help="Virial ratio")
    parser.add_argument("--dipole_ratio", type=float, default=3.0, help="Dipole ratio")
    
    parser.add_argument("--bulk_energy_ratio", type=float, default=0.01, help="bulk energy ratio")
    parser.add_argument("--bulk_force_ratio", type=float, default=3.0, help="bulk force ratio")
    parser.add_argument("--bulk_virial_ratio", type=float, default=0.001, help="bulk virial ratio")
    
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--max_frame_per_mixture", type=int, default=480, help="Max frame per mixture.")
    parser.add_argument("--frame_val_interval", type=int, default=3, help="Validation ratio")

    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--scheduler_gamma", type=float, default=0.99, help="Scheduler gamma")

    parser.add_argument("--val_interval", type=int, default=10, help="val log step interval.")    
    parser.add_argument("--config", type=str, help="Path to a configuration file in JSON format.")

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


class EnsembleFinetune:
    def __init__(self, args) -> None:
        self.args = args

        # Check required arguments.
        assert bool(args.frame_folders), "uncertainty_jobs must be provided"
        assert bool(args.models), "models must be provided"
        assert bool(args.uncertainty_jobs), "uncertainty_jobs must be provided"
        assert len(args.models) == len(args.uncertainty_jobs), "Number of models and uncertainty_jobs must be equal."

        self.work_dir = args.work_dir

        self.frames_output = os.path.join(self.work_dir, "frame")
        self.checkpoint_output = os.path.join(self.work_dir, "checkpoints")
        self.log_output = os.path.join(self.work_dir, "logs")

        make_dirs = [self.frames_output, self.checkpoint_output, self.log_output]
        for dir_tmp in make_dirs:
            os.makedirs(dir_tmp, exist_ok=True)
        
        log_file = os.path.join(self.log_output, "ensemble.log")
        self.logger = create_logger(name="Ensemble", log_file=log_file)
        
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, list):
                val = '\n\t\t\t' + '\n\t\t\t'.join(map(str, val))
                self.logger.info(f"{arg} = {val}")
            else:
                self.logger.info(f"{arg} = {val}")

        self.train_cluster = os.path.join(args.data_path, args.train_cluster)
        self.val_cluster = os.path.join(args.data_path, args.val_cluster)
        self._train_cluster_data = None
        self._val_cluster_data = None
        
        self.loss_ratios = {
            'energy': args.energy_ratio,
            'forces': args.force_ratio,
            'virial': args.virial_ratio,
#            'dipole': args.dipole_ratio,
        }

        self.bulk_energy_ratio = args.bulk_energy_ratio
        self.bulk_force_ratio = args.bulk_force_ratio
        self.bulk_virial_ratio = args.bulk_virial_ratio

        self.batch_size = args.batch_size  # Share batch size for both val and train.
        
        self.lr = args.lr
        self.scheduler_gamma = args.scheduler_gamma
        
        self.val_ratio = args.val_ratio
        self.scheduler_count = args.scheduler_count
        self.val_interval = args.val_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = args.epochs
        self.max_frame_per_mixture = args.max_frame_per_mixture
        self.frame_val_interval = args.frame_val_interval

        self.frame_folders = args.frame_folders
        # Check if all frames_dir exist
        for dir_tmp in self.frame_folders:
            if not os.path.isdir(dir_tmp):
                raise NotADirectoryError(f"frames_dir {dir_tmp} not found")
        
        # Assume the number of models is small, load all the models in memory.
        self.logger.info("Start load all models.")
        self.modules = {}
        self.checkpoints = {}
        for index in range(len(args.models)):
            job = args.uncertainty_jobs[index]
            checkpoint = args.models[index]

            self.checkpoints[job] = checkpoint

            # Only load model once.
            if checkpoint in self.modules:
                continue
            
            module = torch.jit.load(checkpoint, map_location=self.device)
            self.modules[checkpoint] = module
            
        if args.ensemble_model is None:
            self.logger.info(f"Ensemble model not specified, use the first model: {args.models[0]}")
            self.ensemble_model = args.models[0]
        else:
            self.ensemble_model = args.ensemble_model
            if self.ensemble_model not in self.modules:
                raise ValueError(f"Ensemble model {self.ensemble_model} not found in models.")

        self.logger.info(f"Number of models: {len(self.modules)}")
        
        self._cached_pt = {}
        self.uncertainty_train = []
        self.uncertainty_val = []
        self.split_data_flag = False

    @property
    def train_cluster_data(self) -> Dict[str, torch.Tensor]:
        if self._train_cluster_data is None:
            self._train_cluster_data = torch.load(self.train_cluster, map_location="cpu")
            self.gpu_memory_usage("Load train cluster data")
        return self._train_cluster_data
    
    @property
    def val_cluster_data(self) -> Dict[str, torch.Tensor]:
        if self._val_cluster_data is None:
            self._val_cluster_data = torch.load(self.val_cluster, map_location="cpu")
            self.gpu_memory_usage("Load val cluster data")
        return self._val_cluster_data

    def gpu_memory_usage(self, info: str):
        self.logger.info(f"{info} GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")

    def uncertainty(self):
        self.logger.info("Start uncertainty quantification")

        # Data and modules are ready, start inference.
        config_file = os.path.join(self.frames_output, "config.json")
        if os.path.isfile(config_file):
            configs = json.load(open(config_file))
        else:
            configs = {}

        data_path = []
        # Scan the frames folder and find all the data.
        # The code will scan recuivesively.
        for dir_tmp in self.frame_folders:
            for root, _, files in os.walk(dir_tmp):
                for file in files:
                    if file.endswith(".pt"):
                        data_path.append(os.path.join(root, file))
        self.logger.info(f"Number of data: {len(data_path)}")

        uncertainty_count = {}

        def check_uncertainty(mixture_name: str) -> bool:
            return bool(uncertainty_count.get(mixture_name, 0) < self.max_frame_per_mixture)

        def add_frame_data(mixture_name: str, output_file: str):
            if mixture_name not in uncertainty_count:
                uncertainty_count[mixture_name] = 0
            uncertainty_count[mixture_name] += 1
            
            if uncertainty_count[mixture_name] % self.frame_val_interval:
                self.uncertainty_train.append(output_file)
            else:
                self.uncertainty_val.append(output_file)

        self.logger.info("Start inference for uncertainty quantification")
        for data in data_path:
            if data in configs:
                config_data = configs[data]
                if not isinstance(config_data, dict):
                    config_data = {}
                current_mixture_name = config_data.get("mixture_name")
                output_file = config_data.get("output_file")
                if isinstance(current_mixture_name, str) \
                        and isinstance(output_file, str) \
                        and check_uncertainty(current_mixture_name):
                    add_frame_data(current_mixture_name, output_file)
                    self.logger.info(f"Skip {output_file}.")
                    continue
                else:
                    self.logger.info(f"Invalid config for {data}, reload.")
            
            # Load data by normal way.
            single_data: Dict[str, torch.Tensor] = torch.load(data, map_location=self.device)
            current_mixture_name = single_data["config"]["mixture_name"]
            if not check_uncertainty(current_mixture_name):
                self.logger.info(f"Max frame per mixture reached for {current_mixture_name}.")
                continue

            ensemble_pred = {
                'energy': [],
                'forces': [],
                'virial': [],
            }

            # Collect predictions from all modules.
            for module in self.modules.values():
                pred = module.forward(single_data["inputs"])
                
                ensemble_pred['energy'].append(torch.flatten(pred['pred_energy'].detach()))
                ensemble_pred['forces'].append(torch.flatten(pred['pred_forces'].detach()))
                ensemble_pred['virial'].append(torch.flatten(pred['pred_virial'].detach()))
                del pred
            ensemble_results = {}

            # Compute ensemble statistics.
            for k, values in ensemble_pred.items():
                values_tensor = torch.stack(values, dim=0)
                
                mean_tmp = torch.mean(values_tensor, dim=0).detach().cpu().tolist()
                std_tmp = torch.std(values_tensor, dim=0).detach().cpu().tolist()
                if k in ["force", "virial"]:
                    mean_tmp = [mean_tmp[i:i+3] for i in range(0, len(mean_tmp), 3)]
                    std_tmp = [std_tmp[i:i+3] for i in range(0, len(std_tmp), 3)]

                ensemble_results[k] = {
                    "mean": mean_tmp,
                    "std": std_tmp
                }

            ensemble_results.update(single_data)

            # Save file {frames}/a/b/c.pt to {output}/c_{index}.pt
            file_base_name = os.path.splitext(os.path.basename(data))[0]
            index = 0

            def get_output_file(index: int) -> str:
                return os.path.join(self.frames_output, f"{file_base_name}_{index}.pt")

            output_file = get_output_file(index)
            while os.path.isfile(output_file):
                index += 1
                output_file = get_output_file(index)

            configs[data] = {"mixture_name": current_mixture_name, "data": output_file}
            self.logger.info(f"Source: {data}, output: {output_file}")
            torch.save(ensemble_results, output_file)
            add_frame_data(current_mixture_name, output_file)

            del single_data
        # Save configs
        self.logger.info(f"Save config to {config_file}. The config saves the match between input and output pts.")
        json.dump(configs, open(config_file, "w"), indent=4)
        self.logger.info("Uncertainty quantification finished")

    def cluster_validation(self, model: BambooET, cluster: Dict[str, torch.Tensor]) -> Dict[str, float]:
        keys = ['energy', 'forces', 'virial', 'dipole']
        val_rmse = {k: [] for k in keys}
        val_data_size = len(cluster['total_charge'])
        total_step = val_data_size // self.batch_size
        for step in range(total_step):
            batch_data = get_batch(cluster, step*self.batch_size, (step+1)*self.batch_size, device=self.device)
            mse = model.get_mse_loss(batch_data)
            for k in keys:
                val_rmse[k].append(mse[k].item() * self.batch_size)
        total_val_rmse = {}
        total_val_rmse["cluster_energy_rmse"] = np.sqrt(sum(val_rmse["energy"]) / self.batch_size / total_step)
        total_val_rmse["cluster_force_rmse"] = np.sqrt(sum(val_rmse["forces"]) / self.batch_size / total_step)
        total_val_rmse["cluster_virial_rmse"] = np.sqrt(sum(val_rmse["virial"]) / self.batch_size / total_step)
        total_val_rmse["cluster_dipole_rmse"] = np.sqrt(sum(val_rmse["dipole"]) / self.batch_size / total_step)
        return total_val_rmse

    def bulk_validation(self, model: BambooET, file: List[str]) -> Dict[str, float]:
        val_forces_rmse = []
        val_energy_rmse = []
        val_virial_rmse = []

        for f in file:
            single_data_pt = self.load_pt(f)
            inputs = {k: v.to(self.device) for k, v in single_data_pt["inputs"].items()}

            pred = model.forward(inputs)
            mse_forces = torch.mean(torch.square(pred['pred_forces'].flatten() - torch.tensor(single_data_pt['forces']['mean'], device=self.device).flatten()))
            mse_energy = torch.mean(torch.square(pred['pred_energy'].flatten() - torch.tensor(single_data_pt['energy']['mean'], device=self.device).flatten()))                
            mse_virial = torch.mean(torch.square(pred['pred_virial'].flatten() - torch.tensor(single_data_pt['virial']['mean'], device=self.device).flatten()))
                
            val_forces_rmse.append(mse_forces.item())
            val_energy_rmse.append(mse_energy.item())
            val_virial_rmse.append(mse_virial.item())
            del single_data_pt, inputs, pred
        
        result = {}
        result['force_rmse'] = np.sqrt(np.mean(val_forces_rmse))
        result['energy_rmse'] = np.sqrt(np.mean(val_energy_rmse))
        result['virial_rmse'] = np.sqrt(np.mean(val_virial_rmse))
        return result

    def finetune(self):
        if not self.uncertainty_train:
            raise ValueError("No uncertainty_train pts.")
        
        # Save the finetuned model to {save_dir}/finetuned.pt
        self.logger.info(f"Start finetuning for model: {self.ensemble_model}")
        checkpoint_path = os.path.join(self.checkpoint_output, f"ensemble.pt")
        if os.path.isfile(checkpoint_path):
            self.logger.info(f"checkpoint already exists: {checkpoint_path}")
            return
        self.gpu_memory_usage("Before finetuning.")

        module = self.modules[self.ensemble_model]
        model = convert(module, device=self.device)
        model.train()
        self.gpu_memory_usage("After converting to trainable model.")

        previous_cluster_count = len(self.train_cluster_data["cell"])
        target_data_size = self.batch_size * len(self.uncertainty_train)
        if target_data_size < previous_cluster_count:
            split_data(self.train_cluster_data, start_index=0, end_index=target_data_size, device=self.device)
            after_cluster_count = len(self.train_cluster_data["cell"])
            self.logger.info(f"Previous cluster: {previous_cluster_count}, Current: {after_cluster_count}")
        else:
            self.logger.info(f"Not enough data for finetuning. Original: {previous_cluster_count}, Target: {target_data_size}")

        training_curve = {
            'epoch':[],
            'force_rmse':[],
            'energy_rmse':[],
            'virial_rmse':[],
            'cluster_force_rmse':[],
            'cluster_energy_rmse':[],
            'cluster_virial_rmse':[],
            'cluster_dipole_rmse':[]
        }

        # Call logger to log the training_curve
        def log_train_curve() -> None:
            log_string = ""
            for k, v in training_curve.items():
                if isinstance(v[-1], float):
                    log_string += f"{k}: {v[-1]:.4f} "
                else:
                    log_string += f"{k}: {v[-1]} "
            self.logger.info(log_string)

        def add_train_curve(epoch: int, bulk: Dict[str, float], cluster: Dict[str, float]) -> None:
            training_curve['epoch'].append(epoch)
            training_curve['force_rmse'].append(bulk['force_rmse'])
            training_curve['energy_rmse'].append(bulk['energy_rmse'])
            training_curve['virial_rmse'].append(bulk['virial_rmse'])
            training_curve['cluster_force_rmse'].append(cluster['cluster_force_rmse'])
            training_curve['cluster_energy_rmse'].append(cluster['cluster_energy_rmse'])
            training_curve['cluster_virial_rmse'].append(cluster['cluster_virial_rmse'])
            training_curve['cluster_dipole_rmse'].append(cluster['cluster_dipole_rmse'])

        # Initialize the training process
        energy_mlp_params = list(model.energy_mlp.parameters())
        optimizer = torch.optim.Adam(energy_mlp_params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler_gamma)

        # Evaluate the initial model.
        self.gpu_memory_usage("First evaluate.")
        bulk_val_rmse = self.bulk_validation(model, self.uncertainty_val)
        cluster_val_rmse = self.cluster_validation(model, self.val_cluster_data)

        add_train_curve(-1, bulk_val_rmse, cluster_val_rmse)
        log_train_curve()

        # Start finetune training
        total_cluster_data = len(self.train_cluster_data['total_charge'])
        cluster_batch_num = total_cluster_data // self.batch_size
        cluster_random_index = list(range(cluster_batch_num))
        random.shuffle(cluster_random_index)
        n_cluster_index = 0
        
        self.gpu_memory_usage("Start training.")

        for epoch in range(self.epochs):
            train_order = list(range(len(self.uncertainty_train)))
            random.shuffle(train_order)
            self.gpu_memory_usage(f"Epoch: {epoch} start.")
            for idx in train_order:
                file = self.uncertainty_train[idx]
                single_data_pt = torch.load(file, map_location=self.device)
                inputs = single_data_pt["inputs"]

                natoms = len(inputs['atom_types'])
                #training on bulk traj data
                optimizer.zero_grad()
                pred = model.forward(inputs)
                mse_forces = torch.mean(torch.square(pred['pred_forces'].flatten() - torch.tensor(single_data_pt['forces']['mean'], device=self.device).flatten()))
                mse_energy = torch.mean(torch.square(pred['pred_energy'].flatten() - torch.tensor(single_data_pt['energy']['mean'], device=self.device).flatten()))
                mse_virial = torch.mean(torch.square(pred['pred_virial'].flatten() - torch.tensor(single_data_pt['virial']['mean'], device=self.device).flatten()))

                bulk_loss: torch.Tensor = mse_forces * self.bulk_force_ratio \
                        + mse_energy / natoms * self.bulk_energy_ratio \
                        + mse_virial / natoms * self.bulk_virial_ratio
                
                bulk_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                del inputs, pred, single_data_pt

                cluster_loss = torch.tensor(0.0, device=self.device)
                #training on cluster data

                optimizer.zero_grad()
                n_cluster_index = (n_cluster_index + 1) % cluster_batch_num
                start = cluster_random_index[n_cluster_index]
                end = start + self.batch_size
                batch_data = get_batch(self.train_cluster_data, start, end, device=self.device)
                mse = model.get_mse_loss(batch_data)
                for k in self.loss_ratios.keys():
                    cluster_loss += self.loss_ratios[k] * mse[k]
                del batch_data
                cluster_loss.backward()
                optimizer.step()
            
            scheduler.step()
            self.gpu_memory_usage("Complete one epoch.")
            if (not epoch % self.val_interval) or epoch == self.epochs-1:
                # After each epoch, evaluate the model.
                bulk_val_rmse = self.bulk_validation(model, self.uncertainty_val)
                self.gpu_memory_usage("Complete bulk validation.")

                cluster_val_rmse = self.cluster_validation(model, self.val_cluster_data)
                self.gpu_memory_usage("Complete cluster validation.")

                add_train_curve(epoch, bulk_val_rmse, cluster_val_rmse)
                log_train_curve()

        # Save finetuned model.
        module = torch.jit.script(model)
        torch.jit.save(module, checkpoint_path)
        self.logger.info(f"Finetuned model saved to {checkpoint_path}")

        # Save training curve
        curve_path = os.path.join(self.log_output, "training_curve.csv")
        df = pd.DataFrame(training_curve)
        df.to_csv(curve_path, index=False)

    def load_pt(self, file: str):
        # Load the data onto CPU if not already loaded
        if file not in self._cached_pt:
            self._cached_pt[file] = torch.load(file, map_location='cpu')

        # Transfer the data to the GPU when needed
        return self._cached_pt[file]


def ensemble_run(config_path: Optional[str] = None):
    args = get_parser(config_path)
    ensemble_finetune = EnsembleFinetune(args)
    ensemble_finetune.uncertainty()
    ensemble_finetune.finetune()


if __name__ == "__main__":
    # For local test.
    ensemble_run()
