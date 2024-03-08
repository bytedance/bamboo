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
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from models.bamboo_get import BambooGET
from utils.batchify import batchify
from utils.log_helper import create_logger
from utils.path import DATA_PATH, ENSEMBLE_PATH
from utils.rejit import convert


def get_parser():
    # Create the parser
    parser = argparse.ArgumentParser(description="Arguments for bamboo model ensembling.")

    # Required arguments
    parser.add_argument('--config', default='', type=str, help="Path to a configuration file in JSON format.")
    parser.add_argument('--job_name', default='default', type=str)
    
    # Training and validation data paths
    parser.add_argument("--training_data_path", type=str, default="train_data.pt", help="Path to the training data file.")
    parser.add_argument("--validation_data_path", type=str, default="val_data.pt", help="Path to the validation data file.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training and validation.")

    # Data sources and model configuration
    parser.add_argument("--models", nargs="*", type=str, default=[], help="Paths to models for uncertainty calculation.")
    parser.add_argument("--frame_directories", nargs="*", type=str, default=[], help="Directories containing frame data.")
    parser.add_argument("--ensemble_model", type=str, default=None, help="Path to the model used for ensemble predictions.")

    # Training parameters
    parser.add_argument("--validation_split_ratio", type=float, default=0.1, help="Fraction of data to use for validation.")
    parser.add_argument("--lr", type=float, default=1e-6, help="Initial learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.99, help="Learning rate decay factor per epoch.")
    parser.add_argument("--validation_interval", type=int, default=10, help="Interval (in epochs) between validations.")

    # Data property weighting
    parser.add_argument("--energy_ratio", type=float, default=0.3, help="Weight of energy predictions in the loss function.")
    parser.add_argument("--force_ratio", type=float, default=1.0, help="Weight of force predictions in the loss function.")
    parser.add_argument("--virial_ratio", type=float, default=0.1, help="Weight of virial predictions in the loss function.")
    parser.add_argument("--bulk_energy_ratio", type=float, default=0.01, help="Weight of bulk energy predictions in the loss function.")
    parser.add_argument("--bulk_force_ratio", type=float, default=3.0, help="Weight of bulk force predictions in the loss function.")
    parser.add_argument("--bulk_virial_ratio", type=float, default=0.01, help="Weight of bulk virial predictions in the loss function.")

    # Additional training settings
    parser.add_argument("--max_frames_per_mixture", type=int, default=960, help="Maximum number of frames per mixture.")
    parser.add_argument("--frame_validation_interval", type=int, default=3, help="Interval for frame-level validation checks.")

    args = parser.parse_args()

    # Load configuration from a JSON file if specified and the file exists
    if os.path.isfile(args.config):
        with open(args.config, 'r') as config_file:
            config_from_file = json.load(config_file)
    
        # Update the command line arguments with values from the JSON configuration
        for key, value in config_from_file.items():
            # Skip updating args with None values from the configuration file
            if value is not None:
                setattr(args, key, value)

    return args


class DistillationEnsemble:
    def __init__(self, args) -> None:
        self.args = args

        # Validate required arguments
        if not self.args.frame_directories:
            raise ValueError("Frame folders must be provided.")

        if not self.args.models:
            raise ValueError("Models must be provided.")

        if not self.args.ensemble_model:
            raise ValueError("Uncertainty jobs must be provided.")

        self.work_dir = os.path.join(ENSEMBLE_PATH, self.args.job_name)

        self.frames_output = os.path.join(self.work_dir, "frame")
        self.checkpoint_output = os.path.join(self.work_dir, "checkpoints")
        self.log_output = os.path.join(self.work_dir, "logs")

        make_dirs = [self.frames_output, self.checkpoint_output, self.log_output]
        for dir_tmp in make_dirs:
            os.makedirs(dir_tmp, exist_ok=True)
        
        log_file = os.path.join(self.log_output, f"ensemble_{datetime.now().strftime('%m%d%H%M')}.log")

        self.logger = create_logger(name="ENSEMBLE", log_file=log_file)
        self.logger.info(f"Initializing.")

        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, list):
                val = '\n\t\t\t' + '\n\t\t\t'.join(map(str, val))
                self.logger.info(f"{arg} = {val}")
            else:
                self.logger.info(f"{arg} = {val}")

        # Init device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda")
            self.logger.info(f'device = cuda')
        else:
            raise RuntimeError("Cannot find CUDA device.")

        # Training and validation data paths
        self.training_data_path = os.path.join(DATA_PATH, args.training_data_path)
        self.validation_data_path = os.path.join(DATA_PATH, args.validation_data_path)

        # Placeholder for cluster data
        self._train_cluster_data = None
        self._val_cluster_data = None
        
        self.loss_ratios = {
            'energy': args.energy_ratio,
            'forces': args.force_ratio,
            'virial': args.virial_ratio,
        }

        self.bulk_energy_ratio = args.bulk_energy_ratio
        self.bulk_force_ratio = args.bulk_force_ratio
        self.bulk_virial_ratio = args.bulk_virial_ratio

        self.batch_size = args.batch_size  # Share batch size for both val and train.
        
        self.lr = args.lr
        self.scheduler_gamma = args.scheduler_gamma
        
        self.validation_split_ratio = args.validation_split_ratio
        self.validation_interval = args.validation_interval

        self.epochs = args.epochs
        self.max_frames_per_mixture = args.max_frames_per_mixture
        self.frame_validation_interval = args.frame_validation_interval

        # Assign the list of frame directories from command line arguments
        self.frame_directories = args.frame_directories

        # Verify each specified frame directory exists
        for frame_dir in self.frame_directories:
            if not os.path.isdir(frame_dir):
                # Raise an error if a specified directory does not exist
                raise NotADirectoryError(f"Frame directory {frame_dir} not found.")

        
        self.logger.info("Initiating the loading of all models.")
        
        self.script_models = {}
        
        # Load all models into memory for efficiency, assuming a manageable total number.
        for model in self.args.models:
            if model not in self.script_models:    
                self.script_models[model] = torch.jit.load(model, map_location=self.device)
        
        if self.args.ensemble_model is None:
            self.ensemble_model = self.args.models[0]
            self.logger.info(f"Ensemble model not specified, using the first model: {self.ensemble_model}")
        else:
            self.ensemble_model = self.args.ensemble_model
            if self.ensemble_model not in self.script_models:
                raise ValueError(f"Ensemble model {self.ensemble_model} not found in models.")

        self.logger.info(f"Number of models: {len(self.script_models)}")
        
        self._cached_frames = {}
        self.uncertainty_train = []
        self.uncertainty_val = []
        self.split_data_flag = False

    @property
    def train_cluster_data(self) -> Dict[str, torch.Tensor]:
        if self._train_cluster_data is None:
            self._train_cluster_data = torch.load(self.training_data_path, map_location="cpu")
        return self._train_cluster_data
    
    @property
    def val_cluster_data(self) -> Dict[str, torch.Tensor]:
        if self._val_cluster_data is None:
            self._val_cluster_data = torch.load(self.validation_data_path, map_location="cpu")
        return self._val_cluster_data

    def uncertainty(self):
        self.logger.info("Start uncertainty quantification")

        configs = {}

        data_paths = []

        uncertainty_count = {}

        def add_frame_data(mixture_name: str, output_file: str):
            if mixture_name not in uncertainty_count:
                uncertainty_count[mixture_name] = 0
            uncertainty_count[mixture_name] += 1
            
            if uncertainty_count[mixture_name] % self.frame_validation_interval:
                self.uncertainty_train.append(output_file)
            else:
                self.uncertainty_val.append(output_file)

        config_file = os.path.join(self.frames_output, "config.json")
        if os.path.isfile(config_file):
            self.logger.info("Skipping uncertainty quantification, config file exists.")

            with open(config_file, "r") as f:
                configs = json.load(f)

            for data_path, config in configs.items():
                mixture_name = config["mixture_name"]
                output_file = config["data"]
                add_frame_data(mixture_name, output_file)            

            return
        
        # Recursively scan the frames folder to locate all data files.
        for frame_directory in self.frame_directories:
            if not os.path.isdir(frame_directory):
                raise NotADirectoryError(f"Frame directory {frame_directory} not found.")
            for root, _, files in os.walk(frame_directory):
                for file in files:
                    if file.endswith(".pt"):
                        data_paths.append(os.path.join(root, file))
        random.shuffle(data_paths)

        self.logger.info(f"Number of data: {len(data_paths)}")

        self.logger.info("Starting inference for uncertainty quantification.")

        for data_path in data_paths:  # Assuming data_paths is defined and passed correctly.
            # Load data directly onto the specified device.
            single_data: Dict[str, torch.Tensor] = torch.load(data_path, map_location=self.device)
            mixture_name = single_data["mixture_name"]

            # Skip processing if max frames per mixture limit is reached.
            if uncertainty_count.get(mixture_name, 0) > self.max_frames_per_mixture:
                self.logger.info(f"Skipping {mixture_name}: max frames per mixture reached.")
                continue

            ensemble_pred = {
                'energy': [],
                'forces': [],
                'virial': [],
            }

            # Collect predictions from all modules.
            for model in self.script_models.values():
                pred = model.forward(single_data["inputs"])
                
                ensemble_pred['energy'].append(torch.flatten(pred['pred_energy'].detach()))
                ensemble_pred['forces'].append(torch.flatten(pred['pred_forces'].detach()))
                ensemble_pred['virial'].append(torch.flatten(pred['pred_virial'].detach()))
                del pred
            ensemble_results = {}

            # Compute ensemble statistics for each property (energy, forces, virial).
            for property_name, predictions in ensemble_pred.items():
                # Stack all predictions for the current property along a new dimension.
                predictions_tensor = torch.stack(predictions, dim=0)
                
                # Calculate the mean and standard deviation along the stacked dimension.
                mean_list = torch.mean(predictions_tensor, dim=0).detach().cpu().tolist()
                std_list = torch.std(predictions_tensor, dim=0).detach().cpu().tolist()
                
                # Special handling for force and virial to group results in triplets.
                if property_name in ["force", "virial"]:
                    mean_list = [mean_list[i:i+3] for i in range(0, len(mean_list), 3)]
                    std_list = [std_list[i:i+3] for i in range(0, len(std_list), 3)]
                
                # Store the computed mean and standard deviation.
                ensemble_results[property_name] = {
                    "mean": mean_list,
                    "std": std_list
                }

            ensemble_results.update(single_data)

            # Save file {frames}/a/b/c.pt to {output}/c_{index}.pt
            file_base_name = os.path.splitext(os.path.basename(data_path))[0]
            index = 0

            def get_output_file(index: int) -> str:
                return os.path.join(self.frames_output, f"{file_base_name}_{index}.pt")

            output_file = get_output_file(index)
            while os.path.isfile(output_file):
                index += 1
                output_file = get_output_file(index)

            configs[data_path] = {"mixture_name": mixture_name, "data": output_file}
            self.logger.info(f"Source: {data_path}, output: {output_file}")
            torch.save(ensemble_results, output_file)
            add_frame_data(mixture_name, output_file)

            del single_data
        # Save configs
        self.logger.info(f"Save config to {config_file}.")
        with open(config_file, "w") as config_fp:
            json.dump(configs, config_fp, indent=4)
        self.logger.info("Uncertainty quantification finished")

    def cluster_validation(self, model: BambooGET, cluster: Dict[str, torch.Tensor]) -> Dict[str, float]:
        keys = ['energy', 'forces', 'virial', 'dipole']
        val_rmse = {k: [] for k in keys}
        val_data_size = len(cluster['total_charge'])
        total_step = val_data_size // self.batch_size
        for step in range(total_step):
            batch_data = batchify(cluster, step*self.batch_size, (step+1)*self.batch_size, device=self.device)
            mse, _, _ = model.get_loss(batch_data)
            for k in keys:
                val_rmse[k].append(mse[k].item() * self.batch_size)
        total_val_rmse = {}
        total_val_rmse["cluster_energy_rmse"] = np.sqrt(sum(val_rmse["energy"]) / self.batch_size / total_step)
        total_val_rmse["cluster_force_rmse"] = np.sqrt(sum(val_rmse["forces"]) / self.batch_size / total_step)
        total_val_rmse["cluster_virial_rmse"] = np.sqrt(sum(val_rmse["virial"]) / self.batch_size / total_step)
        total_val_rmse["cluster_dipole_rmse"] = np.sqrt(sum(val_rmse["dipole"]) / self.batch_size / total_step)
        return total_val_rmse

    def bulk_validation(self, model: BambooGET, files: List[str]) -> Dict[str, float]:
        val_forces_rmse = []
        val_energy_rmse = []
        val_virial_rmse = []

        for file in files:
            single_data_pt = self.load_frame(file)
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

    def should_evaluate_model(self, epoch: int) -> bool:
        """Determine if the model should be evaluated based on the epoch."""
        is_validation_epoch = (epoch % self.validation_interval == 0)
        is_last_epoch = (epoch == self.epochs - 1)
        return is_validation_epoch or is_last_epoch
    
    def run(self):
        if not self.uncertainty_train:
            raise ValueError("No uncertainty_train frames available.")
        
        # Save the ensembled model to {save_dir}/ensembled.pt
        self.logger.info(f"Start finetuning for model: {self.ensemble_model}")
        checkpoint_path = os.path.join(self.checkpoint_output, f"ensemble.pt")
        if os.path.isfile(checkpoint_path):
            self.logger.info(f"checkpoint already exists: {checkpoint_path}")
            return

        script_model = self.script_models[self.ensemble_model]
        model = convert(script_model, device=self.device)
        model.train()

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
        bulk_val_rmse = self.bulk_validation(model, self.uncertainty_val)
        cluster_val_rmse = self.cluster_validation(model, self.val_cluster_data)

        add_train_curve(-1, bulk_val_rmse, cluster_val_rmse)
        log_train_curve()

        # Start ensemble training
        total_cluster_data = len(self.train_cluster_data['total_charge'])
        cluster_batch_num = total_cluster_data // self.batch_size
        cluster_random_index = list(range(cluster_batch_num))
        random.shuffle(cluster_random_index)
        n_cluster_index = 0
        
        for epoch in range(self.epochs):
            train_order = list(range(len(self.uncertainty_train)))
            random.shuffle(train_order)
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

                # Training on cluster data
                cluster_loss = torch.tensor(0.0, device=self.device)
                n_cluster_index = (n_cluster_index + 1) % cluster_batch_num
                start = cluster_random_index[n_cluster_index] * self.batch_size
                end = start + self.batch_size
                batch_data = batchify(self.train_cluster_data, start, end, device=self.device)
                mse, _, _ = model.get_loss(batch_data)
                for k in self.loss_ratios.keys():
                    cluster_loss += self.loss_ratios[k] * mse[k]
                del batch_data
                cluster_loss.backward()
                optimizer.step()
            
            scheduler.step()
            if self.should_evaluate_model(epoch):
                bulk_val_rmse = self.bulk_validation(model, self.uncertainty_val)
                cluster_val_rmse = self.cluster_validation(model, self.val_cluster_data)

                add_train_curve(epoch, bulk_val_rmse, cluster_val_rmse)
                log_train_curve()

        # Save ensembled model.
        script_model = torch.jit.script(model)
        torch.jit.save(script_model, checkpoint_path)
        self.logger.info(f"Ensembled model saved to {checkpoint_path}")

        # Save training curve
        curve_path = os.path.join(self.log_output, "training_curve.csv")
        df = pd.DataFrame(training_curve)
        df.to_csv(curve_path, index=False)

    def load_frame(self, file: str) -> Dict[str, torch.Tensor]:
        # Load a frame from the specified file. If the frame is not already cached,
        # it loads it into the cache.
        if file not in self._cached_frames:
            self._cached_frames[file] = torch.load(file, map_location='cpu')
        
        return self._cached_frames[file]


def main():
    args = get_parser()
    distiller = DistillationEnsemble(args)
    distiller.uncertainty()
    distiller.run()


if __name__ == "__main__":
    main()
