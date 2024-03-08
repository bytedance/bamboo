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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from models.bamboo_get import BambooGET
from utils.batchify import batchify
from utils.log_helper import create_logger
from utils.path import ALIGNMENT_PATH, DATA_PATH
from utils.rejit import convert
from utils.constant import nktv2p


def get_parser():
    # Create the parser
    parser = argparse.ArgumentParser(description="Arguments for bamboo model alignment.")

    # Required arguments
    parser.add_argument('--config', default='', type=str, help="Path to a configuration file in JSON format.")
    parser.add_argument('--job_name', default='default', type=str)

    # Training and validation data paths
    parser.add_argument("--training_data_path", type=str, default="train_data.pt", help="Path to the training data file.")
    parser.add_argument("--validation_data_path", type=str, default="val_data.pt", help="Path to the validation data file.")
    
    # Data sources and model configuration
    parser.add_argument("--model", type=str, default=None, help="Specify the model's path for use in alignment.")
    parser.add_argument("--frame_directories", nargs="*", type=str, default=[], help="List of directories that contain frame data for processing.")
    parser.add_argument("--mixture_names", nargs="*", type=str, default=[], help="Names of mixtures to be considered during alignment.")
    parser.add_argument("--delta_pressure", nargs="*", type=float, default=[], help="Delta pressures for the respective mixtures, listed in the same order as mixture names.")
    
    # Data property weighting
    parser.add_argument("--energy_ratio", type=float, default=0.3, help="Weight of energy predictions in the loss function.")
    parser.add_argument("--force_ratio", type=float, default=1.0, help="Weight of force predictions in the loss function.")
    parser.add_argument("--virial_ratio", type=float, default=0.1, help="Weight of virial predictions in the loss function.")
    parser.add_argument("--dipole_ratio", type=float, default=3.0, help="Weight of dipole predictions in the loss function.")
    parser.add_argument("--bulk_energy_ratio", type=float, default=1e2, help="Weight of bulk energy predictions in the loss function.")
    parser.add_argument("--bulk_force_ratio", type=float, default=1e6, help="Weight of bulk force predictions in the loss function.")
    parser.add_argument("--bulk_virial_ratio", type=float, default=3e3, help="Weight of bulk virial predictions in the loss function.")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=512, help="Number of samples processed together in one pass.")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of training cycles through the entire dataset.")
    parser.add_argument("--frame_val_interval", type=int, default=3, help="Interval for validating the model with the validation dataset.")
    parser.add_argument("--max_frame_per_mixture", type=int, default=30, help="Maximum number of frames allowed for each mixture.")
    parser.add_argument("--lr", type=float, default=1e-12, help="Initial learning rate for the optimization algorithm.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.99, help="Decay rate for adjusting the learning rate across epochs.")

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


class DensityAlignment:
    def __init__(self, args) -> None:

        self.args = args
        self.work_dir: str = os.path.join(ALIGNMENT_PATH, self.args.job_name)
        self.checkpoint_output = os.path.join(self.work_dir, "checkpoints")
        self.log_output = os.path.join(self.work_dir, "logs")
        os.makedirs(self.checkpoint_output, exist_ok=True)
        os.makedirs(self.log_output, exist_ok=True)

        self.log_file = os.path.join(self.log_output, "alignment.log")
        self.logger = create_logger("ALIGNMENT", self.log_file)

        # Training and validation data paths
        self.training_data_path = os.path.join(DATA_PATH, args.training_data_path)
        self.validation_data_path = os.path.join(DATA_PATH, args.validation_data_path)
        
        # Placeholder for cluster data
        self._train_cluster_data = None
        self._val_cluster_data = None
        
        self.cluster_loss_ratio: Dict[str, float] = {
            "energy": args.energy_ratio,
            "forces": args.force_ratio,
            "virial": args.virial_ratio,
            "dipole": args.dipole_ratio,
        }

        self.bulk_loss_ratios: Dict[str, float] = {
            "pred_energy": args.bulk_energy_ratio,
            "pred_forces": args.bulk_force_ratio,
            "pred_virial": args.bulk_virial_ratio,
        }

        # Initialize delta_pressure dictionary directly from zipped mixture names and delta pressures
        self.delta_pressure = dict(zip(args.mixture_names, args.delta_pressure))

        # Log the current state of delta pressure
        self.logger.info(f"Delta Pressure: {self.delta_pressure}")

        # Determine if we should skip alignment based on the condition that all delta pressures are close to zero
        self.skip_alignment = np.allclose(list(self.delta_pressure.values()), 0, rtol=1e-1)

        # Log the decision on whether to skip finetuning
        if self.skip_alignment:
            self.logger.info("All delta pressures are nearly zero, skipping alignment.")

        self.model = args.model
        if not os.path.isfile(self.model):
            raise FileNotFoundError(f"Model file {self.model} not found.")
        self.train_model = convert(self.model)
        
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, list):
                val = '\n\t\t\t' + '\n\t\t\t'.join(map(str, val))
                self.logger.info(f"{arg} = {val}")
            else:
                self.logger.info(f"{arg} = {val}")

        self.frame_directories = args.frame_directories

        self.lr: float = args.lr
        self.scheduler_gamma: float = args.scheduler_gamma
        self.epochs: int = args.epochs
        self.batch_size: int = args.batch_size
        self.frame_val_interval: int = args.frame_val_interval
        self.max_frame_per_mixture: int = args.max_frame_per_mixture

        self.device = torch.device('cuda')
        self.result = {}

    @property
    def train_cluster_data(self) -> Dict[str, torch.Tensor]:
        if self._train_cluster_data is None:
            self._train_cluster_data = torch.load(self.training_data_path, map_location="cpu")
        return self._train_cluster_data
    
    @property
    def val_cluster_data(self) -> Dict[str, torch.Tensor]:
        if self._val_cluster_data is None:
            self._val_cluster_data = torch.load(self.validation_data_path, map_location=self.device)
        return self._val_cluster_data

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # *.pts -> key: val
        # name: str -> mixture_name: str
        frames = []
        train_frames = []
        val_frames = []

        mixture_name_counter = {k: 0 for k in self.delta_pressure}

        # Recursively scan the frames folder to locate all data files.
        for frame_directory in self.frame_directories:
            if not os.path.isdir(frame_directory):
                raise NotADirectoryError(f"Frame directory {frame_directory} not found.")
            for root, _, files in os.walk(frame_directory):
                for file in files:
                    if file.endswith(".pt"):
                        frames.append(os.path.join(root, file))

        for frame_path in frames:
            frame_data = torch.load(frame_path, map_location=self.device)
            mixture_name = frame_data["mixture_name"]
            
            if mixture_name not in self.delta_pressure:
                self.logger.warning(f"Skipping {frame_path}: {mixture_name} because it is not in delta_pressure.")
                continue

            if mixture_name_counter[mixture_name] >= self.max_frame_per_mixture:
                self.logger.warning(f"Skipping {frame_path}: {mixture_name} because it exceeds exceed max_frame_per_mixture.")
                continue

            # Process frame_data and prepare result template
            pred: Dict[str, torch.Tensor] = self.train_model.forward(frame_data["inputs"])
            # detach the tensors to save memory.
            for k, v in pred.items():
                pred[k] = v.detach()
                pred[k].requires_grad = False
            
            result_tmp = {
                "delta_pressure": self.delta_pressure[mixture_name],
                "frame_path": frame_path,
                "nn_virial_outer": pred["nn_virial_outer"],
                "mixture_name": mixture_name,
            }

            # Decide on whether to add to validation or training frames
            if mixture_name_counter[mixture_name] % self.frame_val_interval == 0:
                val_frames.append(result_tmp)
            else:
                for k in self.bulk_loss_ratios:
                    result_tmp[k] = pred[k]
                train_frames.append(result_tmp)
            mixture_name_counter[mixture_name] += 1
        
            # Explicit cleanup to assist garbage collection and reduce memory footprint
            del frame_data, pred

        # Log the count of train and validation frames
        num_train_frames = len(train_frames)
        num_val_frames = len(val_frames)
        self.logger.info(f"Loaded {num_train_frames} train frames and {num_val_frames} val frames.")

        # Log counts for each mixture name
        mixture_counts_log = "\n".join([f"mixture name: {name}, count: {count}" for name, count in mixture_name_counter.items()])
        self.logger.info(f"Mixture counts:\n{mixture_counts_log}")

        return train_frames, val_frames

    def bulk_validation(self, model: BambooGET, val_frames: List[Dict[str, Any]]) -> Dict[str, float]:
        val_dp_outer = {mixture_name: [] for mixture_name in self.delta_pressure.keys()}

        # Process each validation frame
        for curr_frame in val_frames:
            frame_data = torch.load(curr_frame["frame_path"], map_location=self.device)
            inputs = frame_data["inputs"]
            pred = model.forward(inputs)
            nn_virial_outer_diff = pred['nn_virial_outer'] - curr_frame["nn_virial_outer"]
            volume = inputs['cell'][0][0] * inputs['cell'][1][1] * inputs['cell'][2][2]
            pred_outer_press = nktv2p * nn_virial_outer_diff / (3 * volume) - curr_frame["delta_pressure"]
            val_dp_outer[curr_frame["mixture_name"]].append(pred_outer_press.item())
            del frame_data, pred, inputs
        
        # Compute means for each mixture and overall statistics
        val_dp_outer_mean: Dict[str, float] = {k: np.mean(v) for k, v in val_dp_outer.items()}
        all_means = list(val_dp_outer_mean.values())

        dp_avg = np.mean(all_means)
        dp_std = np.std(all_means)
        val_dp_outer_mean.update({"AVG": dp_avg, "STD": dp_std})
        return val_dp_outer_mean

    def cluster_validation(self, model: BambooGET, cluster: Dict[str, torch.Tensor]) -> Dict[str, float]:
        keys = ['energy', 'forces', 'virial', 'dipole']
        val_rmse = {k: [] for k in keys}
        val_data_size = len(cluster['total_charge'])
        total_step = val_data_size // self.batch_size

        for step in range(total_step):
            start_idx = step * self.batch_size
            end_idx = (step + 1) * self.batch_size
            batch_data = batchify(cluster, start_idx, end_idx, device=self.device)
            mse, _, _ = model.get_loss(batch_data)
            
            for k in keys:
                val_rmse[k].append(mse[k].item() * self.batch_size)
        
        total_val_rmse = {f"cluster_{k}_rmse": np.sqrt(sum(val_rmse[k]) / val_data_size) for k in keys}
        return total_val_rmse

    def construct_log(self, info, name=None, baseline=None):
        log_parts = []
        for k, v in info.items():
            entry = f"{k}: {v:.2f}"
            if baseline is not None:
                diff = v - baseline.get(k, 0)  # Safely get baseline value, defaulting to 0
                diff_sign = "+" if diff >= 0 else "-"
                entry += f" ({diff_sign} {abs(diff):.2f})"
            log_parts.append(entry)
        
        log = ", ".join(log_parts)
        if name:
            log = f"{name}: {log}"
        
        self.logger.info(log)

    def run(self):
        if self.skip_alignment:
            self.conclude()
            return

        params = list(self.train_model.energy_mlp.parameters())
        optimizer = torch.optim.SGD(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler_gamma)

        train_frames, val_frames = self.load_data()
        self.logger.info(f"Train frames: {len(train_frames)}")
        self.logger.info(f"Val frames: {len(val_frames)}")

        if not train_frames:
            raise ValueError("No train frames found.")
        if not val_frames:
            raise ValueError("No val frames found.")


        total_cluster_data = len(self.train_cluster_data['total_charge'])
        cluster_batch_num = total_cluster_data // self.batch_size
        cluster_random_index = list(range(cluster_batch_num))
        random.shuffle(cluster_random_index)
        n_cluster_index = 0

        base_val_cluster_rmse = self.cluster_validation(model=self.train_model, cluster=self.val_cluster_data)
        val_dp_outer = self.bulk_validation(model=self.train_model, val_frames=val_frames)
        
        self.logger.info("Before alignment:")
        self.construct_log(base_val_cluster_rmse, name="[CLUSTER]")
        self.construct_log(val_dp_outer, name="[BULK]")
        self.logger.info(f"Alignment starts. frames: {len(train_frames)}")

        for epoch in range(self.epochs):
            train_order = list(range(len(train_frames)))
            random.shuffle(train_order)
            for idx in train_order:
                optimizer.zero_grad()

                curr_frame = train_frames[idx]
                
                frame_data = torch.load(curr_frame["frame_path"], map_location=self.device)
                inputs = frame_data["inputs"]

                natoms = len(inputs["atom_types"])
                natoms_ratio = {"pred_energy": 1.0 / natoms, "pred_forces": 1.0, "pred_virial": 1.0 / natoms}

                pred = self.train_model.forward(frame_data["inputs"])
                nn_virial_outer_diff = pred['nn_virial_outer'] - curr_frame["nn_virial_outer"]
                volume = inputs['cell'][0][0] * inputs['cell'][1][1] * inputs['cell'][2][2]
                loss: torch.Tensor = (nktv2p * nn_virial_outer_diff / (3*volume) - curr_frame["delta_pressure"])**2
                for k, v in self.bulk_loss_ratios.items():
                    loss += self.bulk_loss_ratios[k] * torch.mean((pred[k] - curr_frame[k])**2) * natoms_ratio[k]
                loss.backward()
                optimizer.step()
                del frame_data, pred, inputs

                # Train on cluster data.
                cluster_loss = torch.tensor(0.0, device=self.device)
                #training on cluster data
                optimizer.zero_grad()
                
                n_cluster_index = (n_cluster_index + 1) % cluster_batch_num
                start = cluster_random_index[n_cluster_index] * self.batch_size
                end = start + self.batch_size
                
                batch_data = batchify(self.train_cluster_data, start, end, device=self.device)
                mse, _, _ = self.train_model.get_loss(batch_data)
                for k, v in self.cluster_loss_ratio.items():
                    cluster_loss += v * mse[k]
                del batch_data
                
                cluster_loss.backward()
                optimizer.step()

            val_cluster_rmse = self.cluster_validation(model=self.train_model, cluster=self.val_cluster_data)
            val_dp_outer = self.bulk_validation(model=self.train_model, val_frames=val_frames)
            
            self.construct_log(val_cluster_rmse, name="[CLUSTER]", baseline=base_val_cluster_rmse)
            self.construct_log(val_dp_outer, name=f"[EPOCH: {epoch}]")

            scheduler.step()
            self.result = val_dp_outer
    
        self.conclude()

    def conclude(self):
        # Save the model.
        module = torch.jit.script(self.train_model)
        module_file = os.path.join(self.checkpoint_output, "alignment.pt")
        module.save(module_file) # type: ignore
        self.result["model"] = module_file

        # save result info.
        result_file = os.path.join(self.work_dir, "result.json")
        with open(result_file, "w") as f:
            json.dump(self.result, f, indent=4)


def main():
    args = get_parser()

    density_alignment = DensityAlignment(args)
    density_alignment.run()


if __name__ == "__main__":
    # For local test.
    main()
