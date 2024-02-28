import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from models.bamboo_et import BambooET
from utils.get_batch import get_batch, split_data
from utils.log_helper import create_logger
from utils.rejit import convert


def get_parser(config_path: Optional[str] = None):
    # Create the parser
    parser = argparse.ArgumentParser(description="Parameters for the finetune script")

    # Required arguments
    parser.add_argument("--work_dir", type=str, default="default", help="Working directory")

    parser.add_argument("--train_cluster", type=str, default="train_pyscf_svpd_b3lyp_more_data_clean_loose_double_10192023.pt", help="Training cluster")
    parser.add_argument("--val_cluster", type=str, default="val_pyscf_svpd_b3lyp_more_data_clean_loose_double_10192023.pt", help="Validation cluster")
    parser.add_argument("--data_path", type=str, default="/mnt/bn/ai4s-hl/bamboo/pyscf_data/data", help="Path to the data")

    parser.add_argument("--model", type=str, default=None, help="Path to the model")
    parser.add_argument("--mixture_names", nargs="*", type=str, default=[], help="List of mixtures")
    parser.add_argument("--frame_folders", nargs="*", type=str, default=[], help="List of frame folders")
    parser.add_argument("--delta_pressure", nargs="*", type=float, default=[], help="List of delta pressure")
    
    parser.add_argument("--energy_ratio", type=float, default=0.3, help="Energy ratio")
    parser.add_argument("--force_ratio", type=float, default=1.0, help="Force ratio")
    parser.add_argument("--virial_ratio", type=float, default=0.1, help="Virial ratio")
    parser.add_argument("--dipole_ratio", type=float, default=3.0, help="Dipole ratio")
    
    parser.add_argument("--bulk_energy_ratio", type=float, default=1e2, help="Bulk energy ratio")
    parser.add_argument("--bulk_force_ratio", type=float, default=1e6, help="Bulk force ratio")
    parser.add_argument("--bulk_virial_ratio", type=float, default=3e3, help="Bulk virial ratio")

    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--frame_val_interval", type=int, default=3, help="Validation ratio")
    parser.add_argument("--max_frame_per_system", type=int, default=30, help="Max frames per system")

    parser.add_argument("--lr", type=float, default=1e-12, help="Learning rate")
    parser.add_argument("--scheduler_gamma", type=float, default=0.99, help="Scheduler gamma")

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


class DensityFinetune:
    def __init__(self, args) -> None:

        # Constant
        self.nktv2p = 68568.4
        
        self.args = args
        self.work_dir: str = args.work_dir
        self.checkpoint_output = os.path.join(self.work_dir, "checkpoints")
        self.log_output = os.path.join(self.work_dir, "logs")
        os.makedirs(self.checkpoint_output, exist_ok=True)
        os.makedirs(self.log_output, exist_ok=True)

        self.log_file = os.path.join(self.log_output, "finetune.log")
        self.logger = create_logger("Finetune", self.log_file)

        self.train_cluster_file = os.path.join(args.data_path, args.train_cluster)
        self.val_cluster_file = os.path.join(args.data_path, args.val_cluster)
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

        self.delta_pressure = {k: v for k, v in zip(args.mixture_names, args.delta_pressure)}
        self.logger.info(f"{self.delta_pressure}")

        # If all delta pressure is nearly zero, skip finetune.
        if np.allclose(list(self.delta_pressure.values()), 0, rtol=1e-1):
            self.logger.info("All delta pressure is zero, skip finetune.")
            self.skip_finetune = True
        else:
            self.skip_finetune = False

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

        self.frame_folders = args.frame_folders

        self.lr: float = args.lr
        self.scheduler_gamma: float = args.scheduler_gamma
        self.epochs: int = args.epochs
        self.batch_size: int = args.batch_size
        self.frame_val_interval: int = args.frame_val_interval
        self.max_frame_per_system: int = args.max_frame_per_system

        self.device = torch.device('cuda')
        self.split_data_flag = False
        self.result = {}

    @property
    def train_cluster_data(self) -> Dict[str, torch.Tensor]:
        if self._train_cluster_data is None:
            self._train_cluster_data = torch.load(self.train_cluster_file, map_location="cpu")
        return self._train_cluster_data
    
    @property
    def val_cluster_data(self) -> Dict[str, torch.Tensor]:
        if self._val_cluster_data is None:
            self._val_cluster_data = torch.load(self.val_cluster_file, map_location=self.device)
        return self._val_cluster_data

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # *.pts -> key: val
        # name: str -> mixture_name: str
        frames = []
        train_frames = []
        val_frames = []

        mixture_name_counter = {k: 0 for k in self.delta_pressure}

        # Check if all frames_dir exist
        for dir_tmp in self.frame_folders:
            if not os.path.isdir(dir_tmp):
                raise NotADirectoryError(f"frames_dir {dir_tmp} not found")
            list_frame_folders = [f for f in os.listdir(dir_tmp) if f.endswith(".pt")]
            list_frame_folders = [os.path.join(dir_tmp, f) for f in list_frame_folders]
            frames.extend(list_frame_folders)

        for frame_path in frames:
            frame_data = torch.load(frame_path, map_location=self.device)
            frame_info = frame_data["config"]
            mixture_name = frame_info["mixture_name"]
            
            if mixture_name not in self.delta_pressure:
                self.logger.warning(f"Skipping {frame_path}: {mixture_name} because it is not in {self.delta_pressure}.")
                continue

            if mixture_name_counter[mixture_name] >= self.max_frame_per_system:
                self.logger.warning(f"Skipping {frame_path}: {mixture_name} because it exceeds exceed max_frame_per_system.")
                continue

            pred: Dict[str, torch.Tensor] = self.train_model.forward(frame_data["inputs"])
            # detach the tensors to save memory.
            for k, v in pred.items():
                pred[k] = v.detach()
                pred[k].requires_grad = False
            
            result_tmp = {}

            result_tmp["delta_pressure"] = self.delta_pressure[mixture_name]
            result_tmp["frame_path"] = frame_path
            result_tmp["nn_virial_outer"] = pred["nn_virial_outer"]
            result_tmp["mixture_name"] = mixture_name

            if mixture_name_counter[mixture_name] % self.frame_val_interval == 0:
                val_frames.append(result_tmp)
            else:
                for k in self.bulk_loss_ratios:
                    result_tmp[k] = pred[k]
                train_frames.append(result_tmp)
            mixture_name_counter[mixture_name] += 1
            # In case of OOM, we need to delete the tensors.
            del frame_data, pred

        self.logger.info(f"Loaded {len(train_frames)} train frames and {len(val_frames)} val frames.")
        for k, v in mixture_name_counter.items():
            self.logger.info(f"mixture name: {k}, count: {v}")

        return train_frames, val_frames

    def bulk_validation(self, model: BambooET, val_frames: List[Dict[str, Any]]) -> Dict[str, float]:
        val_dp_outer = {k: [] for k in self.delta_pressure}
        for curr_frame in val_frames:
            frame_data = torch.load(curr_frame["frame_path"], map_location=self.device)
            inputs = frame_data["inputs"]
            pred = model.forward(inputs)
            nn_virial_outer_diff = pred['nn_virial_outer'] - curr_frame["nn_virial_outer"]
            volume = inputs['cell'][0][0] * inputs['cell'][1][1] * inputs['cell'][2][2]
            pred_outer_press = self.nktv2p * nn_virial_outer_diff / (3 * volume) - curr_frame["delta_pressure"]
            val_dp_outer[curr_frame["mixture_name"]].append(pred_outer_press.item())
            del frame_data, pred, inputs
        val_dp_outer_mean: Dict[str, float] = {k: np.mean(v) for k, v in val_dp_outer.items()}
        dp_avg = np.mean([v for k, v in val_dp_outer_mean.items()])
        dp_std = np.std([v for k, v in val_dp_outer_mean.items()])
        val_dp_outer_mean.update({"AVG": dp_avg, "STD": dp_std})
        return val_dp_outer_mean

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
        return total_val_rmse

    def construct_log(self, info: dict, name: Optional[str] = None, baseline: Optional[dict] = None):
        log = ""
        for k, v in info.items():
            if log:
                log += ", "
            if baseline is not None:
                diff = v - baseline[k]
                diff_sign = "+ " if diff >= 0 else "- "
                log += f"{k}: {v:.2f} ({diff_sign}{abs(diff):.2f})"
            else:
                log += f"{k}: {v:.2f}"
        if isinstance(name, str):
            log = f"{name}: " + log
        self.logger.info(log)

    def finetune(self):
        if self.skip_finetune:
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

        previous_cluster_count = len(self.train_cluster_data["cell"])
        target_data_size = min(len(train_frames) * self.batch_size, 50000)
        if target_data_size < previous_cluster_count:
            split_data(data=self.train_cluster_data, start_index=0, end_index=target_data_size, device=self.device)
        after_cluster_count = len(self.train_cluster_data["cell"])

        self.logger.info(f"Previous cluster: {previous_cluster_count}, Current: {after_cluster_count}")

        total_cluster_data = len(self.train_cluster_data['total_charge'])
        cluster_batch_num = total_cluster_data // self.batch_size
        cluster_random_index = list(range(cluster_batch_num))
        random.shuffle(cluster_random_index)
        n_cluster_index = 0

        result: Dict[str, Any] = {}
        base_val_cluster_rmse = self.cluster_validation(model=self.train_model, cluster=self.val_cluster_data)
        val_dp_outer = self.bulk_validation(model=self.train_model, val_frames=val_frames)
        
        self.logger.info("Before finetune:")
        self.construct_log(base_val_cluster_rmse, name="[CLUSTER]")
        self.construct_log(val_dp_outer, name="[BULK]")
        self.logger.info(f"Finetune starts. frames: {len(train_frames)}")

        for epoch in range(self.epochs):
            epoch_start_timer = time.time()
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
                loss: torch.Tensor = (self.nktv2p * nn_virial_outer_diff / (3*volume) - curr_frame["delta_pressure"])**2
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
                start = cluster_random_index[n_cluster_index]
                end = start + self.batch_size
                
                batch_data = get_batch(self.train_cluster_data, start, end, device=self.device)
                mse = self.train_model.get_mse_loss(batch_data)
                for k, v in self.cluster_loss_ratio.items():
                    cluster_loss += v * mse[k]
                del batch_data
                
                cluster_loss.backward()
                optimizer.step()

            epoch_complete_timer = time.time()

            val_cluster_rmse = self.cluster_validation(model=self.train_model, cluster=self.val_cluster_data)
            val_dp_outer = self.bulk_validation(model=self.train_model, val_frames=val_frames)
            epoch_val_timer = time.time()
            
            # self.logger.info(f"Epoch {epoch} done. Time: {epoch_complete_timer - epoch_start_timer:.2f}, Validation Time: {epoch_val_timer - epoch_complete_timer:.2f}")
            self.construct_log(val_cluster_rmse, name="[CLUSTER]", baseline=base_val_cluster_rmse)
            self.construct_log(val_dp_outer, name=f"[EPOCH: {epoch}]")

            scheduler.step()
            self.result = val_dp_outer
        
    def conclude(self):
        # Save the model.
        module = torch.jit.script(self.train_model)
        module_file = os.path.join(self.checkpoint_output, "finetune.pt")
        module.save(module_file) # type: ignore
        self.result["model"] = module_file

        # save result info.
        result_file = os.path.join(self.work_dir, "result.json")
        with open(result_file, "w") as f:
            json.dump(self.result, f, indent=4)


def finetune_run(config_path: Optional[str] = None):
    args = get_parser(config_path)

    density_finetune = DensityFinetune(args)
    density_finetune.finetune()
    density_finetune.conclude()


if __name__ == "__main__":
    # For local test.
    finetune_run()
