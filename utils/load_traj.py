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
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import KDTree

from utils.constant import atom_mapper


class Domain:
    def __init__(self) -> None:
        self.xlo = None
        self.xhi = None
        self.ylo = None
        self.yhi = None
        self.zlo = None
        self.zhi = None

    @property
    def box(self) -> List[List[float]]:
        if None in [self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi]:
            print("Domain is not initialized")
        return [[self.xlo, self.xhi], [self.ylo, self.yhi], [self.zlo, self.zhi]]  # type: ignore


class Frame:
    def __init__(self, filter_config: Optional[dict] = None) -> None:
        """
        Initialize a Frame object with optional filter configuration.
        
        :param filter_config: A dictionary containing filter configuration options.
        """
        # Initialize instance attributes with default values or provided arguments
        self.step: Optional[int] = None
        self.header: Optional[list] = None
        self.filter_config: dict = filter_config if filter_config is not None else {}
        self.domain: Domain = Domain()

        self.data = None 
    
    def _convert_dtype(self, x: str):
        # Define type conversion in a separate method for clarity
        int_types = ["id", "type", "ix", "iy", "iz"]
        float_types = ["xu", "yu", "zu", "xs", "ys", "zs", "x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "q", "charge"]

        if x in int_types:
            return (x, np.int32)
        elif x in float_types:
            return (x, np.float32)
        else:
            raise ValueError(f"Unknown type {x}")

    def _filter_data(self):
        # Implement data filtering in a separate method for clarity
        if "field" in self.filter_config:
            self.data = self.data[self.filter_config["field"]]
        if "type" in self.filter_config:
            self.data = self.data[np.isin(self.data["type"], self.filter_config["type"])]

    def parse(self, data: list):
        self.step = int(data[1])
        self.natoms = int(data[3])

        self.domain.xlo, self.domain.xhi = map(float, data[5].split())
        self.domain.ylo, self.domain.yhi = map(float, data[6].split())
        self.domain.zlo, self.domain.zhi = map(float, data[7].split())

        # Filter by frame_interval if specified
        if self.filter_config.get("frame_interval", 0) and self.step % self.filter_config["frame_interval"] != 0:
            return
        
        # Process header and set dynamic properties
        self.header = data[8].strip("\n").split()[2:]
        for prop_name in self.header:
            def make_property(name):
                return property(lambda self: self.data[name])
            setattr(self.__class__, prop_name, make_property(prop_name))

        my_dtype = [self._convert_dtype(x) for x in self.header]
        self.data = np.genfromtxt(data, skip_header=9, dtype=my_dtype)
        self.data = np.sort(self.data, order="id")

        self._filter_data()

class LammpsDump:
    def __init__(self, file_path: str, filter_config: Optional[dict] = None):
        """
        Initialize a LammpsDump object to parse LAMMPS dump files.
        
        :param file_path: Path to the LAMMPS dump file.
        :param filter_config: Optional dictionary for filtering data during parsing.
        """
        self.file_path = file_path
        self.filter_config = filter_config if filter_config is not None else {}
        self.frames = []
        self.domain = Domain()
        
        with open(file_path, 'r') as file:
            self.data = file.readlines()

        self.length = len(self.data)
        if self.length < 10:
            raise ValueError("File too short to contain valid data.")
        
        self.natoms = int(self._line(3))
        self.nframes = self.length // (self.natoms + 9)
        self.domain.xlo, self.domain.xhi = map(float, self._line(5).split())
        self.domain.ylo, self.domain.yhi = map(float, self._line(6).split())
        self.domain.zlo, self.domain.zhi = map(float, self._line(7).split())

    def _line(self, n: int):
        return self.data[n].strip("\n")
    
    def series_parse(self):
        """
        Parse all frames in the series.
        """
        self.frames.clear()
        if not self.natoms or not self.nframes:
            return
        
        for i in range(self.nframes):
            frame_data = self.data[i * (self.natoms + 9): (i + 1) * (self.natoms + 9)]
            frame = Frame(self.filter_config)
            frame.parse(frame_data)
            if frame.data is not None:
                self.frames.append(frame)


def get_edge(pos: torch.Tensor, kd_tree: KDTree, cutoff: float, boxsize_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pairs = kd_tree.query_pairs(r=cutoff, output_type='ndarray')
    edge_index = torch.tensor(pairs.T, dtype=torch.long)
    # Concat inverse direction edge index.
    edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)
    row, col = edge_index[0], edge_index[1]
    edge_cell_shift = pos[row] - pos[col]
    edge_cell_shift = torch.remainder(edge_cell_shift + 0.5*boxsize_tensor, boxsize_tensor) - 0.5 * boxsize_tensor
    return edge_index, edge_cell_shift


def parse_in_data_file(in_data_file: str) -> torch.Tensor:
    """Parse molecular data from the input file."""
    molecular = []
    with open(in_data_file) as file:
        start_flag = False
        for line in file:
            if line.startswith('Atoms'):
                start_flag = True
                continue
            if not start_flag or not line.strip():
                continue
            molecular.append(int(line.split()[1]))
    return torch.tensor(molecular)


def parse_in_lammps_file(in_lammps_file: str) -> np.array:
    """Extract type list from the LAMMPS input file."""
    type_list = []
    with open(in_lammps_file) as file:
        for line in file:
            if line.startswith("pair_coeff"):
                strs = line.split()
                type_list = [atom_mapper[symbol] for symbol in strs[2:]]
                break
    if not type_list:
        logging.warning("No type list found.")
    return np.array(type_list)


def fetch_pressure_volume(log_lammps: str):
    log_info = {}
    with open(log_lammps, 'r') as fin:
        lines = fin.readlines()
        start_flag = False
        for line in lines:
            if line.startswith('  G vector'):
                log_info['g_ewald'] = torch.tensor(float(line.split()[-1]))
            if line.startswith('Step Temp Press'):
                start_flag = True
                continue
            if not start_flag:
                continue
            if line.startswith("Loop time of"):
                start_flag = False
                continue
            
            strs = line.split()
            log_info[int(strs[0])] = {
                "pressure": torch.tensor(float(strs[2])),
                "volume": torch.tensor(float(strs[3]))
            }

    return log_info

def prepare_frame_data(frame: Frame, type_array: np.array, molecular: torch.Tensor,
                       nn_cutoff: float, coul_cutoff: float, disp_cutoff: float):
    data = {}
    domain_box = np.array(frame.domain.box)
    pos = np.stack([frame.data['xu'], frame.data['yu'], frame.data['zu']], axis=1)
    pos = pos - domain_box[:, 0]
    atom_type = type_array[frame.data['type'] - 1]
    box_size = domain_box[:, 1] - domain_box[:, 0]

    # # Convert to tensors
    boxsize_tensor = torch.tensor(box_size)
    pos = np.remainder(pos, box_size)
    kd_tree = KDTree(pos, boxsize=boxsize_tensor)
    pos = torch.tensor(pos)
    data['pos'] = pos
    data['atom_types'] = torch.tensor(atom_type, dtype=torch.long)

    edge_index, edge_cell_shift = get_edge(pos, kd_tree, nn_cutoff, boxsize_tensor=boxsize_tensor)
    data['edge_index'], data['edge_cell_shift'] = edge_index, edge_cell_shift
    outer = molecular[data["edge_index"][0]] != molecular[data["edge_index"][1]]
    
    data['molecular'] = torch.LongTensor(molecular)

    data['edge_outer_mask'] = outer.to(torch.float32)

    if nn_cutoff == coul_cutoff:
        data['coul_edge_index'], data['coul_edge_cell_shift'] = edge_index, edge_cell_shift
    else:
        coul_edge_index, coul_edge_cell_shift = get_edge(pos, kd_tree, coul_cutoff, boxsize_tensor=boxsize_tensor)
        data['coul_edge_index'], data['coul_edge_cell_shift'] = coul_edge_index, coul_edge_cell_shift
    disp_edge_index, disp_edge_cell_shift = get_edge(pos, kd_tree, disp_cutoff, boxsize_tensor=boxsize_tensor)
    data['disp_edge_index'], data['disp_edge_cell_shift'] = disp_edge_index, disp_edge_cell_shift
    data['cell'] = torch.tensor([[box_size[0], 0., 0.], [0., box_size[1], 0.], [0., 0., box_size[2]]], dtype=torch.float32)

    return data


def write_data(
        job_folder: str,
        output_folder: str,
        mixture_name: str,
        nn_cutoff: float = 5.0,
        coul_cutoff: float = 5.0,
        disp_cutoff: float = 10.0,
        interval: int = 10,
        log_file: str = "log.lammps"
    ) -> None:
    # interval: unit ps
    # cutoff: unit A

    in_data_file = os.path.join(job_folder, 'in.data')
    in_lammps_file = os.path.join(job_folder, 'in.lammps')
    out_lammps_log = os.path.join(job_folder, log_file)
    out_lammps_traj = os.path.join(job_folder, 'dump_nvt.lammpstrj')

    os.makedirs(output_folder, exist_ok=True)

    molecular = parse_in_data_file(in_data_file)

    type_array = parse_in_lammps_file(in_lammps_file)

    log_info = fetch_pressure_volume(out_lammps_log)

    filter_config = {"field": ["id", "type", "xu", "yu", "zu"], "frame_interval": int(1000 * interval)}
    lammps_dump = LammpsDump(file_path=out_lammps_traj, filter_config=filter_config)
    lammps_dump.series_parse()

    if not lammps_dump.frames:
        return

    for frame in lammps_dump.frames:
        data = prepare_frame_data(frame, type_array, molecular, nn_cutoff, coul_cutoff, disp_cutoff)
        data["g_ewald"] = log_info["g_ewald"]
        if frame.step not in log_info:
            continue
        data.update(log_info[frame.step])

        output_data = {"inputs": data, "mixture_name": mixture_name}
        torch.save(output_data, os.path.join(output_folder, f'frame_{frame.step}.pt'))


if __name__ == "__main__":
    # args for work folder and output folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--mixture_name', type=str, required=True)
    parser.add_argument('--nn_cutoff', type=float, default=5.0)
    parser.add_argument('--coul_cutoff', type=float, default=5.0)
    parser.add_argument('--disp_cutoff', type=float, default=10.0)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--log_file', type=str, default="log.lammps")
    args = parser.parse_args()

    write_data(
        job_folder=args.job_folder,
        output_folder=args.output_folder,
        mixture_name=args.mixture_name,
        nn_cutoff=args.nn_cutoff,
        coul_cutoff=args.coul_cutoff,
        disp_cutoff=args.disp_cutoff,
        interval=args.interval,
        log_file=args.log_file
    )