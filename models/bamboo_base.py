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

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_runstats.scatter import scatter
from utils.constant import element_c6, element_r0, nelems
from utils.funcs import CosineCutoff, ExpNormalSmearing

class BambooBase(torch.nn.Module):
    def __init__(self, device, dtype,
                nn_params = {
                    'dim': 64,
                    'num_rbf': 32,
                    'rcut': 5.0,
                    'act_fn': nn.SiLU(),
                    'energy_mlp_layers': 2
                },
                coul_disp_params = {
                    'disp_cutoff': 10.0,
                }):
        super(BambooBase, self).__init__()
        self.device = device
        self.dtype = dtype
        self.nelems = nelems
        self.coul_disp_params = coul_disp_params

        self.c6_emb = torch.nn.Embedding.from_pretrained(torch.tensor(element_c6, device=device, dtype=dtype).unsqueeze(1), freeze=True)
        self.r0_emb = torch.nn.Embedding.from_pretrained(torch.tensor(element_r0, device=device, dtype=dtype).unsqueeze(1), freeze=True)

        self.dim = nn_params['dim']
        self.num_rbf = nn_params['num_rbf']
        self.rcut = nn_params['rcut']
        self.atom_embtab = nn.Embedding(self.nelems, self.dim)
        self.dis_rbf = ExpNormalSmearing(0.0, self.rcut, self.num_rbf, device=self.device)
        self.dis_rbf.reset_parameters()
        self.cutoff = CosineCutoff(0.0, self.rcut)

        self.energy_mlp_layers = nn_params['energy_mlp_layers']

        def get_mlp_layers(layers: int, dim: int):
            mlp_layers = []
            for i in range(layers):
                if i == 0:
                    mlp_layers.append(nn.Linear(dim, dim))
                else:
                    mlp_layers.append(nn.Linear(dim, dim))
                mlp_layers.append(nn_params['act_fn'])
            mlp_layers.append(nn.Linear(dim, 1))
            return mlp_layers

        self.energy_mlp = nn.Sequential(*get_mlp_layers(self.energy_mlp_layers, self.dim))

        self.nmol = 1

        self.to(self.device)

    def get_dispersion(self,
                       row: torch.Tensor,
                       col: torch.Tensor,
                       dij: torch.Tensor,
                       c6: torch.Tensor,
                       r0: torch.Tensor,
        ) -> List[torch.Tensor]:
        '''
            Compute D3-CSO dispersion energy and pairwise dispersion forces from C6 and r0 parameters.
            Only used in inference. Not valid in training.
        '''
        rij = torch.sqrt(torch.sum(torch.square(dij), dim=-1))
        c6ij = torch.sqrt(c6[row] * c6[col])
        r0ij = 0.5*(r0[row] + r0[col])

        # D3-CSO dispersion correction
        edisp = - c6ij / (rij ** 6 + 4.5 ** 6) * (0.85 + 0.82 / (1. + torch.exp(rij - 2.5 * r0ij)))
        fdisp = - 6 * c6ij * rij ** 5 / ((rij ** 6 + (4.5) ** 6) ** 2) * (0.85 + 0.82 / (1. + torch.exp(rij - 2.5 * r0ij))) \
            - c6ij / (rij ** 6 + (4.5) ** 6) * (0.82 * torch.exp(rij - 2.5 * r0ij) / ((1. + torch.exp(rij - 2.5 * r0ij))**2))
        disp_fij = dij * (fdisp / rij).unsqueeze(-1)

        # cutoff correction to ensure smoothness at cutoff radius
        edisp += c6ij / self.coul_disp_params['disp_cutoff']**6
        return edisp, disp_fij

    def graph_nn(self,
                node_feat: torch.Tensor,
                edge_index: torch.Tensor,
                coord_diff: torch.Tensor,
                radial: torch.Tensor,
                weights_rbf: torch.Tensor) -> torch.Tensor:
        '''
            Graph neural network to update node features.
            Implemented in models/bamboo_get.py
        '''
        raise NotImplementedError('graph_nn is not implemented')

    def energy_nn(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_feat = self.atom_embtab(inputs['atom_types'])
        coord_diff = inputs['edge_cell_shift']
        radial = torch.sqrt(torch.sum(coord_diff**2, 1))
        coord_diff = coord_diff / radial.unsqueeze(-1)
        weights_rbf = self.dis_rbf(radial)
        radial = self.cutoff(radial)

        # GNN message passing
        node_feat = self.graph_nn(node_feat, inputs['edge_index'], coord_diff, radial, weights_rbf)

        # Energy prediction
        energy = self.energy_mlp(node_feat).squeeze(-1) # Na
        nn_energy = scatter(energy, inputs['mol_ids'], dim=0, dim_size=self.nmol) # Nm

        return nn_energy

    def get_loss(self, inputs: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        '''
        Get MSE and MAE in training and validation.
        '''
        pred = self.predict(inputs)
        mse = {}
        pred_energy_ave, label_energy_ave = torch.mean(pred['energy']), torch.mean(inputs['energy'])
        mse['energy'] = torch.mean(torch.square(pred['energy'] - inputs['energy'] - pred_energy_ave + label_energy_ave))
        mse['forces'] = torch.mean(torch.square(pred['forces'] - inputs['forces']))
        mse['virial'] = torch.mean(torch.square(pred['virial'] - inputs['virial']))

        mae = {}
        mae['energy'] = torch.mean(torch.abs(pred['energy'] - inputs['energy'] - pred_energy_ave + label_energy_ave))
        mae['forces'] = torch.mean(torch.abs(pred['forces'] - inputs['forces']))
        mae['virial'] = torch.mean(torch.abs(pred['virial'] - inputs['virial']))

        penalty = {}
        penalty['msnnfij'] = torch.mean(torch.square(pred['nn_fij']))

        h_mask = inputs['atom_types'] == 1
        sum_h_mask = torch.sum(h_mask.float())
        if sum_h_mask < 1e-5:
            penalty['mse_h_force'] = torch.tensor(0.0, device=self.device)
        else:
            penalty['mse_h_force'] = torch.sum(torch.sum(torch.square(pred['forces'] - inputs['forces']),dim=1)*h_mask.float()) / sum_h_mask

        return mse, mae, penalty

    @torch.jit.export
    def predict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''
        Used in training and unit-test
        ------------------
        edge_index: [2, Ne].
        edge_cell_shift: [Ne, 3]. Unit: Angstrom
        all_edge_index: [2, Ne_all].
        all_edge_cell_shift: [Ne_all, 3]. Unit: Angstrom
        atom_types: [Na]. torch.long
        cell: [3, 3]. Unit: Angstrom. None or negative means non-PBC
        total_charge: [Nmol]
        mol_ids: [Na].

        Outputs:
        ---------------------
        energy: [Nmol]. Unit: Kcal/mol
        nn_energy: [Nmol]. Unit: Kcal/mol
        forces: [Natom, 3]. Unit: Kcal/mol/Angstrom
        virial: [Nmol, 3, 3]. Unit: a.u.
        nn_fij:  Ne, 3, for normalization
        '''
        # Prepare data
        input_dtype = self.dtype
        for k in inputs.keys():
            if torch.is_floating_point(inputs[k]):
                input_dtype = inputs[k].dtype
                inputs[k] = inputs[k].to(self.dtype)

        natoms = len(inputs['atom_types'])
        self.nmol = int(torch.max(inputs['mol_ids']).item()) + 1
        if 'total_charge' not in inputs or inputs['total_charge'] is None:
            inputs['total_charge'] = torch.zeros(1, dtype=self.dtype, device=self.device)
        if 'mol_ids' not in inputs or inputs['mol_ids'] is None:
            inputs['mol_ids'] = torch.zeros(natoms, dtype=torch.long, device=self.device)
        row, col = inputs['edge_index'][0], inputs['edge_index'][1]
        inputs['edge_cell_shift'].requires_grad_(True)

        # NN inference
        nn_energy = self.energy_nn(inputs) # N_mol; Na

        grad_outputs : Optional[List[Optional[torch.Tensor]]] = [ torch.ones_like(nn_energy) ]
        nn_fij = torch.autograd.grad([nn_energy], [inputs['edge_cell_shift']], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0] # Ne, 3
        if nn_fij is None:
            nn_fij_cast = torch.zeros(size=inputs['edge_cell_shift'].size(), device=self.device)
        else:
            nn_fij_cast = -1.0 * nn_fij
        nn_forces = scatter(nn_fij_cast, row, dim=0, dim_size=natoms) - scatter(nn_fij_cast, col, dim=0, dim_size=natoms)
        nn_virial = nn_fij_cast.unsqueeze(-2) * inputs['edge_cell_shift'].unsqueeze(-1) # Ne, 3, 3
        nn_virial = scatter(scatter(nn_virial, row, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol) # N_mol, 3, 3

        pred = dict()
        pred['energy'] = nn_energy
        pred['forces'] = nn_forces
        pred['virial'] = nn_virial

        pred['nn_energy'] = nn_energy
        pred['nn_forces'] = nn_forces
        pred['nn_virial'] = nn_virial
        pred['nn_fij'] = nn_fij_cast

        for k, v in pred.items():
            pred[k] = v.to(input_dtype)
        return pred

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''
        Used in LAMMPS inference

        Inputs: always float64.
        ----------------
        edge_index: [2, Ne].
        edge_cell_shift: [Ne, 3]. Unit: Angstrom
        coul_edge_index: [2, Ne_coul].
        coul_edge_cell_shift: [Ne_coul, 3]. Unit: Angstrom
        disp_edge_index: [2, Ne_disp].
        disp_edge_cell_shift: [Ne_disp, 3]. Unit: Angstrom
        atom_types: [Na]. torch.long
        cell: [3, 3]. Unit: Angstrom. None or negative means non-PBC
        g_ewald: [1]. g_ewald parameter in LAMMPS

        Outputs: always float64
        -----------------
        pred_energy: [,]. Unit: Kcal/mol
        pred_forces: [Na, 3]. Unit: Kcal/mol/Angstrom
        pred_virial: [3, 3]. Unit: Kcal/mol
        '''

        # Prepare input data
        input_dtype = self.dtype
        for k in inputs.keys():
            if torch.is_floating_point(inputs[k]):
                input_dtype = inputs[k].dtype
                inputs[k] = inputs[k].to(self.dtype)

        natoms = len(inputs['atom_types'])
        self.nmol = 1
        inputs['total_charge'] = torch.zeros(1, dtype=torch.float32, device=self.device)
        inputs['mol_ids'] = torch.zeros(natoms, dtype=torch.long, device=self.device)
        row, col = inputs['edge_index'][0], inputs['edge_index'][1]
        inputs['edge_cell_shift'].requires_grad_(True) # Ne

        # NN inference
        nn_energy = self.energy_nn(inputs) # 1, Na

        # comute NN atom forces and virial
        grad_outputs : Optional[List[Optional[torch.Tensor]]] = [ torch.ones_like(nn_energy) ]
        nn_fij = torch.autograd.grad([nn_energy], [inputs['edge_cell_shift']], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0] 
        if nn_fij is None: # used for torch.jit.script
            nn_fij_cast = torch.zeros(size=inputs['edge_cell_shift'].size(), device=self.device)
        else:
            nn_fij_cast = -1.0 * nn_fij
        nn_forces = scatter(nn_fij_cast, row, dim=0, dim_size=natoms) - scatter(nn_fij_cast, col, dim=0, dim_size=natoms) 
        nn_virial = torch.sum(nn_fij_cast.unsqueeze(-2) * inputs['edge_cell_shift'].unsqueeze(-1), dim=0) 

        # dispersion energy, force and virial within cutoff
        row_disp, col_disp = inputs['disp_edge_index'][0], inputs['disp_edge_index'][1] 
        c6 = self.c6_emb(inputs['atom_types']).squeeze(-1)
        r0 = self.r0_emb(inputs['atom_types']).squeeze(-1)
        edisp, disp_fij = self.get_dispersion(row_disp, col_disp, inputs['disp_edge_cell_shift'], c6, r0)
        disp_energy = 0.5 * torch.sum(edisp) 
        disp_forces = scatter(disp_fij, row_disp, dim=0, dim_size=natoms) 
        disp_virial = 0.5 * torch.sum(disp_fij.unsqueeze(-2) * inputs['disp_edge_cell_shift'].unsqueeze(-1), dim=0) 

        # Prepare output dict
        outputs = dict()
        outputs['pred_energy'] = nn_energy + disp_energy
        outputs['pred_forces'] = nn_forces + disp_forces
        outputs['pred_virial'] = nn_virial + disp_virial
        outputs['pred_coul_energy'] = torch.zeros_like(nn_energy)
        outputs['pred_charge'] = torch.zeros_like(c6)

        if 'edge_outer_mask' in inputs.keys():
            outputs['nn_virial_outer'] = torch.sum(torch.sum(nn_fij_cast * inputs['edge_cell_shift'], dim=-1) * inputs['edge_outer_mask'])

        for k, v in outputs.items():
            outputs[k] = v.to(input_dtype)
        return outputs