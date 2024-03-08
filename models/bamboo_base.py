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

from utils.constant import (debye_ea, ele_factor, element_c6, element_r0,
                            ewald_a, ewald_f, ewald_p, nelems)
from utils.funcs import CosineCutoff, ExpNormalSmearing


class BambooBase(torch.nn.Module):
    def __init__(self, device,
                nn_params = {
                    'dim': 64,
                    'num_rbf': 32,
                    'rcut': 5.0,
                    'charge_ub': 2.0,
                    'act_fn': nn.SiLU(),
                    'charge_mlp_layers': 2,
                    'energy_mlp_layers': 2
                },
                coul_disp_params = {
                    'coul_damping_beta': 18.7,
                    'coul_damping_r0': 2.2,
                    'disp_damping_beta': 23.0,
                    'disp_cutoff': 10.0
                }):
        super(BambooBase, self).__init__()
        self.device = device
        self.nelems = nelems
        self.coul_disp_params = coul_disp_params

        # constants for ewald computation of coulomb forces
        self.ewald_f = ewald_f
        self.ewald_p = ewald_p
        self.ewald_a = ewald_a
        self.ele_factor = ele_factor
        self.debye_ea = debye_ea

        # constants for dispersion correction
        self.c6_emb = torch.nn.Embedding.from_pretrained(torch.tensor(element_c6, device=self.device, dtype=torch.float32).unsqueeze(1), freeze=True)
        self.r0_emb = torch.nn.Embedding.from_pretrained(torch.tensor(element_r0, device=self.device, dtype=torch.float32).unsqueeze(1), freeze=True)

        self.dim = nn_params['dim']
        self.num_rbf = nn_params['num_rbf']
        self.rcut = nn_params['rcut']
        self.charge_ub = nn_params['charge_ub']
        self.atom_embtab = nn.Embedding(self.nelems, self.dim)
        self.dis_rbf = ExpNormalSmearing(0.0, self.rcut, self.num_rbf, device=self.device)
        self.dis_rbf.reset_parameters()
        self.cutoff = CosineCutoff(0.0, self.rcut)

        self.charge_mlp_layers = nn_params['charge_mlp_layers']
        self.energy_mlp_layers = nn_params['energy_mlp_layers']

        def get_mlp_layers(layers: int, dim: int):
            mlp_layers = []
            for i in range(layers):
                if i == 0:
                    mlp_layers.append(nn.Linear(dim, dim//2))
                else:
                    mlp_layers.append(nn.Linear(dim//2, dim//2))
                mlp_layers.append(nn_params['act_fn'])
            mlp_layers.append(nn.Linear(dim//2, 1))
            return mlp_layers

        self.charge_mlp = nn.Sequential(*get_mlp_layers(self.charge_mlp_layers, self.dim))
        self.energy_mlp = nn.Sequential(*get_mlp_layers(self.energy_mlp_layers, self.dim))
        self.pred_electronegativity_mlp = nn.Sequential(*get_mlp_layers(self.charge_mlp_layers, self.dim))
        self.pred_electronegativity_hardness_mlp = nn.Sequential(*get_mlp_layers(self.charge_mlp_layers, self.dim))
        
        self.coul_softplus = nn.Softplus(beta = coul_disp_params['coul_damping_beta'])
        self.nmol = 1

        self.to(self.device)

    def get_coulomb(self, 
                    row: torch.Tensor,
                    col: torch.Tensor,
                    dij: torch.Tensor,
                    pred_charge: torch.Tensor,
                    g_ewald: Optional[torch.Tensor] = None,
        ) -> List[torch.Tensor]:
        '''
            Compute Coulomb energy and pairwise Coulomb force from predicted charge
        '''
        rij = torch.sqrt(torch.sum(torch.square(dij), dim=-1)) 
        prefactor_coul = self.ele_factor * pred_charge[row] * pred_charge[col] / rij 
        beta, r0 = self.coul_disp_params['coul_damping_beta'], self.coul_disp_params['coul_damping_r0']
        damp_coul = torch.sigmoid(beta / r0 * (rij - r0))

        # damping of Coulomb energy and force
        softplus_coul = self.coul_softplus((rij - r0) / r0)
        ecoul = prefactor_coul * rij / r0 / (1 + softplus_coul)
        fcoul = prefactor_coul * damp_coul * (rij / r0 / (1 + softplus_coul))**2

        # compute erfc correction in inference, not available in trianing
        if g_ewald is not None: 
            grij = g_ewald * rij 
            expm2 = torch.exp(-grij * grij) 
            t = 1.0 / (1.0 + self.ewald_p * grij) 
            erfc = t * (self.ewald_a[0] + t * (self.ewald_a[1] + t * (self.ewald_a[2] + t * (self.ewald_a[3] + t * self.ewald_a[4])))) * expm2 
            ecoul += prefactor_coul * (erfc - 1.0)
            fcoul += prefactor_coul * (erfc + self.ewald_f * grij * expm2 - 1.0) 

        coul_fij = dij * (fcoul / rij / rij).unsqueeze(-1)
        return ecoul, coul_fij

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

    def energy_nn(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        node_feat = self.atom_embtab(inputs['atom_types']) 
        coord_diff = inputs['edge_cell_shift'] 
        radial = torch.sqrt(torch.sum(coord_diff**2, 1)) 
        coord_diff = coord_diff / radial.unsqueeze(-1) 
        weights_rbf = self.dis_rbf(radial) 
        radial = self.cutoff(radial) 

        # compute electronegativity and hardness from atom embeddings
        pred_electronegativity = self.pred_electronegativity_mlp(node_feat).squeeze(-1) 
        pred_electronegativity_hardness = self.pred_electronegativity_hardness_mlp(node_feat).squeeze(-1)

        # GNN message passing
        node_feat = self.graph_nn(node_feat, inputs['edge_index'], coord_diff, radial, weights_rbf)

        # predict charge from atom embeddings and normalize the charges
        charge = self.charge_mlp(node_feat).squeeze(-1) 
        charge = self.charge_ub * torch.tanh(charge / self.charge_ub) # an upper bound of atomic partial charge 
        sum_charge = scatter(charge, inputs['mol_ids'], dim=0, dim_size=self.nmol) 
        natoms = scatter(torch.ones_like(inputs['mol_ids'], dtype=torch.float32), inputs['mol_ids'], dim=0, dim_size=self.nmol) 
        diff_charge = (inputs['total_charge'] - sum_charge)/natoms 
        pred_charge = charge + torch.gather(diff_charge, 0, inputs['mol_ids']) # make sure summation of charges is preserved

        # compute electronegativity energy
        electronegativity_energy = pred_electronegativity**2 * pred_charge + \
                                   pred_electronegativity_hardness**2 * pred_charge * pred_charge #using physical electronegative value "en_value" as starting point
        electronegativity_energy = scatter(electronegativity_energy, inputs['mol_ids'], dim=0, dim_size=self.nmol)

        # predict NN energy
        energy = self.energy_mlp(node_feat).squeeze(-1) 
        nn_energy = scatter(energy, inputs['mol_ids'], dim=0, dim_size=self.nmol) 

        return nn_energy, pred_charge, electronegativity_energy

    def get_loss(self, inputs: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        '''
        Get MSE and MAE in training and validation.
        '''
        pred = self.predict(inputs)

        # compute mean square errors with removing batch average in energy
        mse = dict()
        pred_energy_ave, label_energy_ave = torch.mean(pred['energy']), torch.mean(inputs['energy'])
        mse['energy'] = torch.mean(torch.square(pred['energy'] - inputs['energy'] - pred_energy_ave + label_energy_ave))
        mse['forces'] = torch.mean(torch.square(pred['forces'] - inputs['forces']))
        mse['virial'] = torch.mean(torch.square(pred['virial'] - inputs['virial']))
        mse['charge'] = torch.mean(torch.square(pred['charge'] - inputs['charge']))
        mse['dipole'] = torch.mean(torch.square(pred['dipole'] - inputs['dipole']))

        # compute mean sabsolute errors with removing batch average in energy
        mae = dict()
        mae['energy'] = torch.mean(torch.abs(pred['energy'] - inputs['energy'] - pred_energy_ave + label_energy_ave))
        mae['forces'] = torch.mean(torch.abs(pred['forces'] - inputs['forces']))
        mae['virial'] = torch.mean(torch.abs(pred['virial'] - inputs['virial']))
        mae['charge'] = torch.mean(torch.abs(pred['charge'] - inputs['charge']))
        mae['dipole'] = torch.mean(torch.abs(pred['dipole'] - inputs['dipole']))

        # compute charge equilibrium force penalty
        penalty = {}
        penalty['qeq_force'] = torch.mean(torch.square(pred['qeq_force']))
        
        return mse, mae, penalty

    @torch.jit.export
    def predict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''
        Used in training and validation
        Inputs units and shape:
        ------------------
        edge_index: [2, Num_edges].
        edge_cell_shift: [Num_edges, 3]. Unit: Angstrom
        all_edge_index: [2, Num_all_edges]. 
        all_edge_cell_shift: [Num_all_edges, 3]. Unit: Angstrom
        atom_types: [Num_atoms]. torch.long
        total_charge: [Num_molecules]
        mol_ids: [Num_atoms].

        Outputs units and shape:
        ---------------------
        energy: [Num_molecules]. Unit: Kcal/mol
        forces: [Num_atoms, 3]. Unit: Kcal/mol/Angstrom
        virial: [Num_molecules, 3, 3]. Unit: a.u.
        charge: [Num_atoms]. Unit: a.u.
        dipole: [Num_molecules, 3]. Unit: Debye
        qeq_forces: [Num_atoms, 3]. Unit: Kcal/mol/Angstrom. Residual in charge equilibrium which should be regularized to zero
        '''

        # prepare data
        natoms = len(inputs['atom_types'])
        self.nmol = int(torch.max(inputs['mol_ids']).item()) + 1
        row, col = inputs['edge_index'][0], inputs['edge_index'][1]
        inputs['edge_cell_shift'].requires_grad_(True)

        # NN inference
        nn_energy, pred_charge, electronegativity_energy = self.energy_nn(inputs)

        # comute NN atom forces and virial
        grad_outputs : Optional[List[Optional[torch.Tensor]]] = [ torch.ones_like(nn_energy) ]
        nn_fij = torch.autograd.grad([nn_energy], [inputs['edge_cell_shift']], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0] 
        if nn_fij is None: # used for torch.jit.script
            nn_fij_cast = torch.zeros(size=inputs['edge_cell_shift'].size(), device=self.device)
        else:
            nn_fij_cast = -1.0 * nn_fij
        nn_forces = scatter(nn_fij_cast, row, dim=0, dim_size=natoms) - scatter(nn_fij_cast, col, dim=0, dim_size=natoms)
        nn_virial = nn_fij_cast.unsqueeze(-2) * inputs['edge_cell_shift'].unsqueeze(-1) 
        nn_virial = scatter(scatter(nn_virial, row, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol)

        # compute Coulomb energy, forces and virial for all pairs
        row_all, col_all = inputs['all_edge_index'][0], inputs['all_edge_index'][1]
        ecoul, coul_fij = self.get_coulomb(row_all, col_all, inputs['all_edge_cell_shift'], pred_charge)
        coul_energy = 0.5 * scatter(scatter(ecoul, row_all, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol) 
        coul_forces = scatter(coul_fij, row_all, dim=0, dim_size=natoms) 
        coul_virial = 0.5 * scatter(scatter(coul_fij.unsqueeze(-2) * inputs['all_edge_cell_shift'].unsqueeze(-1), row_all, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol) 

        # compute residual in charge equilibrium formula, which should be regularized to zero
        grad_outputs : Optional[List[Optional[torch.Tensor]]] = [ torch.ones_like(nn_energy) ]
        charge_energy = coul_energy + electronegativity_energy
        qeq_fij = torch.autograd.grad([charge_energy], [inputs['edge_cell_shift']], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0] 
        if qeq_fij is None: # used for torch.jit.script
            qeq_fij_cast = torch.zeros(size=inputs['edge_cell_shift'].size(), device=self.device)
        else:
            qeq_fij_cast = -1.0 * qeq_fij
        qeq_force = scatter(qeq_fij_cast, row, dim=0, dim_size=natoms) - scatter(qeq_fij_cast, col, dim=0, dim_size=natoms) 
            
        # prepare output dictionary
        pred = dict()
        pred['energy'] = nn_energy + coul_energy + electronegativity_energy
        pred['forces'] = nn_forces + coul_forces
        pred['virial'] = nn_virial + coul_virial
        pred['charge'] = pred_charge
        pred['dipole'] = scatter(inputs['pos'] * pred_charge.unsqueeze(-1), inputs['mol_ids'], dim=0, dim_size=self.nmol) / self.debye_ea 
        pred['qeq_force'] = qeq_force
        return pred

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ''' 
        Used in LAMMPS inference

        Inputs: always float64. 
        ----------------
        edge_index: [2, Num_edges].
        edge_cell_shift: [Num_edges, 3]. Unit: Angstrom
        coul_edge_index: [2, Num_coul_edges]. 
        coul_edge_cell_shift: [Num_coul_edges, 3]. Unit: Angstrom
        disp_edge_index: [2, Num_disp_edges]. 
        disp_edge_cell_shift: [Num_disp_edges, 3]. Unit: Angstrom
        atom_types: [Num_atoms]. torch.long
        g_ewald: [1]. g_ewald parameter in LAMMPS

        Outputs: always float64
        -----------------
        pred_energy: [1]. Unit: Kcal/mol
        pred_forces: [Num_atoms, 3]. Unit: Kcal/mol/Angstrom
        pred_virial: [3, 3]. Unit: Kcal/mol
        coul_energy: [1]. Unit: Kcal/mol
        pred_charge: [Num_atoms]. Unit: a.u.
        '''

        # Prepare input data
        for k in inputs.keys():
            if torch.is_floating_point(inputs[k]):
                inputs[k] = inputs[k].to(torch.float32)

        natoms = len(inputs['atom_types'])
        self.nmol = 1
        inputs['total_charge'] = torch.zeros(1, dtype=torch.float32, device=self.device)
        inputs['mol_ids'] = torch.zeros(natoms, dtype=torch.long, device=self.device)
        row, col = inputs['edge_index'][0], inputs['edge_index'][1]
        inputs['edge_cell_shift'].requires_grad_(True) # Ne

        # NN inference
        nn_energy, pred_charge, electronegativity_energy = self.energy_nn(inputs) # 1, Na
        
        # comute NN atom forces and virial
        grad_outputs : Optional[List[Optional[torch.Tensor]]] = [ torch.ones_like(nn_energy) ]
        nn_fij = torch.autograd.grad([nn_energy], [inputs['edge_cell_shift']], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0] 
        if nn_fij is None: # used for torch.jit.script
            nn_fij_cast = torch.zeros(size=inputs['edge_cell_shift'].size(), device=self.device)
        else:
            nn_fij_cast = -1.0 * nn_fij
        nn_forces = scatter(nn_fij_cast, row, dim=0, dim_size=natoms) - scatter(nn_fij_cast, col, dim=0, dim_size=natoms) 
        nn_virial = torch.sum(nn_fij_cast.unsqueeze(-2) * inputs['edge_cell_shift'].unsqueeze(-1), dim=0) 

        # Coulomb energy, force and virial within cutoff
        row_coul, col_coul = inputs['coul_edge_index'][0], inputs['coul_edge_index'][1]
        ecoul, coul_fij = self.get_coulomb(row_coul, col_coul, inputs['coul_edge_cell_shift'], pred_charge, g_ewald=inputs['g_ewald'])
        coul_energy = 0.5 * torch.sum(ecoul) 
        coul_forces = scatter(coul_fij, row_coul, dim=0, dim_size=natoms) 
        coul_virial = 0.5 * torch.sum(coul_fij.unsqueeze(-2) * inputs['coul_edge_cell_shift'].unsqueeze(-1), dim=0)
        
        # dispersion energy, force and virial within cutoff
        row_disp, col_disp = inputs['disp_edge_index'][0], inputs['disp_edge_index'][1] 
        c6 = self.c6_emb(inputs['atom_types']).squeeze(-1)
        r0 = self.r0_emb(inputs['atom_types']).squeeze(-1)
        edisp, disp_fij = self.get_dispersion(row_disp, col_disp, inputs['disp_edge_cell_shift'], c6, r0)
        disp_energy = 0.5 * torch.sum(edisp) 
        disp_forces = scatter(disp_fij, row_disp, dim=0, dim_size=natoms) 
        disp_virial = 0.5 * torch.sum(disp_fij.unsqueeze(-2) * inputs['disp_edge_cell_shift'].unsqueeze(-1), dim=0) 
            
        # prepare output dictionary and convert back to float64
        outputs = dict()
        outputs['pred_energy'] = nn_energy + coul_energy + disp_energy + electronegativity_energy 
        outputs['pred_forces'] = nn_forces + coul_forces + disp_forces 
        outputs['pred_virial'] = nn_virial + coul_virial + disp_virial 
        outputs['pred_coul_energy'] = coul_energy
        outputs['pred_charge'] = pred_charge

        if 'edge_outer_mask' in inputs.keys():
            outputs['nn_virial_outer'] = torch.sum(torch.sum(nn_fij_cast * inputs['edge_cell_shift'], dim=-1) * inputs['edge_outer_mask'])

        for k, v in outputs.items():
            outputs[k] = v.to(torch.float64)
        return outputs