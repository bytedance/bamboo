import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_runstats.scatter import scatter

from utils.constant import (debye_ea, ele_factor, electronegativity_table,
                            element_c6, element_r0, ewald_a, ewald_f, ewald_p,
                            nelems, oplsaa_charge)
from utils.dispersion import Dispersion, apply_dispersion
from utils.funcs import CosineCutoff, ExpNormalSmearing

SQRT_3 = 1.73205080757

class BambooBase(torch.nn.Module):
    def __init__(self, device, dtype, 
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
                    'disp_cutoff': 10.0,
                    'disp_training': Dispersion.NO.name,
                    'disp_simulation': Dispersion.D3CSO.name
                }):
        super(BambooBase, self).__init__()
        self.device = device
        self.dtype = dtype
        self.nelems = nelems
        self.disp_training = coul_disp_params.pop('disp_training') # str type
        self.disp_simulation = coul_disp_params.pop('disp_simulation') # str type
        self.coul_disp_params = coul_disp_params

        # Constants for ecoul
        self.ewald_f = ewald_f
        self.ewald_p = ewald_p
        self.ewald_a = ewald_a
        self.ele_factor = ele_factor
        self.debye_ea = debye_ea
        self.sqrt_3 = SQRT_3

        # constants for electronegativity
        self.electronegativity_table = torch.nn.Embedding.from_pretrained(torch.tensor(electronegativity_table, device=device, dtype=dtype).unsqueeze(1), freeze=True)
        # constants for oplsaa charge
        self.oplsaa_charge = torch.nn.Embedding.from_pretrained(torch.tensor(oplsaa_charge, device=device, dtype=dtype).unsqueeze(1), freeze=True)

        self.c6_emb = torch.nn.Embedding.from_pretrained(torch.tensor(element_c6, device=device, dtype=dtype).unsqueeze(1), freeze=True)
        self.r0_emb = torch.nn.Embedding.from_pretrained(torch.tensor(element_r0, device=device, dtype=dtype).unsqueeze(1), freeze=True)

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

    def get_quadrupole(self, inputs: Dict[str, torch.Tensor], pred_charge: torch.Tensor) -> torch.Tensor:
        quadrupole = []
        xs, ys, zs = inputs['pos'][:,0], inputs['pos'][:,1], inputs['pos'][:,2]
        r2 = xs*xs + ys*ys + zs*zs
        quadrupole.append(0.5 * pred_charge * (3*zs*zs - r2)) # Na
        quadrupole.append(self.sqrt_3 * pred_charge * (xs*zs)) # Na
        quadrupole.append(self.sqrt_3 * pred_charge * (ys*zs))  # Na
        quadrupole.append(0.5 * self.sqrt_3 * pred_charge * (xs*xs - ys*ys)) # Na
        quadrupole.append(-1.0 * self.sqrt_3 * pred_charge * (xs*ys)) # Na
        return scatter(torch.stack(quadrupole, dim=-1), inputs['mol_ids'], dim=0) / self.debye_ea

    def get_coulomb(self, row: torch.Tensor,
                          col: torch.Tensor,
                          dij: torch.Tensor,
                          pred_charge: torch.Tensor,
                          g_ewald: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        rij = torch.sqrt(torch.sum(torch.square(dij), dim=-1)) 
        prefactor_coul = self.ele_factor * pred_charge[row] * pred_charge[col] / rij # Ne
        beta, r0 = self.coul_disp_params['coul_damping_beta'], self.coul_disp_params['coul_damping_r0']
        damp_coul = torch.sigmoid(beta / r0 * (rij - r0)) # Ne

        # coul damping
        softplus_coul = self.coul_softplus((rij - r0) / r0)
        ecoul = prefactor_coul * rij / r0 / (1 + softplus_coul)
        fcoul = prefactor_coul * damp_coul * (rij / r0 / (1 + softplus_coul))**2

        if g_ewald is not None: # erfc term in forward() not in predict()
            grij = g_ewald * rij # Ne_coul
            expm2 = torch.exp(-grij * grij) # Ne_coul
            t = 1.0 / (1.0 + self.ewald_p * grij) # Ne_coul
            erfc = t * (self.ewald_a[0] + t * (self.ewald_a[1] + t * (self.ewald_a[2] + t * (self.ewald_a[3] + t * self.ewald_a[4])))) * expm2 # Ne_coul
            ecoul += prefactor_coul * (erfc - 1.0)
            fcoul += prefactor_coul * (erfc + self.ewald_f * grij * expm2 - 1.0) # Ne_coul

        coul_fij = dij * (fcoul / rij / rij).unsqueeze(-1) # Ne_coul, 3
        return ecoul, coul_fij

    def get_dispersion(self, row: torch.Tensor,
                         col: torch.Tensor,
                         dij: torch.Tensor,
                         c6: torch.Tensor,
                         r0: torch.Tensor,
                         disp_method: str) -> List[torch.Tensor]:
        rij = torch.sqrt(torch.sum(torch.square(dij), dim=-1)) 
        c6ij = torch.sqrt(c6[row] * c6[col]) # Ne
        r0ij = 0.5*(r0[row] + r0[col]) # Ne
        c8ij = c6ij * r0ij ** 2 # Ne
        
        edisp, disp_fij, c6ij = apply_dispersion(disp_method, dij, rij, c6ij, r0ij, c8ij)
        edisp += c6ij / self.coul_disp_params['disp_cutoff']**6

        return edisp, disp_fij

    def graph_nn(self, node_feat: torch.Tensor, 
                       edge_index: torch.Tensor, 
                       coord_diff: torch.Tensor, 
                       radial: torch.Tensor, 
                       weights_rbf: torch.Tensor) -> torch.Tensor:
        '''
            Update node feature by GNN
            node_feat: [Na, D]
            edge_index: [2, Ne]
            coord_diff: [Ne, 3]
            radial: [Ne]
            weights_rbf: [Ne, Nrbf]
        '''
        raise NotImplementedError('graph_nn is not implemented')

    def energy_nn(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        node_feat = self.atom_embtab(inputs['atom_types']) # Na, D
        coord_diff = inputs['edge_cell_shift'] # Ne, 3
        radial = torch.sqrt(torch.sum(coord_diff**2, 1)) # Ne
        coord_diff = coord_diff / radial.unsqueeze(-1) # Ne, 3
        weights_rbf = self.dis_rbf(radial) # Ne, N_rbf
        radial = self.cutoff(radial) # Ne,1

        pred_electronegativity = self.pred_electronegativity_mlp(node_feat).squeeze(-1)  # Na
        pred_electronegativity_hardness = self.pred_electronegativity_hardness_mlp(node_feat).squeeze(-1) #Na

        # GNN message passing
        node_feat = self.graph_nn(node_feat, inputs['edge_index'], coord_diff, radial, weights_rbf)

        # Charge and energy prediction
        oplsaa_charge_value = self.oplsaa_charge(inputs['atom_types']).squeeze(-1) # Na
        charge = self.charge_mlp(node_feat).squeeze(-1)  # Na
        charge = charge + oplsaa_charge_value # oplsaa charge as starting point
        charge = self.charge_ub * torch.tanh(charge / self.charge_ub) # an upper bound of atomic partial charge 
        sum_charge = scatter(charge, inputs['mol_ids'], dim=0, dim_size=self.nmol) # Nm
        natoms = scatter(torch.ones_like(inputs['mol_ids'], dtype=torch.float32), inputs['mol_ids'], dim=0, dim_size=self.nmol) # Nm
        diff_charge = (inputs['total_charge'] - sum_charge)/natoms # Nm
        pred_charge = charge + torch.gather(diff_charge, 0, inputs['mol_ids']) # Na

        # Electronegativity energy
        electronegativity_value = self.electronegativity_table(inputs['atom_types']).squeeze(-1) # electronegativity_value in physics
        electronegativity_energy = ((electronegativity_value**0.5+pred_electronegativity)**2) * pred_charge + \
                                   (pred_electronegativity_hardness**2) * pred_charge * pred_charge #Na; using physical electronegative value "en_value" as starting point
        electronegativity_energy = scatter(electronegativity_energy, inputs['mol_ids'], dim=0, dim_size=self.nmol) #Nm

        # Energy prediction
        energy = self.energy_mlp(node_feat).squeeze(-1) # Na
        nn_energy = scatter(energy, inputs['mol_ids'], dim=0, dim_size=self.nmol) # Nm

        return nn_energy, pred_charge, electronegativity_energy


    def __call__(self, inputs) -> List[Dict[str, torch.Tensor]]:
        return self.get_loss(inputs)

    def get_loss(self, inputs) -> List[Dict[str, torch.Tensor]]:
        '''
        Get MSE and MAE.
        '''
        pred = self.predict(inputs)
        mse = {}
        pred_energy_ave, label_energy_ave = torch.mean(pred['energy']), torch.mean(inputs['energy'])
        mse['energy'] = torch.mean(torch.square(pred['energy'] - inputs['energy'] - pred_energy_ave + label_energy_ave))
        mse['forces'] = torch.mean(torch.square(pred['forces'] - inputs['forces']))
        mse['virial'] = torch.mean(torch.square(pred['virial'] - inputs['virial']))
        mse['charge'] = torch.mean(torch.square(pred['charge'] - inputs['charge']))
        mse['dipole'] = torch.mean(torch.square(pred['dipole'] - inputs['dipole']))

        mae = {}
        mae['energy'] = torch.mean(torch.abs(pred['energy'] - inputs['energy'] - pred_energy_ave + label_energy_ave))
        mae['forces'] = torch.mean(torch.abs(pred['forces'] - inputs['forces']))
        mae['virial'] = torch.mean(torch.abs(pred['virial'] - inputs['virial']))
        mae['charge'] = torch.mean(torch.abs(pred['charge'] - inputs['charge']))
        mae['dipole'] = torch.mean(torch.abs(pred['dipole'] - inputs['dipole']))

        penalty = {}

        penalty['macharge'] = torch.mean(torch.abs(pred['charge']))
        penalty['msnnfij'] = torch.mean(torch.square(pred['nn_fij']))

        penalty['decoul_dr_force'] = torch.mean(torch.square(pred['decoul_dr_force']))

        h_mask = inputs['atom_types'] == 1
        penalty['mse_h_force'] = torch.sum(torch.sum(torch.square(pred['forces'] - inputs['forces']),dim=1)*h_mask.float()) / torch.sum(h_mask.float())
        
        return mse, mae, penalty

    def get_mse_loss(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = self.predict(inputs)
        mse = {}
        pred_energy_ave, label_energy_ave = torch.mean(pred['energy']), torch.mean(inputs['energy'])
        mse['energy'] = torch.mean(torch.square(pred['energy'] - inputs['energy'] - pred_energy_ave + label_energy_ave))
        mse['forces'] = torch.mean(torch.square(pred['forces'] - inputs['forces']))
        mse['virial'] = torch.mean(torch.square(pred['virial'] - inputs['virial']))
        mse['charge'] = torch.mean(torch.square(pred['charge'] - inputs['charge']))
        mse['dipole'] = torch.mean(torch.square(pred['dipole'] - inputs['dipole']))

        return mse

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
        coul_energy: [Nmol]. Unit: Kcal/mol
        disp_energy: [Nmol]. Unit: Kcal/mol
        forces: [Natom, 3]. Unit: Kcal/mol/Angstrom
        virial: [Nmol, 3, 3]. Unit: a.u.
        charge: [Natom]. Unit: a.u.
        dipole: [Nmol, 3]. Unit: Debye
        quadrupole: [Nmol, 3, 3]. Unit: Debye*Angstrom
        nn_fij_cast:  Ne, 3, for normalization
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
        nn_energy, pred_charge, electronegativity_energy = self.energy_nn(inputs) # N_mol; Na

        grad_outputs : Optional[List[Optional[torch.Tensor]]] = [ torch.ones_like(nn_energy) ]
        nn_fij = torch.autograd.grad([nn_energy], [inputs['edge_cell_shift']], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0] # Ne, 3
        if nn_fij is None:
            nn_fij_cast = torch.zeros(size=inputs['edge_cell_shift'].size(), device=self.device)
        else:
            nn_fij_cast = -1.0 * nn_fij
        nn_forces = scatter(nn_fij_cast, row, dim=0, dim_size=natoms) - scatter(nn_fij_cast, col, dim=0, dim_size=natoms)
        nn_virial = nn_fij_cast.unsqueeze(-2) * inputs['edge_cell_shift'].unsqueeze(-1) # Ne, 3, 3
        nn_virial = scatter(scatter(nn_virial, row, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol) # N_mol, 3, 3

        # Coulomb energy, forces and virial for all pairs
        row_all, col_all = inputs['all_edge_index'][0], inputs['all_edge_index'][1]
        ecoul, coul_fij = self.get_coulomb(row_all, col_all, inputs['all_edge_cell_shift'], pred_charge)

        coul_energy = 0.5 * scatter(scatter(ecoul, row_all, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol) # N_mol
        coul_forces = scatter(coul_fij, row_all, dim=0, dim_size=natoms) # Na, 3
        coul_virial = 0.5 * scatter(scatter(coul_fij.unsqueeze(-2) * inputs['all_edge_cell_shift'].unsqueeze(-1), row_all, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol) # N_mol, 3, 3

        grad_outputs : Optional[List[Optional[torch.Tensor]]] = [ torch.ones_like(nn_energy) ]
        charge_energy = coul_energy+electronegativity_energy
        decoul_dr_fij = torch.autograd.grad([charge_energy], [inputs['edge_cell_shift']], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0] # Ne, 3
        if decoul_dr_fij is None:
            decoul_dr_fij_cast = torch.zeros(size=inputs['edge_cell_shift'].size(), device=self.device)
        else:
            decoul_dr_fij_cast = -1.0 * decoul_dr_fij
        decoul_dr_force = scatter(decoul_dr_fij_cast, row, dim=0, dim_size=natoms) - scatter(decoul_dr_fij_cast, col, dim=0, dim_size=natoms) # Na, 3

       # Dispersion energy, forces and virial for all pairs
        c6 = self.c6_emb(inputs['atom_types']).squeeze(-1)
        r0 = self.r0_emb(inputs['atom_types']).squeeze(-1)
        
        if self.disp_training != Dispersion.NO.name:
            edisp, disp_fij = self.get_dispersion(row_all, col_all, inputs['all_edge_cell_shift'], 
                                c6, r0, disp_method=self.disp_training)
            disp_energy = 0.5 * scatter(scatter(edisp, row_all, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol) # N_mol
            disp_forces = scatter(disp_fij, row_all, dim=0, dim_size=natoms) # Na, 3 
            disp_virial = 0.5 * scatter(scatter(disp_fij.unsqueeze(-2) * inputs['all_edge_cell_shift'].unsqueeze(-1), row_all, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol) # N_mol, 3, 3

            
        pred = dict()
        pred['energy'] = nn_energy + coul_energy + electronegativity_energy
        pred['forces'] = nn_forces + coul_forces
        pred['virial'] = nn_virial + coul_virial
        pred['disp_energy'] = torch.zeros_like(nn_energy)
    
        # Avoid zero tensor create to save time.
        if self.disp_training != Dispersion.NO.name:
            edisp, disp_fij = self.get_dispersion(row_all, col_all, inputs['all_edge_cell_shift'], 
                                c6, r0, disp_method=self.disp_training)
            disp_energy = 0.5 * scatter(scatter(edisp, row_all, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol) # N_mol
            disp_forces = scatter(disp_fij, row_all, dim=0, dim_size=natoms) # Na, 3 
            disp_virial = 0.5 * scatter(scatter(disp_fij.unsqueeze(-2) * inputs['all_edge_cell_shift'].unsqueeze(-1), row_all, dim=0, dim_size=natoms), inputs['mol_ids'], dim=0, dim_size=self.nmol) # N_mol, 3, 3
            pred['energy'] += disp_energy
            pred['forces'] += disp_forces 
            pred['virial'] += disp_virial
            pred['disp_energy'] = disp_energy

        pred['charge'] = pred_charge
        pred['dipole'] = scatter(inputs['pos'] * pred_charge.unsqueeze(-1), inputs['mol_ids'], dim=0, dim_size=self.nmol) / self.debye_ea # Nm, 3
        pred['nn_energy'] = nn_energy
        pred['coul_energy'] = coul_energy
        pred['electronegativity_energy'] = electronegativity_energy
  
        pred['coul_forces'] = coul_forces
        pred['coul_virial'] = coul_virial
        pred['nn_forces'] = nn_forces
        pred['nn_virial'] = nn_virial
        pred['nn_fij'] = nn_fij_cast

        pred['decoul_dr_force'] = decoul_dr_force

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
        coul_energy: [,]. Unit: Kcal/mol
        pred_charge: [Na]. Unit: a.u.
        '''

        # Prepare input data
        input_dtype = self.dtype
        for k in inputs.keys():
            if torch.is_floating_point(inputs[k]):
                input_dtype = inputs[k].dtype
                inputs[k] = inputs[k].to(self.dtype)

        natoms = len(inputs['atom_types'])
        self.nmol = 1
        inputs['total_charge'] = torch.zeros(1, dtype=self.dtype, device=self.device)
        inputs['mol_ids'] = torch.zeros(natoms, dtype=torch.long, device=self.device)
        row, col = inputs['edge_index'][0], inputs['edge_index'][1]
        inputs['edge_cell_shift'].requires_grad_(True) # Ne

        # NN inference
        nn_energy, pred_charge, electronegativity_energy = self.energy_nn(inputs) # 1, Na
        
        # Coulomb energy, force and virial within cutoff
        row_coul, col_coul = inputs['coul_edge_index'][0], inputs['coul_edge_index'][1]
        ecoul, coul_fij = self.get_coulomb(row_coul, col_coul, inputs['coul_edge_cell_shift'], 
                            pred_charge, g_ewald=inputs['g_ewald'])

        coul_energy = 0.5 * torch.sum(ecoul) # 1
        coul_forces = scatter(coul_fij, row_coul, dim=0, dim_size=natoms)  # Ne_coul, 3
        coul_virial = 0.5 * torch.sum(coul_fij.unsqueeze(-2) * inputs['coul_edge_cell_shift'].unsqueeze(-1), dim=0) # 3, 3
        
        grad_outputs : Optional[List[Optional[torch.Tensor]]] = [ torch.ones_like(nn_energy) ]
        nn_fij = torch.autograd.grad([nn_energy], [inputs['edge_cell_shift']], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0] # Ne, 3
        if nn_fij is None:
            nn_fij_cast = torch.zeros(size=inputs['edge_cell_shift'].size(), device=self.device)
        else:
            nn_fij_cast = -1.0 * nn_fij
        nn_forces = scatter(nn_fij_cast, row, dim=0, dim_size=natoms) - scatter(nn_fij_cast, col, dim=0, dim_size=natoms) # Na, 3
        nn_virial = torch.sum(nn_fij_cast.unsqueeze(-2) * inputs['edge_cell_shift'].unsqueeze(-1), dim=0) # 3, 3

        # dispersion energy, force and virial within cutoff
        row_disp, col_disp = inputs['disp_edge_index'][0], inputs['disp_edge_index'][1] # Ne_disp; Ne_disp
        c6 = self.c6_emb(inputs['atom_types']).squeeze(-1) # Na
        r0 = self.r0_emb(inputs['atom_types']).squeeze(-1) # Na
        edisp, disp_fij = self.get_dispersion(row_disp, col_disp, inputs['disp_edge_cell_shift'], 
                            c6, r0, disp_method=self.disp_simulation)
        
        disp_energy = 0.5 * torch.sum(edisp) # 1
        disp_forces = scatter(disp_fij, row_disp, dim=0, dim_size=natoms) # Na, 3
        disp_virial = 0.5 * torch.sum(disp_fij.unsqueeze(-2) * inputs['disp_edge_cell_shift'].unsqueeze(-1), dim=0) # 3, 3
            
        # Prepare output dict
        outputs = dict()
        outputs['pred_energy'] = nn_energy + coul_energy + disp_energy + electronegativity_energy # 1
        outputs['pred_forces'] = nn_forces + coul_forces + disp_forces # Na, 3
        
        outputs['pred_virial'] = nn_virial + coul_virial + disp_virial # 3, 3
        outputs['pred_coul_energy'] = coul_energy
        outputs['pred_charge'] = pred_charge

        if 'edge_outer_mask' in inputs.keys():
            outputs['nn_virial_outer'] = torch.sum(torch.sum(nn_fij_cast * inputs['edge_cell_shift'], dim=-1) * inputs['edge_outer_mask'])
        
        # # remove # for test
        # outputs['pred_nn_energy'] = nn_energy
        # outputs['pred_disp_energy'] = disp_energy
        # outputs['pred_nn_virial'] = nn_virial
        # outputs['pred_coul_virial'] = coul_virial
        # outputs['pred_disp_virial'] = disp_virial
        # outputs['pred_nn_fij'] = nn_fij_cast # Ne, 3
        # outputs['pred_coul_fij'] = coul_fij 
        # outputs['pred_disp_fij'] = disp_fij
        # outputs['pred_nn_fi'] = nn_forces # Na, 3
        # outputs['pred_coul_fi'] = coul_forces # Na, 3
        # outputs['pred_disp_fi'] = disp_forces # Na, 3
        # outputs['disp_energy'] = disp_energy
        # outputs['disp_forces'] = disp_forces
        # outputs['disp_virial'] = disp_virial
        # outputs['coul_energy'] = coul_energy
        # outputs['coul_forces'] = coul_forces
        # outputs['coul_virial'] = coul_virial
        # outputs['nn_energy'] = nn_energy
        # outputs['nn_forces'] = nn_forces
        # outputs['nn_virial'] = nn_virial

        for k, v in outputs.items():
            outputs[k] = v.to(input_dtype)
        return outputs
