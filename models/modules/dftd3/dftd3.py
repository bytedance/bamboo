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

import os
from typing import Optional

import torch
import torch.nn as nn

from utils import constant
from torch_runstats.scatter import scatter

INITIAL_PARAMS_PATH = os.path.join(os.path.dirname(__file__), 'dftd3.pt')


class DFTD3Params(nn.Module):
    def __init__(self, dtype=torch.double):
        super().__init__()
        self.default_params = torch.load(INITIAL_PARAMS_PATH, map_location='cpu')

        self.ntype = 90
        self.nweight = 5

        self.sqrt_z_r4r2 = nn.Parameter(torch.empty(self.ntype, dtype=dtype), requires_grad=False)
        self.cov_d3 = nn.Parameter(torch.empty(self.ntype, dtype=dtype), requires_grad=False)
        self.cn = nn.Parameter(torch.empty(self.ntype, self.nweight, dtype=dtype), requires_grad=False)
        self.c6 = nn.Parameter(torch.empty(self.ntype, self.ntype, self.nweight, self.nweight, dtype=dtype), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.sqrt_z_r4r2.copy_(self.default_params['sqrt_z_r4_over_r2'].to(self.sqrt_z_r4r2))
            self.cov_d3.copy_(self.default_params['cov_d3'].to(self.cov_d3))
            self.cn.copy_(self.default_params['cn'].to(self.cn))
            self.c6.copy_(self.default_params['c6'].to(self.c6))


class DFTD3Base(nn.Module):
    def __init__(self, dftd3_params=None, disp_cutoff=50.0):
        super().__init__()
        self.disp_cutoff = disp_cutoff / constant.bohr_angstrom  # disp_cutoff is specified in angstrom
        self.params = dftd3_params or DFTD3Params()

    def reset_parameters(self):
        self.params.reset_parameters()

    def cn_d3(self, atom_types, row, col, rij):
        KCN = 16.0
        rcov = self.params.cov_d3[atom_types]
        rc = rcov[row] + rcov[col]
        cf = 1.0 / (1.0 + torch.exp(-KCN * (rc / rij - 1.0)))
        cn = scatter(cf, row, dim=0, dim_size=atom_types.shape[0]) + scatter(cf, col, dim=0, dim_size=atom_types.shape[0])
        d_cn = -KCN * cf * (1.0 - cf) * rc / (rij ** 2)
        return cn, d_cn

    def weight_references(self, atom_types, cn):
        refcn = self.params.cn[atom_types]
        dcn = refcn - cn.unsqueeze(-1)

        dcn = dcn.double()

        factor = 4.0
        weights = torch.exp(-factor * dcn.pow(2))

        norm = weights.sum(dim=-1, keepdim=True)
        normalized_weights = weights / norm

        d_weights = 2.0 * factor * dcn * weights
        norm_d = d_weights.sum(dim=-1, keepdim=True)
        d_1_over_norm = - norm_d / norm ** 2
        d_normalized_weights = d_weights / norm + weights * d_1_over_norm

        normalized_weights = normalized_weights.type(cn.dtype)
        d_normalized_weights = d_normalized_weights.type(cn.dtype)
        assert normalized_weights.isnan().sum() == 0
        assert d_normalized_weights.isnan().sum() == 0

        one = torch.tensor(1.0, device=cn.device, dtype=cn.dtype)
        zero = torch.tensor(0.0, device=cn.device, dtype=cn.dtype)
        maxcn = torch.max(refcn, dim=-1, keepdim=True)[0]
        is_exceptional = (norm == 0) | (normalized_weights > 1e50)
        normalized_weights = torch.where(is_exceptional, torch.where(refcn == maxcn, one, zero), normalized_weights)
        d_normalized_weights = torch.where(is_exceptional, zero, d_normalized_weights)

        return normalized_weights, d_normalized_weights

    def compute_c6(self, atom_types, weights, dweights, row, col):
        rc6 = self.params.c6[atom_types[col], atom_types[row]].contiguous()

        rc6_mask = rc6 != 0.0
        rc6 = rc6 * rc6_mask

        rc6_wc = (rc6 * weights[col].unsqueeze(-1)).sum(dim=-2)
        rc6_wr = (rc6 * weights[row].unsqueeze(-2)).sum(dim=-1)

        c6ij = (weights[row] * rc6_wc).sum(dim=-1)

        d_c6ij_dcni = (dweights[row] * rc6_wc).sum(dim=-1)
        d_c6ij_dcnj = (dweights[col] * rc6_wr).sum(dim=-1)
        return c6ij, d_c6ij_dcni, d_c6ij_dcnj

    def forward(
        self,
        atom_types: torch.Tensor,
        dij: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        # note that cutoff is already applied in LAMMPS
        edge_mask = edge_index[1] > edge_index[0]
        edge_index = edge_index[:, edge_mask]
        dij = dij[edge_mask]

        # kernel
        row, col = edge_index.unbind(0)
        dij = dij / constant.bohr_angstrom
        rij = dij.norm(dim=-1, p=2)

        cn, d_cn_d_rij = self.cn_d3(atom_types, row, col, rij)
        weights, d_weights_d_cn = self.weight_references(atom_types, cn)
        c6ij, d_c6ij_d_cni, d_c6ij_d_cnj = self.compute_c6(atom_types, weights, d_weights_d_cn, row, col)

        num_r4r2 = self.params.sqrt_z_r4r2[atom_types]
        kij = num_r4r2[row] * num_r4r2[col] * 3
        r0ij = torch.sqrt(kij)

        # call the implementation of damping function in subclass
        edisp, gdisp = self.compute_dispersion(rij, r0ij, kij)

        # results w/o contributions from cn
        energy = scatter(- edisp * c6ij, row, dim=0, dim_size=atom_types.shape[0])
        dE_drij = - c6ij * gdisp

        # add contributions from cn
        dE_dcn = (
            scatter(-d_c6ij_d_cni * edisp, row, dim=0, dim_size=atom_types.shape[0])
            + scatter(-d_c6ij_d_cnj * edisp, col, dim=0, dim_size=atom_types.shape[0])
        )
        d_E_d_cn_d_cn_d_rij = d_cn_d_rij * dE_dcn[edge_index].sum(dim=0)
        dE_drij = dE_drij + d_E_d_cn_d_cn_d_rij

        # calculate forces and virial
        dE_ddij = (dE_drij / rij).unsqueeze(-1) * dij
        gradients = scatter(dE_ddij, row, dim=0, dim_size=atom_types.shape[0]) + scatter(-dE_ddij, col, dim=0, dim_size=atom_types.shape[0])
        forces = - gradients
        virial = - (dE_ddij.unsqueeze(-2) * dij.unsqueeze(-1))

        # aggregate to mol level
        energy = energy.sum()
        virial = virial.sum(dim=0)

        # convert the units back to kcal/mol and angstrom
        energy = energy * constant.hartree_kcal_mol
        forces = forces * constant.hartree_kcal_mol / constant.bohr_angstrom
        virial = virial * constant.hartree_kcal_mol

        return energy, forces, virial


class DFTD3CSO(DFTD3Base):
    def __init__(self, s6=1.0, a1=0.86, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s6 = s6
        self.a1 = a1

    def compute_dispersion(self, rij, r0ij, kij):
        # note that the a4=6.25 is changed to 7.75 here
        e = torch.exp(rij - 2.5 * r0ij)
        e1 = 1 + e
        t = 1 / (rij ** 6 + 7.75 ** 6)
        m = self.s6 + self.a1 / e1
        dt = - 6 * rij ** 5 * (t ** 2)
        dm = - self.a1 * e / (e1**2)
        edisp = m * t
        fdisp = dm * t + m * dt
        return edisp, fdisp
