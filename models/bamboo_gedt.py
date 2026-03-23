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

from typing import List

import torch
import torch.nn as nn
from torch_runstats.scatter import scatter

from models.bamboo_base import BambooBase


class LinearAttnFirst(nn.Module):
    """
    Graph Equivariant Transformer First Layer
    No node_vec in the input compared to middle layers
    """
    def __init__(self, dim, num_heads, act_fn=nn.GELU()):
        super(LinearAttnFirst, self).__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.o_proj = nn.Linear(dim, dim)
        self.x_proj = nn.Linear(dim, dim)

        self.layer_norm = nn.LayerNorm(dim)
        self.dim = dim
        self.dim_per_head = dim // num_heads
        self.num_heads = num_heads
        self.attn_act = act_fn

    def qkv_attn(self, node_feat, row, col):
        node_feat = self.layer_norm(node_feat) # Na, D
        qkv = self.qkv_proj(node_feat) # Na, 3D
        qkv = qkv.reshape(qkv.shape[:-1]+(self.num_heads, self.dim_per_head * 3)) # Na, H, 3D/H

        q1, q2, k1, k2, v = qkv[...,:self.dim_per_head//2], qkv[...,self.dim_per_head//2:self.dim_per_head], qkv[...,self.dim_per_head:(3*self.dim_per_head)//2], qkv[...,(3*self.dim_per_head)//2:2*self.dim_per_head], qkv[...,2*self.dim_per_head:]

        q_row_1 = q1[row] # Ne, H, D/H
        q_row_2 = q2[row] # Ne, H, D/H
        k_col_1 = k1[col] # Ne, H, D/H 
        k_col_2 = k2[col] # Ne, H, D/H
        v_col = v[col] # Ne, H, D/H
        return q_row_1, q_row_2, k_col_1, k_col_2, v_col

    def xv_proc(self, v_col, attn, dv, dvdij, row, natoms: int):
        v = v_col * dv  # Ne, H, D/H;
        x = v * attn.unsqueeze(-1)  # Ne, H, D/H;
        x = scatter(x, row, dim=0, dim_size=natoms) # Na, H, D/H
        x = x.reshape(x.shape[:-2]+(self.dim,))  # Na, D
        x = self.attn_act(self.x_proj(x)) + x # Na, D

        vec = v_col.unsqueeze(-3) * dvdij  #  Ne, 3, H, D/H
        vec = scatter(vec, row, dim=0, dim_size=natoms) # Na, 3, H, D/H
        vec = vec.reshape(vec.shape[:-2]+(self.dim,))  # Na, 3, D

        o = self.o_proj(x) # Na, D
        return o, vec

    def forward(self, node_feat, row, col, dv, rij, dvdij, natoms: int):
        q_row_1, q_row_2, k_col_1, k_col_2, v_col = self.qkv_attn(node_feat, row, col) # Ne, H, D/H; Ne, H, D/H; Ne, H, D/H
        attn = (self.attn_act(torch.sum(q_row_1 * k_col_1, dim=-1)) - self.attn_act(torch.sum(q_row_2 * k_col_2, dim=-1))) * rij.unsqueeze(-1)# Ne, H
        dx, dvec = self.xv_proc(v_col, attn, dv, dvdij, row, natoms) # Na, D; Na, 3, D
        return dx, dvec


class LinearAttn(nn.Module):
    """
    Graph Equivariant Transformer Layer
    """
    def __init__(self, dim, num_heads, act_fn=nn.GELU()):
        super(LinearAttn, self).__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.o_proj = nn.Linear(dim, dim * 3)
        self.vec_proj = nn.Linear(dim, dim * 3, bias=False)
        self.x_proj = nn.Linear(dim, dim)

        self.layer_norm = nn.LayerNorm(dim)
        self.dim = dim
        self.dim_per_head = dim // num_heads
        self.num_heads = num_heads
        self.attn_act = act_fn

    def qkv_attn(self, node_feat, row, col):
        node_feat = self.layer_norm(node_feat) # Na, D
        qkv = self.qkv_proj(node_feat) # Na, 3D
        qkv = qkv.reshape(qkv.shape[:-1]+(self.num_heads, self.dim_per_head * 3)) # Na, H, 3D/H

        q1, q2, k1, k2, v = qkv[...,:self.dim_per_head//2], qkv[...,self.dim_per_head//2:self.dim_per_head], qkv[...,self.dim_per_head:(3*self.dim_per_head)//2], qkv[...,(3*self.dim_per_head)//2:2*self.dim_per_head], qkv[...,2*self.dim_per_head:]

        q_row_1 = q1[row] # Ne, H, D/H
        q_row_2 = q2[row] # Ne, H, D/H
        k_col_1 = k1[col] # Ne, H, D/H 
        k_col_2 = k2[col] # Ne, H, D/H
        v_col = v[col] # Ne, H, D/H
        return q_row_1, q_row_2, k_col_1, k_col_2, v_col

    def xv_proc(self, v_col, attn, dv, dvdij, vec, row, natoms: int):
        u = self.vec_proj(vec)  # Na, 3, 3D
        u1, u2, u3 = u[...,:self.dim], u[...,self.dim:2*self.dim], u[...,2*self.dim:] # Na, 3, D; Na, 3, D; Na, 3, D;
        vec = vec.reshape(vec.shape[:-1]+(self.num_heads,self.dim_per_head))  # Na, 3, H, D/H
        u_dot = (u1 * u2).sum(dim=-2)  # Na, D
        
        v = v_col * dv  # Ne, H, D/H
        x = v * attn.unsqueeze(-1)  # Ne, H, D/H
        x = scatter(x, row, dim=0, dim_size=natoms) # Na, H, D/H
        x = x.reshape(x.shape[:-2]+(self.dim,))  # Na, D
        x = self.attn_act(self.x_proj(x)) + x # Na, D

        vec = v_col.unsqueeze(-3) * dvdij  # Ne, 3, H, D/H
        vec = scatter(vec, row, dim=0, dim_size=natoms) # Na, 3, H, D/H
        vec = vec.reshape(vec.shape[:-2]+(self.dim,))  # Na, 3, D

        o = self.o_proj(x)  # Na, 3D
        o1, o2, o3 = o[...,:self.dim], o[...,self.dim:2*self.dim], o[...,2*self.dim:]  # Na, D; Na, D; Na, D
        dx = u_dot * o2 + o3 # Na, D
        dvec = u3 * o1.unsqueeze(-2) + vec # Na, 3, D
        return dx, dvec

    def forward(self, node_feat, row, col, dv, rij, dvdij, vec, natoms: int):
        q_row_1, q_row_2, k_col_1, k_col_2, v_col = self.qkv_attn(node_feat, row, col) # Ne, H, D/H; Ne, H, D/H; Ne, H, D/H
        attn = (self.attn_act(torch.sum(q_row_1 * k_col_1, dim=-1)) -  self.attn_act(torch.sum(q_row_2 * k_col_2, dim=-1))) * rij.unsqueeze(-1)# Ne, H
        dx, dvec = self.xv_proc(v_col, attn, dv, dvdij, vec, row, natoms) # Na, D; Na, 3, D
        return dx, dvec


class LinearAttnLast(nn.Module):
    """
    Graph Equivariant Transformer Last Layer
    No node_vec output compared to middle layers
    """
    def __init__(self, dim, num_heads, act_fn=nn.GELU()):
        super(LinearAttnLast, self).__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.o_proj = nn.Linear(dim, dim * 2)
        self.x_proj = nn.Linear(dim, dim)
        self.vec_proj = nn.Linear(dim, dim * 2, bias=False)

        self.layer_norm = nn.LayerNorm(dim)
        self.dim = dim
        self.dim_per_head = dim // num_heads
        self.num_heads = num_heads
        self.attn_act = act_fn

    def qkv_attn(self, node_feat, row, col):
        node_feat = self.layer_norm(node_feat) # Na, D
        qkv = self.qkv_proj(node_feat) # Na, 3D
        qkv = qkv.reshape(qkv.shape[:-1]+(self.num_heads, self.dim_per_head * 3)) # Na, H, 3D/H

        q1, q2, k1, k2, v = qkv[...,:self.dim_per_head//2], qkv[...,self.dim_per_head//2:self.dim_per_head], qkv[...,self.dim_per_head:(3*self.dim_per_head)//2], qkv[...,(3*self.dim_per_head)//2:2*self.dim_per_head], qkv[...,2*self.dim_per_head:]

        q_row_1 = q1[row] # Ne, H, D/H/2
        q_row_2 = q2[row] # Ne, H, D/H/2
        k_col_1 = k1[col] # Ne, H, D/H/2 
        k_col_2 = k2[col] # Ne, H, D/H/2
        v_col = v[col] # Ne, H, D/H
        return q_row_1, q_row_2, k_col_1, k_col_2, v_col

    def xv_proc(self, v_col, attn, dv, vec, row, natoms: int):
        u = self.vec_proj(vec)  # Na, 3, 2D
        u1, u2 = u[...,:self.dim], u[...,self.dim:]  # Na, 3, D; Na, 3, D
        u_dot = (u1 * u2).sum(dim=-2)  # Na, D

        v = v_col * dv  # Ne, H, D/H
        x = v * attn.unsqueeze(-1)  # Ne, H, D/H
        x = scatter(x, row, dim=0, dim_size=natoms) # Na, H, D/H
        x = x.reshape(x.shape[:-2]+(self.dim,))  # Na, D
        x = self.attn_act(self.x_proj(x)) + x # Na, D

        o = self.o_proj(x)  # Na, 2D
        o2, o3 = o[...,:self.dim], o[...,self.dim:]  # Na, D; Na, D
        dx = u_dot * o2 + o3 # Na, D
        return dx

    def forward(self, node_feat, row, col, dv, rij, vec, natoms: int):
        q_row_1, q_row_2, k_col_1, k_col_2, v_col = self.qkv_attn(node_feat, row, col) # Ne, H, D/H; Ne, H, D/H; Ne, H, D/H
        attn = (self.attn_act(torch.sum(q_row_1 * k_col_1, dim=-1)) -  self.attn_act(torch.sum(q_row_2 * k_col_2, dim=-1))) * rij.unsqueeze(-1)# Ne, H
        dx = self.xv_proc(v_col, attn, dv, vec, row, natoms)  # Na, D;
        return dx


class BambooGEDT(BambooBase):
    def __init__(self, device, dtype, coul_disp_params, nn_params,
                gnn_params = {
                    'n_layers': 3,
                    'num_heads': 8,
                    'act_fn': nn.GELU(),
                }):
        super().__init__(device=device, dtype=dtype, nn_params=nn_params, coul_disp_params=coul_disp_params)
        self.n_layers = gnn_params['n_layers']
        self.num_heads = gnn_params['num_heads']
        self.dim_per_head = self.dim // self.num_heads
        self.act_fn = gnn_params['act_fn']
        self.dv_proj = nn.Sequential(
            nn.Linear(self.num_rbf, self.dim, bias=False),
            self.act_fn
        )  # D
        self.first_attn = LinearAttnFirst(self.dim, self.num_heads, self.act_fn)
        self.attns = nn.ModuleList([
            LinearAttn(self.dim, self.num_heads, self.act_fn) for _ in range(self.n_layers-2)
        ])
        self.last_attn = LinearAttnLast(self.dim, self.num_heads, self.act_fn)
        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def w_proc(self, w):
        dv = self.dv_proj(w) # Ne, D
        dv = dv.reshape(dv.shape[:-1]+(self.num_heads,self.dim_per_head)) # Ne, H, D/H
        return dv

    def graph_nn(self, node_feat, edge_index, coord_diff, radial, weights_rbf):
        dv = self.w_proc(weights_rbf) # Ne, H, D/H
        dvdij = dv.unsqueeze(-3) * coord_diff.unsqueeze(-1).unsqueeze(-1)  # Ne, 3, H, D/H 
        row, col = edge_index[0], edge_index[1]  # Ne
        natoms = node_feat.shape[0]  # 1

        # First block
        dx, dvec = self.first_attn(node_feat, row, col, dv, radial, dvdij, natoms) # Na, D; Na, 3, D
        node_feat = node_feat + dx  # Na, D
        vec = dvec  # Na, 3, D

        # Middle blocks
        for attn in self.attns:
            dx, dvec = attn(node_feat, row, col, dv, radial, dvdij, vec, natoms) # Na, D; Na, 3, D
            node_feat = node_feat + dx  # Na, D
            vec = vec + dvec  # Na, 3, D

        # Last block
        dx = self.last_attn(node_feat, row, col, dv, radial, vec, natoms) # Na, D; Na, 3, D
        node_feat = node_feat + dx  # Na, D
        return node_feat