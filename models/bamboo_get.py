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
    def __init__(self, 
                dim = 64, 
                num_heads = 16, 
                act_fn = nn.GELU()):
        super(LinearAttnFirst, self).__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.output_proj = nn.Linear(dim, dim)

        self.layer_norm = nn.LayerNorm(dim)
        self.dim = dim
        self.dim_per_head = dim // num_heads
        self.num_heads = num_heads
        self.attn_act = act_fn

    def qkv_attn(self, 
                node_feat: torch.Tensor, 
                row: torch.Tensor, 
                col: torch.Tensor,
        ) -> List[torch.Tensor]:
        node_feat = self.layer_norm(node_feat)
        qkv = self.qkv_proj(node_feat)
        qkv = qkv.reshape(qkv.shape[:-1]+(self.num_heads, self.dim_per_head * 3)) 

        q, k, v = qkv[...,:self.dim_per_head], qkv[...,self.dim_per_head:2*self.dim_per_head], qkv[...,2*self.dim_per_head:]
        q_row, k_col, v_col = q[row], k[col], v[col]
        return q_row, k_col, v_col

    def forward(self, 
                node_feat: torch.Tensor,
                edge_feat: torch.Tensor, 
                edge_vec: torch.Tensor,
                row: torch.Tensor, 
                col: torch.Tensor, 
                radial: torch.Tensor, 
                natoms: int,
        ) -> List[torch.Tensor]:
        # attention layer
        q_row, k_col, v_col = self.qkv_attn(node_feat, row, col)
        attn = self.attn_act(torch.sum(q_row * k_col, dim=-1)) * radial.unsqueeze(-1)

        # update scalar messages
        m_feat = v_col * edge_feat * attn.unsqueeze(-1)
        m_feat = scatter(m_feat, row, dim=0, dim_size=natoms)
        m_feat = m_feat.reshape(m_feat.shape[:-2]+(self.dim,)) 

        # update vector messages
        m_vec = v_col.unsqueeze(-3) * edge_vec
        m_vec = scatter(m_vec, row, dim=0, dim_size=natoms) 
        delta_node_vec = m_vec.reshape(m_vec.shape[:-2]+(self.dim,))

        # update scalar node features
        delta_node_feat = self.output_proj(m_feat)
        return delta_node_feat, delta_node_vec


class LinearAttn(nn.Module):
    """
    Graph Equivariant Transformer Layer
    """
    def __init__(self, 
                dim = 64, 
                num_heads = 16, 
                act_fn = nn.GELU()):
        super(LinearAttn, self).__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.output_proj = nn.Linear(dim, dim * 3)
        self.vec_proj = nn.Linear(dim, dim * 3, bias=False)

        self.layer_norm = nn.LayerNorm(dim)
        self.dim = dim
        self.dim_per_head = dim // num_heads
        self.num_heads = num_heads
        self.attn_act = act_fn

    def qkv_attn(self, 
                node_feat: torch.Tensor, 
                row: torch.Tensor, 
                col: torch.Tensor,
        ) -> List[torch.Tensor]:
        node_feat = self.layer_norm(node_feat)
        qkv = self.qkv_proj(node_feat)
        qkv = qkv.reshape(qkv.shape[:-1]+(self.num_heads, self.dim_per_head * 3)) 

        q, k, v = qkv[...,:self.dim_per_head], qkv[...,self.dim_per_head:2*self.dim_per_head], qkv[...,2*self.dim_per_head:]
        q_row, k_col, v_col = q[row], k[col], v[col]
        return q_row, k_col, v_col

    def forward(self, 
                node_feat: torch.Tensor,
                edge_feat: torch.Tensor, 
                node_vec: torch.Tensor,
                edge_vec: torch.Tensor,
                row: torch.Tensor, 
                col: torch.Tensor, 
                radial: torch.Tensor, 
                natoms: int,
        ) -> List[torch.Tensor]:
        # attention layer
        q_row, k_col, v_col = self.qkv_attn(node_feat, row, col)
        attn = self.attn_act(torch.sum(q_row * k_col, dim=-1)) * radial.unsqueeze(-1)

        # preprocess node vectors; user inner product to produce scalar feature to ensure equivariance
        input_vec = self.vec_proj(node_vec)
        input_1, input_2, input_3 = input_vec[...,:self.dim], input_vec[...,self.dim:2*self.dim], input_vec[...,2*self.dim:] 
        input_dot = (input_1 * input_2).sum(dim=-2)  

        # update scalar messages
        m_feat = v_col * edge_feat * attn.unsqueeze(-1)
        m_feat = scatter(m_feat, row, dim=0, dim_size=natoms)
        m_feat = m_feat.reshape(m_feat.shape[:-2]+(self.dim,)) 

        # update vector messages
        m_vec = v_col.unsqueeze(-3) * edge_vec
        m_vec = scatter(m_vec, row, dim=0, dim_size=natoms) 
        m_vec = m_vec.reshape(m_vec.shape[:-2]+(self.dim,))

        # update scalar node features
        output_feat = self.output_proj(m_feat)
        output_1, output_2, output_3 = output_feat[...,:self.dim], output_feat[...,self.dim:2*self.dim], output_feat[...,2*self.dim:]
        delta_node_feat = input_dot * output_2 + output_3

        # update node vectors
        delta_node_vec = input_3 * output_1.unsqueeze(-2) + m_vec

        return delta_node_feat, delta_node_vec


class LinearAttnLast(nn.Module):
    """
    Graph Equivariant Transformer Last Layer
    No node_vec output compared to middle layers
    """
    def __init__(self, 
                dim = 64, 
                num_heads = 16, 
                act_fn = nn.GELU()):
        super(LinearAttnLast, self).__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.output_proj = nn.Linear(dim, dim * 2)
        self.vec_proj = nn.Linear(dim, dim * 2, bias=False)

        self.layer_norm = nn.LayerNorm(dim)
        self.dim = dim
        self.dim_per_head = dim // num_heads
        self.num_heads = num_heads
        self.attn_act = act_fn

    def qkv_attn(self, 
                node_feat: torch.Tensor, 
                row: torch.Tensor, 
                col: torch.Tensor,
        ) -> List[torch.Tensor]:
        node_feat = self.layer_norm(node_feat)
        qkv = self.qkv_proj(node_feat)
        qkv = qkv.reshape(qkv.shape[:-1]+(self.num_heads, self.dim_per_head * 3)) 

        q, k, v = qkv[...,:self.dim_per_head], qkv[...,self.dim_per_head:2*self.dim_per_head], qkv[...,2*self.dim_per_head:]
        q_row, k_col, v_col = q[row], k[col], v[col]
        return q_row, k_col, v_col

    def forward(self, 
                node_feat: torch.Tensor,
                edge_feat: torch.Tensor, 
                node_vec: torch.Tensor,
                row: torch.Tensor, 
                col: torch.Tensor, 
                radial: torch.Tensor, 
                natoms: int,
        ) -> torch.Tensor:
        # attention layer
        q_row, k_col, v_col = self.qkv_attn(node_feat, row, col)
        attn = self.attn_act(torch.sum(q_row * k_col, dim=-1)) * radial.unsqueeze(-1)

        # preprocess node vectors; user inner product to produce scalar feature to ensure equivariance
        input_vec = self.vec_proj(node_vec)
        input_1, input_2= input_vec[...,:self.dim], input_vec[...,self.dim:]
        input_dot = (input_1 * input_2).sum(dim=-2)  

        # update scalar messages
        m_feat = v_col * edge_feat * attn.unsqueeze(-1)
        m_feat = scatter(m_feat, row, dim=0, dim_size=natoms)
        m_feat = m_feat.reshape(m_feat.shape[:-2]+(self.dim,)) 

        # update scalar node features
        output_feat = self.output_proj(m_feat)
        output_2, output_3 = output_feat[...,:self.dim], output_feat[...,self.dim:]
        delta_node_feat = input_dot * output_2 + output_3

        return delta_node_feat


class BambooGET(BambooBase):
    def __init__(self, device, coul_disp_params, nn_params,
                gnn_params = {
                    'n_layers': 3,
                    'num_heads': 16,
                    'act_fn': nn.GELU(),
                }):
        super(BambooGET, self).__init__(device=device, nn_params=nn_params, coul_disp_params=coul_disp_params)
        self.n_layers = gnn_params['n_layers']
        self.num_heads = gnn_params['num_heads']
        self.dim_per_head = self.dim // self.num_heads
        self.act_fn = gnn_params['act_fn']
        self.rbf_proj = nn.Sequential(
            nn.Linear(self.num_rbf, self.dim, bias=False),
            self.act_fn
        )  
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

    def graph_nn(self, 
                node_feat: torch.Tensor, 
                edge_index: torch.Tensor, 
                coord_diff: torch.Tensor, 
                radial: torch.Tensor, 
                weights_rbf: torch.Tensor,
        ) -> torch.Tensor:
        # compute initial edge feature and edge vector
        edge_feat = self.rbf_proj(weights_rbf)
        edge_feat = edge_feat.reshape(edge_feat.shape[:-1]+(self.num_heads, self.dim_per_head))
        edge_vec = edge_feat.unsqueeze(-3) * coord_diff.unsqueeze(-1).unsqueeze(-1) 
        row, col = edge_index[0], edge_index[1] 
        natoms = node_feat.shape[0] 

        # first GET layer
        delta_node_feat, delta_node_vec = self.first_attn(node_feat, edge_feat, edge_vec, row, col, radial, natoms)
        node_feat = node_feat + delta_node_feat
        node_vec = delta_node_vec

        # middle GET layerss
        for attn in self.attns:
            delta_node_feat, delta_node_vec = attn(node_feat, edge_feat, node_vec, edge_vec, row, col, radial, natoms)
            node_feat = node_feat + delta_node_feat
            node_vec = node_vec + delta_node_vec

        # last GEt layer
        delta_node_feat = self.last_attn(node_feat, edge_feat, node_vec, row, col, radial, natoms) 
        node_feat = node_feat + delta_node_feat 
        return node_feat