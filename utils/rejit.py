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

import torch
import torch.nn as nn

from models.bamboo_get import BambooGET


def convert(checkpoint, device=torch.device('cuda')):
    if isinstance(checkpoint, str):
        model = torch.jit.load(checkpoint)
    elif isinstance(checkpoint, torch.jit.RecursiveScriptModule):
        model = checkpoint
    else:
        raise ValueError("Input must be a string or torch.jit.RecursiveScriptModule.")

    act_fn_map = {
        'ELU': nn.ELU(),
        'CELU': nn.CELU(),
        'GELU': nn.GELU(),
        'SiLU': nn.SiLU(),
        'Mish': nn.Mish(),
        'Softplus': nn.Softplus()
    }
    
    torch_script_dtype_mapper = {
        1: torch.int8,
        2: torch.int16,
        3: torch.int32,
        4: torch.int64,
        5: torch.float16,
        6: torch.float32,
        7: torch.float64,
    }

    nn_params_act_fn_name: str = list(model.charge_mlp.children())[1].original_name
    gnn_params_act_fn_name: str = model.act_fn.original_name
    nn_params = {
        'dim': model.dim,
        'num_rbf': model.num_rbf,
        'rcut': model.rcut,
        'charge_ub': model.charge_ub,
        'act_fn': act_fn_map[nn_params_act_fn_name],
        'charge_mlp_layers': model.charge_mlp_layers,
        'energy_mlp_layers': model.energy_mlp_layers,
    }

    gnn_params = {
        'n_layers': model.n_layers,
        'num_heads': model.num_heads,
        'act_fn': act_fn_map[gnn_params_act_fn_name]
    }

    origin_model = BambooGET(
        device=model.device, 
        coul_disp_params=model.coul_disp_params,
        nn_params=nn_params,
        gnn_params=gnn_params
    )
    
    origin_model.load_state_dict(model.state_dict())
    origin_model = origin_model.to(device)
    return origin_model


def main():
    arparser = argparse.ArgumentParser()
    arparser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    arparser.add_argument('--destination', type=str, required=True, help='Path to save the converted model.')
    arparser.add_argument('--no_cuda', action='store_true', help='Do not use GPU for training.') 
    args = arparser.parse_args()

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'model loaded from {args.checkpoint}')

    model = convert(args.checkpoint, device)

    model_jit = torch.jit.script(model)
    model_jit.save(args.destination)
    print(f'model saved at {args.destination}')
    

if __name__ == "__main__":
    main()
