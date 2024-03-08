/* -------------------------------------------------------------------------
-----  BAMBOO: Bytedance AI Molecular Booster -----
Copyright 2022-2024 Bytedance Ltd. and/or its affiliates 

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(bamboo,PairBAMBOO)

#else

#ifndef LMP_PAIR_BAMBOO_H
#define LMP_PAIR_BAMBOO_H

#include "pair.h"
#include <time.h>
#include <torch/torch.h>

namespace LAMMPS_NS {

class PairBAMBOO : public Pair {
public:
    PairBAMBOO(class LAMMPS *);
    virtual ~PairBAMBOO();
    virtual void compute(int, int);
    void settings(int, char **);
    virtual void coeff(int, char **);
    virtual double init_one(int, int);
    virtual void init_style();
    void allocate();
    void *extract(const char *str, int &dim);
    void click_timer(const char *str);
    double cutoff_disp, cutoff_coul, cutoff_net, cutoff_max;
    double cutoff_disp_sq, cutoff_coul_sq, cutoff_net_sq, cutoff_max_sq;
    torch::jit::Module model;
    torch::Device device = torch::kCPU;

protected:
    std::vector<int> type_mapper;
    int debug_mode = 0;
    int bamboo_timer = 0;
    int timer_counter = 0;
    int edge_mode = 0;  // 0: undirected edge, 1 directed edge.
    double g_ewald;
    std::chrono::high_resolution_clock::time_point t_start, t_end; // Start time for the timer
};

}
#endif
#endif

