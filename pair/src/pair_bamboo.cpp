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

#include <pair_bamboo.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "kspace.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <mpi.h>


using namespace LAMMPS_NS;

PairBAMBOO::PairBAMBOO(LAMMPS *lmp) : Pair(lmp) {
    restartinfo = 0;
    manybody_flag = 1;
    evflag = 1;
    msmflag = 1;
    ewaldflag = 1;
    pppmflag = 1;

    int device_count = torch::cuda::device_count();
    if (device_count == 0) {
        error->all(FLERR,"pair_bamboo: no GPUs available");
    }

    int cuda_device_id = -1;
    if (comm->nprocs > 1) {
        if (comm->nprocs > device_count) {
            error->all(FLERR,"pair_bamboo: mismatch between number of ranks and number of available GPUs");
        }
        MPI_Comm shmcomm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
            MPI_INFO_NULL, &shmcomm);
        int shmrank;
        MPI_Comm_rank(shmcomm, &shmrank);
        cuda_device_id = shmrank;
    }

    device = c10::Device(torch::kCUDA, cuda_device_id);

    std::cout << "BAMBOO is using device " << device << "\n";
  
    if(const char* env_p = std::getenv("BAMBOO_DEBUG")){
        if (strcmp(env_p, "1") == 0)
        std::cout << "Found env BAMBOO_DEBUG=1, Pair BAMBOO is in DEBUG mode.\n";
        debug_mode = 1;
    }

    if(const char* env_p = std::getenv("BAMBOO_TIMER")){
        if (strcmp(env_p, "1") == 0)
        std::cout << "Found env BAMBOO_TIMER=1, Pair BAMBOO show timer.\n";
        bamboo_timer = 1;
        t_start = std::chrono::high_resolution_clock::now();
    }
}

PairBAMBOO::~PairBAMBOO(){
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
    }
}

void PairBAMBOO::init_style(){
    // Error check for parameters.
    if (atom->tag_enable == 0){
        error->all(FLERR,"Pair style BAMBOO requires atom IDs");
    }

    if (force->newton_pair == 1) {
        error->all(FLERR,"Pair style BAMBOO requires newton pair off");
    }
    
    // need a full neighbor list
    int irequest = neighbor->request(this,instance_me);
    auto req = neighbor->requests[irequest];
    req->enable_full();

    // Safely access and store the Ewald summation accuracy parameter, if applicable
    if (force->kspace && std::isnormal(force->kspace->g_ewald)) {
        g_ewald = force->kspace->g_ewald;
    } else {
        g_ewald = 0.0; // Default to 0.0 if kspace is not used or g_ewald is not valid
        fmt::print("pair_bamboo: KSAPCE is not set or is not valid. Defaulting to 0.0 for g_ewald.\n");
    }
}

void *PairBAMBOO::extract(const char *str, int &dim)
{
    dim = 0;
    return (void *) &cutoff_coul;
}

double PairBAMBOO::init_one(int i, int j)
{
    return cutoff_max;
}

void PairBAMBOO::allocate()
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create(setflag, n+1, n+1, "pair:setflag");
    memory->create(cutsq, n+1, n+1, "pair:cutsq");
}

void PairBAMBOO::settings(int narg, char **arg) {
    constexpr int MinArgs = 3;
    constexpr int MaxArgs = 4;
    constexpr int DirectEdgeMode = 1;
    constexpr int UndirectedEdgeMode = 0;

    // Validate the number of arguments
    if (narg < MinArgs || narg > MaxArgs) {
        error->all(FLERR, "Illegal pair_style command");
    }

    // Parse cutoff values
    cutoff_net = utils::numeric(FLERR, arg[0], false, lmp);
    cutoff_coul = utils::numeric(FLERR, arg[1], false, lmp);
    cutoff_disp = utils::numeric(FLERR, arg[2], false, lmp);

    // Determine edge mode, defaulting to undirected if not specified
    edge_mode = (narg == MaxArgs) ? utils::inumeric(FLERR, arg[3], false, lmp) : DirectEdgeMode;

    switch (edge_mode) {
        case DirectEdgeMode:
            fmt::print("Using direct edge mode\n");
            break;
        case UndirectedEdgeMode:
            fmt::print("Using undirected edge mode\n");
            break;
        default:
            error->all(FLERR, "Illegal edge mode");
            break;
    }

    // Compute the maximum cutoff
    cutoff_max = std::max({cutoff_net, cutoff_coul, cutoff_disp});

    // Display cutoff values
    fmt::print("cutoff_net: {}, cutoff_coul: {}, cutoff_disp: {}\n",
               cutoff_net, cutoff_coul, cutoff_disp);

    // Compute squared cutoff values for later use
    cutoff_coul_sq = cutoff_coul * cutoff_coul;
    cutoff_disp_sq = cutoff_disp * cutoff_disp;
    cutoff_net_sq = cutoff_net * cutoff_net;
    cutoff_max_sq = cutoff_max * cutoff_max;

}

void PairBAMBOO::coeff(int narg, char **arg) {

    // Allocate memory if not already done
    if (!allocated) allocate();

    const int ntypes = atom->ntypes;

    // Ensure there is exactly one argument for each type plus the model file name
    if (narg != ntypes + 1) {
        error->all(FLERR, "Incorrect args for pair coefficients");
    }

    // Clear previous settings
    for (int i = 1; i <= ntypes; i++) {
        for (int j = i; j <= ntypes; j++) {
            setflag[i][j] = 0;
        }
    }

    std::vector<std::string> elements(ntypes);
    for(int i = 0; i < ntypes; i++){
        elements[i] = arg[i+1];
    }

    // to construct a type mapper from LAMMPS type to Bamboo atom_types
    std::unordered_map<std::string, int> symbol_to_index = {
        {"H", 1}, {"He", 2}, {"LI", 3}, {"Li", 3}, {"Be", 4}, {"B", 5}, {"C", 6}, {"N", 7},
        {"O", 8}, {"F", 9}, {"Ne", 10}, {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14},
        {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20}, {"Sc", 21},
        {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28},
        {"Cu", 29}, {"Zn", 30}, {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35},
        {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40}, {"Nb", 41}, {"Mo", 42},
        {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49},
        {"Sn", 50}, {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54}, {"Cs", 55}, {"Ba", 56},
        {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60}, {"Pm", 61}, {"Sm", 62}, {"Eu", 63},
        {"Gd", 64}, {"Tb", 65}, {"Dy", 66}, {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70},
        {"Lu", 71}, {"Hf", 72}, {"Ta", 73}, {"W", 74}, {"Re", 75}, {"Os", 76}, {"Ir", 77},
        {"Pt", 78}, {"Au", 79}, {"Hg", 80}, {"Tl", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84},
        {"At", 85}, {"Rn", 86}
    };
    std::cout << "Construct type mapper:" << "\n";

    // Initiate type mapper
    for (int i = 0; i< ntypes; i++){
        std::cout << "i: " << i << " symbol: " << elements[i] << " index: " <<  symbol_to_index[elements[i]] << "\n";
        type_mapper.push_back(symbol_to_index[elements[i]]);
    }

    std::cout << "Loading model from " << arg[0] << "\n";

    try {
        model = torch::jit::load(arg[0]);
    } catch (const c10::Error& e) {
        error->all(FLERR, "Failed to load the model");
    }  
    model.eval();
  
    // disable fusion.
    torch::jit::setGraphExecutorOptimize(false);

    torch::set_default_dtype(caffe2::TypeMeta::Make<double>());

    // Set whether to allow TF32
    bool allow_tf32 = false;
    at::globalContext().setAllowTF32CuBLAS(allow_tf32);
    at::globalContext().setAllowTF32CuDNN(allow_tf32);

    // set setflag i,j for type pairs where both are mapped to elements
    for (int i = 1; i <= ntypes; i++) {
        for (int j = i; j <= ntypes; j++) {
            if ((type_mapper[i-1] >= 0) && (type_mapper[j-1] >= 0)) {
                setflag[i][j] = 1;
            }
        }
    }

}


void PairBAMBOO::click_timer(const char *str){
// Return immediately if the bamboo_timer is not enabled
    if (!bamboo_timer) {
        return;
    }

    // Capture the current time as the end point
    t_end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds since the last checkpoint
    auto durationInMillis = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;

    // Log the timer tag, counter, and the duration
    fmt::print("[tag: {}], [steps: {}], [Time used: {} ms]\n", str, timer_counter, durationInMillis);

    // Reset the start time for the next duration measurement
    t_start = t_end;
}


void PairBAMBOO::compute(int eflag, int vflag){
    // Raise error, this method should not be called.
    error->all(FLERR, "Pair style BAMBOO NON-KOKKOS mode is not supported.");
}
