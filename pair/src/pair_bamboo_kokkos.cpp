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



#include <pair_bamboo_kokkos.h>
#include <torch/torch.h>
#include <torch/script.h>


using namespace LAMMPS_NS;

#ifdef LMP_KOKKOS_GPU
  int vector_length = 32;
#define TEAM_SIZE 4
#define SINGLE_BOND_TEAM_SIZE 16
#else
  int vector_length = 8;
#define TEAM_SIZE Kokkos::AUTO()
#define SINGLE_BOND_TEAM_SIZE Kokkos::AUTO()
#endif

// Buffer for edge size memory.
#define kEdgeRatio 1.1f


template<class DeviceType>
PairBAMBOOKokkos<DeviceType>::PairBAMBOOKokkos(LAMMPS *lmp) : PairBAMBOO(lmp)
{
    respa_enable = 0;

    atomKK = (AtomKokkos *) atom;
    domainKK = (DomainKokkos *) domain;
    execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
    datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
    datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}


template<class DeviceType>
PairBAMBOOKokkos<DeviceType>::~PairBAMBOOKokkos()
{
    if (!copymode) {
        memoryKK->destroy_kokkos(k_eatom,eatom);
        memoryKK->destroy_kokkos(k_vatom,vatom);
        eatom = NULL;
        vatom = NULL;
    }
}

template<class DeviceType>
void PairBAMBOOKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
    eflag = eflag_in;
    vflag = vflag_in;
    ev_init(eflag,vflag, 0);
    click_timer("Compute start");

    // reallocate per-atom arrays if necessary

    if (eflag_atom) {
        memoryKK->destroy_kokkos(k_eatom,eatom);
        memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
        d_eatom = k_eatom.view<DeviceType>();
    }
    if (vflag_atom) {
        memoryKK->destroy_kokkos(k_vatom,vatom);
        memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
        d_vatom = k_vatom.view<DeviceType>();
    }

    atomKK->sync(execution_space,datamask_read);
    if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
    else atomKK->modified(execution_space,F_MASK);

    x = atomKK->k_x.view<DeviceType>();
    f = atomKK->k_f.view<DeviceType>();
    q = atomKK->k_q.view<DeviceType>();
    tag = atomKK->k_tag.view<DeviceType>();
    type = atomKK->k_type.view<DeviceType>();
    nlocal = atom->nlocal;

    nall = atom->nlocal + atom->nghost;
    const int inum = list->inum;

    NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
    d_ilist = k_list->d_ilist;
    d_numneigh = k_list->d_numneigh;
    d_neighbors = k_list->d_neighbors;

    copymode = 1;

    // build short neighbor list
    const int max_neighs = d_neighbors.extent(1);
    // Some extra buffer to decrease resize frequence.
    const int max_neighs_memory = static_cast<int>(kEdgeRatio * d_neighbors.extent(1));

    if(d_numneigh_coul.extent(0) < inum){
        // edges for Coulomb
        reallocateView(d_numneigh_coul, "BAMBOO::numneighs_coul", inum);
        reallocateView(d_cumsum_numneigh_coul, "BAMBOO::cumsum_numneighs_coul", inum);

        // edges for dispersion
        reallocateView(d_numneigh_disp, "BAMBOO::numneighs_disp", inum);
        reallocateView(d_cumsum_numneigh_disp, "BAMBOO::cumsum_numneighs_disp", inum);

        // edges for neural network
        reallocateView(d_numneigh_net, "BAMBOO::numneighs_net", inum);
        reallocateView(d_cumsum_numneigh_net, "BAMBOO::cumsum_numneighs_net", inum);
    }

    if(d_neighbors_coul.extent(0) < inum || d_neighbors_coul.extent(1) < max_neighs_memory){
        reallocateView(d_neighbors_coul, "BAMBOO::neighbors_coul", inum, max_neighs_memory);
        reallocateView(d_neighbors_disp, "BAMBOO::neighbors_disp", inum, max_neighs_memory);
        reallocateView(d_neighbors_net, "BAMBOO::neighbors_net", inum, max_neighs_memory);
        reallocateView(d_edge_shift, "BAMBOO::edge_shift", inum, max_neighs_memory, 3);
    }

    click_timer("Init views");

    // compute short neighbor list
    auto d_numneigh_coul = this->d_numneigh_coul;
    auto d_neighbors_coul = this->d_neighbors_coul;
    auto d_cumsum_numneigh_coul = this->d_cumsum_numneigh_coul;
    auto d_numneigh_disp = this->d_numneigh_disp;
    auto d_neighbors_disp = this->d_neighbors_disp;
    auto d_cumsum_numneigh_disp = this->d_cumsum_numneigh_disp;
    auto d_numneigh_net = this->d_numneigh_net;
    auto d_neighbors_net = this->d_neighbors_net;
    auto d_cumsum_numneigh_net = this->d_cumsum_numneigh_net;

    auto d_edge_shift = this->d_edge_shift;

    double cutoff_max_sq = this->cutoff_max_sq;
    double cutoff_coul_sq = this->cutoff_coul_sq;
    double cutoff_disp_sq = this->cutoff_disp_sq;
    double cutoff_net_sq = this->cutoff_net_sq;
    auto x = this->x;
    auto d_type = this->type;
    auto d_ilist = this->d_ilist;
    auto d_numneigh = this->d_numneigh;
    auto d_neighbors = this->d_neighbors;
    auto f = this->f;
    auto d_eatom = this->d_eatom;
    auto d_type_mapper = this->d_type_mapper;
    auto tag = this->tag;
    auto q = this->q;
    auto edge_mode = this->edge_mode;

    click_timer("Pre-Loop NeignborList");
    Kokkos::parallel_for("BAMBOO: Loop NeignborList", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii){
        const int i = d_ilist(ii);
        const X_FLOAT xtmp = x(i,0);
        const X_FLOAT ytmp = x(i,1);
        const X_FLOAT ztmp = x(i,2);

        const int jnum = d_numneigh(i);
        int coul_index_count = 0;
        int disp_index_count = 0;
        int net_index_count = 0;
        for (int jj = 0; jj < jnum; jj++) {
            int j = d_neighbors(i,jj);
            j &= NEIGHMASK;

            const X_FLOAT delx = xtmp - x(j,0);
            const X_FLOAT dely = ytmp - x(j,1);
            const X_FLOAT delz = ztmp - x(j,2);
            const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
            // default: disp is the max cutoff
            if (rsq < cutoff_max_sq && (edge_mode || tag(i) < tag(j))) {
            if (rsq < cutoff_disp_sq){
                d_neighbors_disp(ii, disp_index_count) = jj;
                disp_index_count++;
            }
            if (rsq < cutoff_coul_sq){
                d_neighbors_coul(ii, coul_index_count) = jj;
                coul_index_count++;
            }
            if (rsq < cutoff_net_sq){
                d_neighbors_net(ii, net_index_count) = jj;
                net_index_count++;
            }
            d_edge_shift(ii, jj, 0) = delx;
            d_edge_shift(ii, jj, 1) = dely;
            d_edge_shift(ii, jj, 2) = delz;
            }
        }
        d_numneigh_net(ii) = net_index_count;
        d_numneigh_coul(ii) = coul_index_count;
        d_numneigh_disp(ii) = disp_index_count;
    });

    click_timer("Loop NeignborList");

    if(debug_mode){
        std::cout << "index: " << "\n";
        std::cout << "d_cumsum_numneigh_coul: " <<  d_cumsum_numneigh_coul.extent(0) << "\n";
        std::cout << "d_numneigh_coul: " <<  d_numneigh_coul.extent(0) << "\n";
        std::cout << "d_cumsum_numneigh_disp: " <<  d_cumsum_numneigh_disp.extent(0) << "\n";
        std::cout << "d_numneigh_disp: " <<  d_numneigh_disp.extent(0) << "\n";
        std::cout << "d_cumsum_numneigh_net: " <<  d_cumsum_numneigh_net.extent(0) << "\n";
        std::cout << "d_numneigh_net: " <<  d_numneigh_net.extent(0) << "\n";
    }
  
    Kokkos::deep_copy(d_cumsum_numneigh_coul, d_numneigh_coul);
    Kokkos::deep_copy(d_cumsum_numneigh_disp, d_numneigh_disp);
    Kokkos::deep_copy(d_cumsum_numneigh_net, d_numneigh_net);

    Kokkos::parallel_scan("BAMBOO: cumsum coul_neighs", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii, int& update, const bool is_final){
        const int curr_val = d_cumsum_numneigh_coul(ii);
        update += curr_val;
        if(is_final) d_cumsum_numneigh_coul(ii) = update;
    });
    Kokkos::parallel_scan("BAMBOO: cumsum disp_neighs", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii, int& update, const bool is_final){
        const int curr_val = d_cumsum_numneigh_disp(ii);
        update += curr_val;
        if(is_final) d_cumsum_numneigh_disp(ii) = update;
    });
    Kokkos::parallel_scan("BAMBOO: cumsum net_neighs", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii, int& update, const bool is_final){
        const int curr_val = d_cumsum_numneigh_net(ii);
        update += curr_val;
        if(is_final) d_cumsum_numneigh_net(ii) = update;
    });

    click_timer("Scan NeignborList");

    int n_coul_edges = 0;
    int n_disp_edges = 0;
    int n_net_edges = 0;
    Kokkos::View<int*, Kokkos::HostSpace> n_coul_edges_view("BAMBOO: n_coul_edges_view",1);
    Kokkos::View<int*, Kokkos::HostSpace> n_disp_edges_view("BAMBOO: n_disp_edges_view",1);
    Kokkos::View<int*, Kokkos::HostSpace> n_net_edges_view("BAMBOO: n_net_edges_view",1);
    Kokkos::deep_copy(n_coul_edges_view, Kokkos::subview(d_cumsum_numneigh_coul, Kokkos::make_pair(inum-1, inum)));
    Kokkos::deep_copy(n_disp_edges_view, Kokkos::subview(d_cumsum_numneigh_disp, Kokkos::make_pair(inum-1, inum)));
    Kokkos::deep_copy(n_net_edges_view, Kokkos::subview(d_cumsum_numneigh_net, Kokkos::make_pair(inum-1, inum)));
    n_coul_edges = n_coul_edges_view(0);
    n_disp_edges = n_disp_edges_view(0);
    n_net_edges  = n_net_edges_view(0);
    
    click_timer("Edge cumsum");
  
    if(d_coul_edges.extent(1) < n_coul_edges){
        reallocateView(d_coul_edges, "BAMBOO: coul_edges", 2, n_coul_edges);
        reallocateView(d_edge_shift_coul, "BAMBOO: coul_edge_shift", n_coul_edges, 3);
    }
    if(d_disp_edges.extent(1) < n_disp_edges){
        reallocateView(d_disp_edges, "BAMBOO: disp_edges", 2, n_disp_edges);
        reallocateView(d_edge_shift_disp, "BAMBOO: disp_edge_shift", n_disp_edges, 3);

    }
    if(d_net_edges.extent(1) < n_net_edges){
        reallocateView(d_net_edges, "BAMBOO: net_edges", 2, n_net_edges);
        reallocateView(d_edge_shift_net, "BAMBOO: net_edge_shift", n_net_edges, 3);
    }
    if(d_atom_types.extent(0) < inum){
        reallocateView(d_atom_types, "BAMBOO: atom_types", inum);
        reallocateView(d_atom_pos, "BAMBOO: atom_pos", inum, 3);
    }
    
    click_timer("Pre-Loop edge");

    auto d_coul_edges = this->d_coul_edges;
    auto d_net_edges  = this->d_net_edges;
    auto d_disp_edges = this->d_disp_edges;
    auto d_atom_types    = this->d_atom_types;
    auto d_atom_pos     = this->d_atom_pos;
    auto d_edge_shift_coul = this->d_edge_shift_coul;
    auto d_edge_shift_disp = this->d_edge_shift_disp;
    auto d_edge_shift_net  = this->d_edge_shift_net;

    Kokkos::parallel_for("BAMBOO: atom type and pos", Kokkos::RangePolicy<DeviceType>(0, inum), KOKKOS_LAMBDA(const int i){
        const int itag = tag(i) - 1;
        // 1 based to 0 based index.
        d_atom_types(itag) = d_type_mapper(d_type(i) - 1);
        d_atom_pos(itag,0) = x(i,0);
        d_atom_pos(itag,1) = x(i,1);
        d_atom_pos(itag,2) = x(i,2);
    });
    click_timer("Position");

    Kokkos::parallel_for("BAMBOO: create coul edges", Kokkos::TeamPolicy<DeviceType>(inum, Kokkos::AUTO()), KOKKOS_LAMBDA(const MemberType team_member){
        const int ii = team_member.league_rank();
        const int i = d_ilist(ii);
        const int startedge = ii==0 ? 0 : d_cumsum_numneigh_coul(ii-1);
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, d_numneigh_coul(ii)), [&] (const int jj){
            const int jj_origin = d_neighbors_coul(ii,jj);
            const int j = d_neighbors(i, jj_origin);
            d_coul_edges(0, startedge + jj) = tag(i) - 1;
            d_coul_edges(1, startedge + jj) = tag(j) - 1;
            d_edge_shift_coul(startedge + jj, 0) = d_edge_shift(ii, jj_origin, 0);
            d_edge_shift_coul(startedge + jj, 1) = d_edge_shift(ii, jj_origin, 1);
            d_edge_shift_coul(startedge + jj, 2) = d_edge_shift(ii, jj_origin, 2);
        });
    });

    click_timer("Edge coul");

    Kokkos::parallel_for("BAMBOO: create disp edges", Kokkos::TeamPolicy<DeviceType>(inum, Kokkos::AUTO()), KOKKOS_LAMBDA(const MemberType team_member){
        const int ii = team_member.league_rank();
        const int i = d_ilist(ii);
        const int startedge = ii==0 ? 0 : d_cumsum_numneigh_disp(ii-1);
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, d_numneigh_disp(ii)), [&] (const int jj){
            const int jj_origin = d_neighbors_disp(ii,jj);
            const int j = d_neighbors(i, jj_origin);
            d_disp_edges(0, startedge + jj) = tag(i) - 1;
            d_disp_edges(1, startedge + jj) = tag(j) - 1;
            d_edge_shift_disp(startedge + jj, 0) = d_edge_shift(ii, jj_origin, 0);
            d_edge_shift_disp(startedge + jj, 1) = d_edge_shift(ii, jj_origin, 1);
            d_edge_shift_disp(startedge + jj, 2) = d_edge_shift(ii, jj_origin, 2);
        });
    });

    click_timer("Edge disp");

    Kokkos::parallel_for("BAMBOO: create net edges", Kokkos::TeamPolicy<DeviceType>(inum, Kokkos::AUTO()), KOKKOS_LAMBDA(const MemberType team_member){
        const int ii = team_member.league_rank();
        const int i = d_ilist(ii);
        const int startedge = ii==0 ? 0 : d_cumsum_numneigh_net(ii-1);
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, d_numneigh_net(ii)), [&] (const int jj){
            const int jj_origin = d_neighbors_net(ii,jj);
            const int j = d_neighbors(i, jj_origin);
            d_net_edges(0, startedge + jj) = tag(i) - 1;
            d_net_edges(1, startedge + jj) = tag(j) - 1;
            d_edge_shift_net(startedge + jj, 0) = d_edge_shift(ii, jj_origin, 0);
            d_edge_shift_net(startedge + jj, 1) = d_edge_shift(ii, jj_origin, 1);
            d_edge_shift_net(startedge + jj, 2) = d_edge_shift(ii, jj_origin, 2);
        });
    });

    click_timer("Edge net");

    DoubleView2D d_domain_box("domain_box", 3, 3);
    auto h_domain_box = Kokkos::create_mirror_view(d_domain_box);
    h_domain_box(0, 0) = domain->boxhi[0] - domain->boxlo[0];

    h_domain_box(1, 0) = domain->xy;
    h_domain_box(1, 1) = domain->boxhi[1] - domain->boxlo[1];

    h_domain_box(2, 0) = domain->xz;
    h_domain_box(2, 1) = domain->yz;
    h_domain_box(2, 2) = domain->boxhi[2] - domain->boxlo[2];
    Kokkos::deep_copy(d_domain_box, h_domain_box);
    click_timer("Domain box");

    torch::Tensor edge_shift_tensor = torch::from_blob(d_edge_shift.data(),  {inum, max_neighs, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    torch::Tensor neighbors_net_tensor = torch::from_blob(d_neighbors_net.data(),   {inum, max_neighs}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    torch::Tensor neighbors_coul_tensor = torch::from_blob(d_neighbors_coul.data(), {inum, max_neighs}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    torch::Tensor neighbors_disp_tensor = torch::from_blob(d_neighbors_disp.data(), {inum, max_neighs}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    torch::Tensor coul_edge_index = torch::from_blob(d_coul_edges.data(), {2,n_coul_edges}, {(long) d_coul_edges.extent(1),1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    torch::Tensor net_edges_index = torch::from_blob(d_net_edges.data(),  {2,n_net_edges }, {(long) d_net_edges.extent(1), 1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    torch::Tensor disp_edge_index = torch::from_blob(d_disp_edges.data(), {2,n_disp_edges}, {(long) d_disp_edges.extent(1),1}, torch::TensorOptions().dtype(torch::kInt64).device(device));

    torch::Tensor coul_edge_cell_shift = torch::from_blob(d_edge_shift_coul.data(), {n_coul_edges, 3}, {3 ,1}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    torch::Tensor net_edge_cell_shift  = torch::from_blob(d_edge_shift_net.data(),  {n_net_edges, 3},  {3 ,1}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    torch::Tensor disp_edge_cell_shift = torch::from_blob(d_edge_shift_disp.data(), {n_disp_edges, 3}, {3 ,1}, torch::TensorOptions().dtype(torch::kFloat64).device(device)); 

    torch::Tensor ij2type_tensor = torch::from_blob(d_atom_types.data(), {inum}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    torch::Tensor pos_tensor = torch::from_blob(d_atom_pos.data(), {inum,3}, {3,1}, torch::TensorOptions().device(device));
    torch::Tensor cell_tensor = torch::from_blob(d_domain_box.data(), {3,3}, {3, 1}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    torch::Tensor ewald_tensor = torch::tensor({g_ewald}, torch::TensorOptions().dtype(torch::kFloat64).device(device));

    if(debug_mode){
        std::cout << "coul_edge_index: " << coul_edge_index.sizes() << "\n";
        std::cout << "net_edges_index: " << net_edges_index.sizes() << "\n";
        std::cout << "disp_edge_index: " << disp_edge_index.sizes() << "\n";
        std::cout << "coul_edge_cell_shift: " << coul_edge_cell_shift.sizes() << "\n";
        std::cout << "net_edge_cell_shift: " << net_edge_cell_shift.sizes() << "\n";
        std::cout << "disp_edge_cell_shift: " << disp_edge_cell_shift.sizes() << "\n";

        torch::save({
            coul_edge_index,
            net_edges_index,
            disp_edge_index,
            coul_edge_cell_shift,
            net_edge_cell_shift,
            disp_edge_cell_shift,
            ij2type_tensor, pos_tensor,
            cell_tensor,
            ewald_tensor,
            edge_shift_tensor,
            neighbors_net_tensor,
            neighbors_coul_tensor,
            neighbors_disp_tensor
        }, "tensor.pt");

        std::cout << "Save tensor to tensor.pt" << "\n";
    }


    c10::Dict<std::string, torch::Tensor> input;
    input.insert("pos", pos_tensor);
    input.insert("edge_index", net_edges_index);
    input.insert("edge_cell_shift", net_edge_cell_shift);
    input.insert("coul_edge_index", coul_edge_index);
    input.insert("coul_edge_cell_shift", coul_edge_cell_shift);
    input.insert("disp_edge_index", disp_edge_index);
    input.insert("disp_edge_cell_shift", disp_edge_cell_shift);
    input.insert("cell", cell_tensor);
    input.insert("atom_types", ij2type_tensor);
    input.insert("g_ewald", ewald_tensor);
    std::vector<torch::IValue> input_vector(1, input);
    click_timer("Pre-inference");

    auto output = model.forward(input_vector).toGenericDict();
    torch::Tensor pred_virial_tensor    = output.at("pred_virial").toTensor().cpu();
    click_timer("Inference");

    torch::Tensor pred_coul_energy_tensor = output.at("pred_coul_energy").toTensor().cpu();
    torch::Tensor pred_energy_tensor   = output.at("pred_energy").toTensor().cpu();
    torch::Tensor predict_charge_tensor = output.at("pred_charge").toTensor();
    torch::Tensor predict_forces_tensor = output.at("pred_forces").toTensor();

    UnmanagedDoubleView2D d_forces(predict_forces_tensor.data_ptr<double>(), inum, 3);
    UnmanagedDoubleView1D d_charge(predict_charge_tensor.data_ptr<double>(), inum);

    click_timer("Begin postprocess.");
    eng_vdwl = pred_energy_tensor.data_ptr<double>()[0] - pred_coul_energy_tensor.data_ptr<double>()[0];
    eng_coul = pred_coul_energy_tensor.data_ptr<double>()[0];

    click_timer("Update fpair.");
    Kokkos::parallel_for("BAMBOO: update fpair", Kokkos::RangePolicy<DeviceType>(0, inum), KOKKOS_LAMBDA(const int i){
        // 1 based to 0 based index.
        const int itag = tag(i) - 1;
        f(i,0) = d_forces(itag, 0);
        f(i,1) = d_forces(itag, 1);
        f(i,2) = d_forces(itag, 2);
        q(i)   = d_charge(itag);
    });

    click_timer("Update nn_virial.");
    auto predict_virial = pred_virial_tensor.accessor<double, 2>();
    virial[0] += predict_virial[0][0];
    virial[1] += predict_virial[1][1];
    virial[2] += predict_virial[2][2];
    virial[3] += 0.5 * (predict_virial[0][1] + predict_virial[1][0]);
    virial[4] += 0.5 * (predict_virial[0][2] + predict_virial[2][0]);
    virial[5] += 0.5 * (predict_virial[1][2] + predict_virial[2][1]);
    
    click_timer("Update charge.");
    auto h_q = Kokkos::create_mirror_view(q);
    Kokkos::deep_copy(h_q, q);
    double *q_atom = atom->q;
    for(int i=0; i<inum; ++i){
        q_atom[i] = h_q(i);
    }
    click_timer("Update charge end.");

    if(debug_mode){
        std::cout << "virial[0]: " << virial[0] << "\n";
        std::cout << "virial[1]: " << virial[1] << "\n";
        std::cout << "virial[2]: " << virial[2] << "\n";
        std::cout << "virial[3]: " << virial[3] << "\n";
        std::cout << "virial[4]: " << virial[4] << "\n";
        std::cout << "virial[5]: " << virial[5] << "\n";
        std::cout << "eng_vdwl: "  << eng_vdwl << "\n";
        std::cout << "eng_ecoul: " << eng_coul << "\n";
    }

    click_timer("End postprocess");
    timer_counter++;
    copymode = 0;
}


template<class DeviceType>
void PairBAMBOOKokkos<DeviceType>::coeff(int narg, char **arg)
{
    PairBAMBOO::coeff(narg, arg);

    d_type_mapper = IntView1D("BAMBOO: type_mapper", type_mapper.size());
    auto h_type_mapper = Kokkos::create_mirror_view(d_type_mapper);
    for(int i = 0; i < type_mapper.size(); i++){
        h_type_mapper(i) = type_mapper[i];
    }
    Kokkos::deep_copy(d_type_mapper, h_type_mapper);
}


template<class DeviceType>
void PairBAMBOOKokkos<DeviceType>::init_style()
{
    PairBAMBOO::init_style();

    // irequest = neigh request made by parent class
    neighflag = lmp->kokkos->neighflag;
    int irequest = neighbor->nrequest - 1;

    auto req = neighbor->requests[irequest];

    req->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
        !std::is_same<DeviceType,LMPDeviceType>::value);

    req->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
    req->enable_full();
}

namespace LAMMPS_NS {
template class PairBAMBOOKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairBAMBOOKokkos<LMPHostType>;
#endif
}

