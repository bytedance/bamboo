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

PairStyle(bamboo/kk,PairBAMBOOKokkos<LMPDeviceType>);
PairStyle(bamboo/kk/device,PairBAMBOOKokkos<LMPDeviceType>);
PairStyle(bamboo/kk/host,PairBAMBOOKokkos<LMPHostType>);

#else

#ifndef LMP_PAIR_BAMBOO_KOKKOS_H
#define LMP_PAIR_BAMBOO_KOKKOS_H


#include "pair_bamboo.h"
#include <pair_kokkos.h>
#include <time.h>
#include <torch/torch.h>

#include <cmath>
#include "kokkos.h"
#include "atom_kokkos.h"
#include "domain_kokkos.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "error.h"
#include "atom_masks.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairBAMBOOKokkos : public PairBAMBOO {
public:
   using MemberType = typename Kokkos::TeamPolicy<DeviceType>::member_type;
   typedef ArrayTypes<DeviceType> AT;

   PairBAMBOOKokkos(class LAMMPS *);
   virtual ~PairBAMBOOKokkos();
   // Override compute method from PairBAMBOO
   virtual void compute(int eflag, int vflag) override;

   // Override coeff method from PairBAMBOO
   virtual void coeff(int narg, char **arg) override;

   // Override init_style method from PairBAMBOO
   virtual void init_style() override;

   typename AT::t_efloat_1d d_eatom;
   typename AT::t_virial_array d_vatom;

   template<typename ViewType>
   static void reallocateView(ViewType& view, const std::string& name, const size_t dim1) {
      view = ViewType(); // Reset the view
      view = ViewType(Kokkos::ViewAllocateWithoutInitializing(name), dim1);
   }

   template<typename ViewType>
   static void reallocateView(ViewType& view, const std::string& name, const size_t dim1, const size_t dim2) {
      view = ViewType(); // Reset the view
      view = ViewType(Kokkos::ViewAllocateWithoutInitializing(name), dim1, dim2);
   }

   template<typename ViewType>
   static void reallocateView(ViewType& view, const std::string& name, const size_t dim1, const size_t dim2, const size_t dim3) {
      view = ViewType(); // Reset the view
      view = ViewType(Kokkos::ViewAllocateWithoutInitializing(name), dim1, dim2, dim3);
   }

protected:


   class DomainKokkos *domainKK;

   using IntView1D = Kokkos::View<int*, Kokkos::LayoutRight, DeviceType>;
   using IntView2D = Kokkos::View<int**, Kokkos::LayoutRight, DeviceType>;
   using LongView1D = Kokkos::View<long*, Kokkos::LayoutRight, DeviceType>;
   using LongView2D = Kokkos::View<long**, Kokkos::LayoutRight, DeviceType>;
   using View1D = Kokkos::View<F_FLOAT*, Kokkos::LayoutRight, DeviceType>;
   using View2D = Kokkos::View<F_FLOAT**, Kokkos::LayoutRight, DeviceType>;
   using FloatView2D = Kokkos::View<float**, Kokkos::LayoutRight, DeviceType>;
   using DoubleView2D = Kokkos::View<double**, Kokkos::LayoutRight, DeviceType>;
   using DoubleView3D = Kokkos::View<double***, Kokkos::LayoutRight, DeviceType>;
   using UnmanagedFloatView1D = Kokkos::View<float*, Kokkos::LayoutRight, DeviceType>;
   using UnmanagedFloatView2D = Kokkos::View<float**, Kokkos::LayoutRight, DeviceType>;
   using UnmanagedDoubleView1D = Kokkos::View<double*, Kokkos::LayoutRight, DeviceType>;
   using UnmanagedDoubleView2D = Kokkos::View<double**, Kokkos::LayoutRight, DeviceType>;

   typename AT::t_x_array_randomread x;
   typename AT::t_f_array f;
   typename AT::t_tagint_1d tag;
   typename AT::t_float_1d q;
   typename AT::t_int_1d_randomread type;
   typename AT::t_neighbors_2d d_neighbors;
   typename AT::t_int_1d_randomread d_ilist;
   typename AT::t_int_1d_randomread d_numneigh;
   
   DAT::tdual_efloat_1d k_eatom;
   DAT::tdual_virial_array k_vatom;

   View1D d_ewald;
   IntView1D d_type_mapper;
   LongView1D d_atom_types;
   LongView2D d_disp_edges, d_coul_edges, d_net_edges;
   DoubleView2D d_atom_pos;

   IntView1D d_numneigh_net, d_numneigh_coul, d_numneigh_disp;
   IntView1D d_cumsum_numneigh_net, d_cumsum_numneigh_coul, d_cumsum_numneigh_disp;
   IntView2D d_neighbors_net, d_neighbors_coul, d_neighbors_disp;
   DoubleView3D d_edge_shift;
   DoubleView2D d_edge_shift_net, d_edge_shift_coul, d_edge_shift_disp;

   int neighflag, newton_pair;
   int nlocal, nall, eflag, vflag;
};

}

#endif
#endif

