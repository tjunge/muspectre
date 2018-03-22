/**
 * @file   material_crystal_plasticity_finite.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Francesco Maresca <francesco.maresca@epfl.ch>
 *
 * @date   23 Feb 2018
 *
 * @brief finite strain crystal plasticity implementation
 *
 * Copyright © 2018 Till Junge, Francesco Maresca
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "materials/material_crystal_plasticity_finite.hh"

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM, Int nb_slip>
  MaterialCrystalPlasticityFinite<DimS, DimM, nb_slip>::
  MaterialCrystalPlasticityFinite(std::string name, Real bulk_m, Real shear_m, Real gammadot0, Real m_par, Real tauy0, Real h0, Real s_infty, Real a_par, Real q_n, SlipVecs_ref Slip0, SlipVecs_ref Normal0)
    : Parent{name},
      FpField("Plastic Deformation Gradient Fₚ(t)",this->internal_fields),
      GammadotField("Plastic slip rates dγᵅ/dt",this->internal_fields),
      TauyField("Critical resolved shear stress τᵅy(t)",this->internal_fields),
      GammaField("Accumulated slips γᵅ(t)",this->internal_fields),
      bulk_m{bulk_m}, shear_m{shear_m}, gammadot_0{gammadot_0}, m_par{m_par}, tauy0{tauy0}, h0{h0},
      s_infty{s_infty}, a_par{a_par}, q_n{q_n}, Slip0{Slip0}, Normal0{Normal0};
  {}
  
} // muSpectre
