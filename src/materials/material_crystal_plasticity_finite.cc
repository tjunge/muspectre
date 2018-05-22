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

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  MaterialCrystalPlasticityFinite(std::string name, Real bulk_m, Real shear_m, Real gammadot_0, Real m_par, Real tauy0, Real h0, Real s_infty, Real a_par, Real q_n, SlipVecs_ref Slip0, SlipVecs_ref Normal0, Real DeltaT, Real tolerance, Int maxiter)
    : Parent{name},
      FpField("Plastic Deformation Gradient Fₚ(t)",this->internal_fields),
      GammadotField("Plastic slip rates dγᵅ/dt",this->internal_fields),
      TauyField("Critical resolved shear stress τᵅy(t)",this->internal_fields),
      GammaField{
        make_field<MatrixField<LColl_t, Real, NbSlip, 1>>
          ("Accumulated slips γᵅ(t)",this->internal_fields)},
      EulerField{
        make_field<MatrixField<LColl_t, Real, NbEuler, 1>>
          ("Euler angles", this->internal_fields)},
      bulk_m{bulk_m}, shear_m{shear_m}, gammadot_0{gammadot_0}, m_par{m_par},
      tauy0{tauy0}, h0{h0}, s_infty{s_infty}, a_par{a_par}, q_n{q_n},
      DeltaT{DeltaT}, tolerance{tolerance}, maxiter{maxiter}, Slip0{Slip0},
      Normal0{Normal0},
      internal_variables{FpField.get_map(),
          GammadotField.get_map(),
          TauyField.get_map(),
          GammaField.get_map(),
          ArrayFieldMap<LColl_t, Real, NbEuler, 1, true>(EulerField)}
  {
    // Enforce n₀ and s₀ to be unit vectors!
    auto lambda{MatTB::convert_elastic_modulus<ElasticModulus::lambda,
                                               ElasticModulus::Bulk,
                                               ElasticModulus::Shear>(bulk_m,shear_m)};
    this->C_el=Hooke::compute_C_T4(lambda,shear_m);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  void MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  add_pixel(const Ccoord_t<DimS> & /*pixel*/) {
    throw std::runtime_error
      ("this material needs pixels with a column vector of Euler angles");
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  void MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  add_pixel(const Ccoord_t<DimS> & pixel,
	    const Eigen::Ref<Eigen::Array<Real, NbEuler, 1>> Euler){
    this->internal_fields.add_pixel(pixel);
    this->EulerField.push_back(Euler);
  }


  /* ---------------------------------------------------------------------- */
  template class MaterialCrystalPlasticityFinite<  twoD,   twoD,  7>;
  template class MaterialCrystalPlasticityFinite<  twoD, threeD, 12>;
  template class MaterialCrystalPlasticityFinite<threeD, threeD, 12>;


} // muSpectre
