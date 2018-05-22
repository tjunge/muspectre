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
#include "common/iterators.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  MaterialCrystalPlasticityFinite(std::string name, Real bulk_m, Real shear_m, Real gamma_dot0, Real m_par, Real tau_y0, Real h0, Real delta_tau_y_max, Real a_par, Real q_n, SlipVecs_ref Slip0, SlipVecs_ref Normal0, Real delta_t, Real tolerance, Int maxiter)
    : Parent{name},
      FpField("Plastic Deformation Gradient Fₚ(t)",this->internal_fields),
      GammaDotField("Plastic slip rates dγᵅ/dt",this->internal_fields),
      TauYField("Critical resolved shear stress τᵅy(t)",this->internal_fields),
      GammaField{
        make_field<MatrixField<LColl_t, Real, NbSlip, 1>>
          ("Accumulated slips γᵅ(t)",this->internal_fields)},
      EulerField{
        make_field<MatrixField<LColl_t, Real, NbEuler, 1>>
          ("Euler angles", this->internal_fields)},
      bulk_m{bulk_m}, shear_m{shear_m}, gamma_dot0{gamma_dot0}, m_par{m_par},
      tau_y0{tau_y0}, h0{h0}, tau_infty{tau_y0+delta_tau_y_max}, a_par{a_par}, q_n{q_n},
      delta_t{delta_t}, tolerance{tolerance}, maxiter{maxiter}, Slip0{Slip0},
      Normal0{Normal0},
      internal_variables{FpField.get_map(),
          GammaDotField.get_map(),
          TauYField.get_map(),
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
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  void MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  initialise() {
    if (not this->is_initialised) {
      Parent::initialise();
      using T2_t = Eigen::Matrix<Real, DimM, DimM>;
      this->FpField.get_map().current() = T2_t::Identity();

      using ColArray_t = Eigen::Matrix<Real, NbSlip, 1>;
      this->GammaDotField.get_map().current() = ColArray_t::Zero();
      this->TauYField.get_map().current() = ColArray_t::Constant(this->tau_y0);

      this->GammaField.set_zero();

      this->FpField.cycle();
      this->GammaDotField.cycle();
      this->TauYField.cycle();

    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  void MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  save_history_variables() {
    auto GammaMap = this->GammaField.get_map();
    auto GammaDotMap = this->GammaDotField.get_map();

    for (auto && tup: akantu::zip(GammaMap, GammaDotMap)) {
      auto & gamma = std::get<0>(tup);
      auto & gamma_dot = std::get<1>(tup);
      gamma += .5 * this->delta_t * (gamma_dot.current() + gamma_dot.old());
    }

    this->FpField.cycle();
    this->GammaDotField.cycle();
    this->TauYField.cycle();

  }

  /* ---------------------------------------------------------------------- */
  template class MaterialCrystalPlasticityFinite<  twoD,   twoD,  3>;
  template class MaterialCrystalPlasticityFinite<  twoD, threeD, 12>;
  template class MaterialCrystalPlasticityFinite<threeD, threeD, 12>;


} // muSpectre
