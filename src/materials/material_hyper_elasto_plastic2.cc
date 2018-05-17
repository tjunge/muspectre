/**
 * @file   material_hyper_elasto_plastic2.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   08 May 2018
 *
 * @brief  implementation for MaterialHyperElastoPlastic2
 *
 * Copyright © 2018 Till Junge
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

#include "materials/material_hyper_elasto_plastic2.hh"


namespace muSpectre {

  //----------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  MaterialHyperElastoPlastic2<DimS, DimM>::
  MaterialHyperElastoPlastic2(std::string name)
    : Parent{name},
      plast_flow_field("cumulated plastic flow εₚ", this->internal_fields),
      F_prev_field("Previous placement gradient Fᵗ", this->internal_fields),
      be_prev_field("Previous left Cauchy-Green deformation bₑᵗ",
                    this->internal_fields),
      lambda_field{make_field<Field_t>("local first Lame constant",
					this->internal_fields )},
      mu_field{make_field<Field_t>("local second lame constant",
					this->internal_fields )},
      tau_y0_field{make_field<Field_t>("local yield stress",
				       this->internal_fields )},
      H_field{make_field<Field_t>("local hardening modulus",
				  this->internal_fields )},

      internal_variables{F_prev_field.get_map(), be_prev_field.get_map(),
          plast_flow_field.get_map(), lambda_field.get_const_map(),
	  mu_field.get_const_map(), tau_y0_field.get_const_map(),
	  H_field.get_const_map()}
  {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialHyperElastoPlastic2<DimS, DimM>::save_history_variables() {
    this->plast_flow_field.cycle();
    this->F_prev_field.cycle();
    this->be_prev_field.cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialHyperElastoPlastic2<DimS, DimM>::initialise(bool /*stiffness*/) {
    Parent::initialise();
    this->F_prev_field.get_map().current() =
      Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->be_prev_field.get_map().current() =
      Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->save_history_variables();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialHyperElastoPlastic2<DimS, DimM>::
  add_pixel(const Ccoord_t<DimS> & /*pixel*/) {
    throw std::runtime_error
      ("This material needs pixels with Youngs modulus, Poisson ratio,\n"
       "Yield stress and Hardening modulus.");
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialHyperElastoPlastic2<DimS, DimM>::
  add_pixel(const Ccoord_t<DimS> & pixel,
            const Real & Youngs_modulus, const Real & Poisson_ratio,
	    const Real & tau_y0, const Real & H) {
    this->internal_fields.add_pixel(pixel);
    this->tau_y0_field.push_back(tau_y0);
    this->H_field.push_back(H);

    // compute lambda and mu
    auto lambda{Hooke::compute_lambda(Youngs_modulus, Poisson_ratio)};
    auto mu{Hooke::compute_mu(Youngs_modulus, Poisson_ratio)};

    this->lambda_field.push_back(lambda);
    this->mu_field.push_back(mu);
  }


  template class MaterialHyperElastoPlastic2<  twoD,   twoD>;
  template class MaterialHyperElastoPlastic2<  twoD, threeD>;
  template class MaterialHyperElastoPlastic2<threeD, threeD>;
}  // muSpectre
