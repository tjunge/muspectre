/**
 * file   projection_finite_strain_fast.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   12 Dec 2017
 *
 * @brief  implementation for fast projection in finite strain
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include "fft/projection_finite_strain_fast.hh"
#include "fft/fft_utils.hh"
#include "common/tensor_algebra.hh"
#include "common/iterators.hh"

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  ProjectionFiniteStrainFast<DimS, DimM>::
  ProjectionFiniteStrainFast(FFT_Engine_ptr engine)
    :Parent{std::move(engine)},
     xis{make_field<Proj_t>("Projection Operator",
                            this->projection_container)}
  {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void ProjectionFiniteStrainFast<DimS, DimM>::
  initialise(FFT_PlanFlags flags) {
    Parent::initialise(flags);
    FFT_freqs<DimS> fft_freqs(this->fft_engine->get_resolutions(),
                              this->fft_engine->get_lengths());
    for (auto && tup: akantu::zip(*this->fft_engine, this->xis)) {
      const auto & ccoord = std::get<0> (tup);
      auto & xi = std::get<1>(tup);
      xi = fft_freqs.get_unit_xi(ccoord);
    }
    this->xis[0].setZero();
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void ProjectionFiniteStrainFast<DimS, DimM>::apply_projection(Field_t & field) {
    Grad_map field_map{this->fft_engine->fft(field)};
    Real factor = this->fft_engine->normalisation();
    for (auto && tup: akantu::zip(this->xis, field_map)) {
      auto & xi{std::get<0>(tup)};
      auto & f{std::get<1>(tup)};
      f = factor * ((f*xi).eval()*xi.transpose());
    }
    this->fft_engine->ifft(field);
  }

  template class ProjectionFiniteStrainFast<twoD,   twoD>;
  template class ProjectionFiniteStrainFast<threeD, threeD>;
}  // muSpectre
