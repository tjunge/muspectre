/**
 * @file   projection_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   06 Dec 2017
 *
 * @brief  implementation of base class for projections
 *
 * Copyright © 2017 Till Junge
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

#include "fft/projection_base.hh"


namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  ProjectionBase<DimS, DimM>::ProjectionBase(FFT_Engine_ptr engine,
                                             Formulation form)
    : fft_engine{std::move(engine)}, form{form},
      projection_container{this->fft_engine->get_field_collection()}
  {
    static_assert((DimS == FFT_Engine::sdim),
                  "spatial dimensions are incompatible");
    static_assert((DimM == FFT_Engine::mdim),
                  "material dimensions are incompatible");

    /*
      throw an ProjectionError if one grid dimension is even, i.e.
      this->fft_engine->get_resolutions() % 2 == 0

      Problem: with test_till_random_material.py the programm crashes with
      'Segmentation fault (core dumped)'
      I think the problem is that I try to call
      this->fft_engine->get_resolutions()[i]
      before it is assigned!
      Solution: either check if resolutions is not empty
                or move ProjectionError to a place where resolution is always assigned.
    */

    /*
    for (auto res: this->fft_engine->get_resolutions()) {
      if (res % 2 == 0) {
      	throw ProjectionError
	  ("Only an odd number of gridpoints in each direction is supported");
      }
    }
    */

    /*

    for(int i = 0; i < DimS; i++) {
      if ((this->fft_engine->get_resolutions()[i] % 2) == 0) {
	throw ProjectionError
	  ("Only an odd number of gridpoints in each direction is supported");
      }
    }
    */
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void ProjectionBase<DimS, DimM>::
  initialise(FFT_PlanFlags flags) {
    fft_engine->initialise(flags);
  }

  template class ProjectionBase<twoD,   twoD>;
  template class ProjectionBase<twoD,   threeD>;
  template class ProjectionBase<threeD, threeD>;
}  // muSpectre
