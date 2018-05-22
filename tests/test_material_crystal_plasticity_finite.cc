/**
 * file   test_material_crystal_plasticity_finite.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   22 May 2018
 *
 * @brief Tests for the basic crystal plasticity material,
 * MaterialCrystalPlasticityFinite
 *
 * @section LICENSE
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


#include "tests.hh"
#include "materials/material_crystal_plasticity_finite.hh"

#include <boost/mpl/list.hpp>


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(crystal_pasticity);

  template <class Material>
  struct CrystalPlastFixture {
    using Mat_t = Material;
    constexpr static Dim_t get_DimS() {return Mat_t::sdim();}
    constexpr static Dim_t get_DimM() {return Mat_t::mdim();}
    constexpr static Dim_t get_NbSlip() {return Mat_t::get_NbSlip();}
    Mat_t mat;

    std::string name{"material"};
    Real bulk_m{20e6};
    Real shear_m{30e6};
    Real gammaa_dot0{25};
    Real m_par{3};
    Real tau_y0{10e6};
    Real h0{1e-2};
    Real s_infty{12e6};
    Real a_par{4};
    Real q_n{1.4};

  };


  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
