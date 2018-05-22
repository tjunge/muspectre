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

  BOOST_AUTO_TEST_SUITE(crystal_plasticity);

  template <class Material>
  struct CrystalPlastFixture {
    using Mat_t = Material;
    constexpr static Dim_t get_DimS() {return Mat_t::sdim();}
    constexpr static Dim_t get_DimM() {return Mat_t::mdim();}
    constexpr static Dim_t get_NbSlip() {return Mat_t::get_NbSlip();}

    CrystalPlastFixture():
      mat(name, bulk_m, shear_m, gamma_dot0, m_par, tau_y0, h0, delta_tau_y, a_par, q_n, slip0, normals, delta_t)
    {}
    std::string name{"material"};
    Real bulk_m{175e9};
    Real shear_m{120e9};
    Real gamma_dot0{10e-2};
    Real m_par{.1};
    Real tau_y0{200e6};
    Real h0{10e9};
    Real delta_tau_y{100e6};
    Real a_par{2};
    Real q_n{1.4};
    typename Mat_t::SlipVecs slip0;
    typename Mat_t::SlipVecs normals;
    Real delta_t{1e-2};
    Mat_t mat;

  };

  /**
   * material with ridiculously high yield strength to test elastic sanity
   */
  template <class Material>
  struct Crystal_non_plastFixture {
    using Mat_t = Material;
    constexpr static Dim_t get_DimS() {return Mat_t::sdim();}
    constexpr static Dim_t get_DimM() {return Mat_t::mdim();}
    constexpr static Dim_t get_NbSlip() {return Mat_t::get_NbSlip();}

      Crystal_non_plastFixture():
      mat(name, bulk_m, shear_m, gamma_dot0, m_par, tau_y0, h0, delta_tau_y, a_par, q_n, slip0, normals, delta_t)
    {}
    std::string name{"material"};
    Real bulk_m{175e9};
    Real shear_m{120e9};
    Real gamma_dot0{10e-2};
    Real m_par{.1};
    Real tau_y0{1e200};
    Real h0{10e9};
    Real delta_tau_y{300e6};
    Real a_par{2};
    Real q_n{1.4};
    typename Mat_t::SlipVecs slip0;
    typename Mat_t::SlipVecs normals;
    Real delta_t{1e-2};

    Mat_t mat;

  };

  using mat_list = boost::mpl::list<
    CrystalPlastFixture<MaterialCrystalPlasticityFinite<  twoD,   twoD, 3 >>,
    CrystalPlastFixture<MaterialCrystalPlasticityFinite<  twoD, threeD, 12>>,
    CrystalPlastFixture<MaterialCrystalPlasticityFinite<threeD, threeD, 12>>>;


  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, mat_list, Fix) {

  }

  using elast_mat_list = boost::mpl::list<
    Crystal_non_plastFixture<MaterialCrystalPlasticityFinite<  twoD,   twoD, 3 >>,
    Crystal_non_plastFixture<MaterialCrystalPlasticityFinite<threeD, threeD, 12>>>;


  BOOST_FIXTURE_TEST_CASE_TEMPLATE(elastic_test_no_loop, Fix, elast_mat_list, Fix) {
    constexpr Dim_t Dim{Fix::get_DimS()};
    using Ccoord = Ccoord_t<Dim>;
    using Euler_t = Eigen::Array<Real, Fix::Mat_t::NbEuler, 1>;
    using T2_t = Eigen::Matrix<Real, Dim, Dim>;
    using T4_t = T4Mat<Real, Dim>;

    using Hooke = typename MatTB::Hooke<Dim, T2_t, T4_t>;

    auto & mat{Fix::mat};

    Euler_t angles = 2*pi*Euler_t::Random();

    constexpr Ccoord pixel{0};

    mat.add_pixel(pixel, angles);

    T2_t F{T2_t::Identity()};
    F(0,1) += .2;


    T2_t E{.5*(F.transpose()*F - T2_t::Identity())};

    const Real lambda{MatTB::convert_elastic_modulus
        <ElasticModulus::lambda,
         ElasticModulus::mu,
         ElasticModulus::Bulk>(Fix::shear_m, Fix::bulk_m)};

    T2_t stress_ref{Hooke::evaluate_stress(lambda, Fix::shear_m, E)};

    auto & internals{mat.get_internals()};

    mat.initialise();

    auto & Fp_map = std::get<0>(internals);
    auto & gamma_dot_map = std::get<1>(internals);
    auto & tau_y_map = std::get<2>(internals);
    auto & Euler_map = std::get<3>(internals);

    T2_t stress = mat.evaluate_stress(F,
                                      *Fp_map.begin(),
                                      *gamma_dot_map.begin(),
                                      *tau_y_map.begin(),
                                      *Euler_map.begin());

    Real error{(stress-stress_ref).norm()/stress_ref.norm()};
    BOOST_CHECK_LT(error, tol);
    if (error >= tol) {
      std::cout << "stress_ref =" << std::endl << stress_ref << std::endl;
      std::cout << "stress =" << std::endl << stress << std::endl;
    }


  }
  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
