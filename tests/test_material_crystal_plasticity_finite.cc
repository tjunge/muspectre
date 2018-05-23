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
#include "materials/material_linear_elastic1.hh"
#include "materials/materials_toolbox.hh"
#include "common/tensor_algebra.hh"

#include "solver/solvers.hh"
#include "solver/solver_cg.hh"
#include "cell/cell_factory.hh"
#include "common/iterators.hh"

#include <boost/mpl/list.hpp>

#include <vector>

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
      slip0{Mat_t::SlipVecs::Zero()},
      normals{Mat_t::SlipVecs::Zero()},
      mat(name, bulk_m, shear_m, gamma_dot0, m_par, tau_y0, h0, delta_tau_y, a_par, q_n, slip0, normals, delta_t)
    {
      this->slip0.col(0).setConstant(1.);
      this->normals.col(1).setConstant(1.);
    }
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
    using MatElast_t = MaterialLinearElastic1<Dim, Dim>;
    Real Young{MatTB::convert_elastic_modulus<ElasticModulus::Young,
                                              ElasticModulus::Bulk,
                                              ElasticModulus::Shear>
        (Fix::bulk_m, Fix::shear_m)};
    Real Poisson{MatTB::convert_elastic_modulus<ElasticModulus::Poisson,
                                                ElasticModulus::Bulk,
                                                ElasticModulus::Shear>
        (Fix::bulk_m, Fix::shear_m)};
    MatElast_t mat_ref("hard", Young, Poisson);


    mat.add_pixel(pixel, angles);

    T2_t F{T2_t::Identity()};
    F(0,1) += .01;


    T2_t E{.5*(F.transpose()*F - T2_t::Identity())};

    const Real lambda{MatTB::convert_elastic_modulus
        <ElasticModulus::lambda,
         ElasticModulus::mu,
         ElasticModulus::Bulk>(Fix::shear_m, Fix::bulk_m)};

    T2_t stress_ref{Hooke::evaluate_stress(lambda, Fix::shear_m, E)};
    // for ∂E/∂F, see Curnier
    T4_t C{Hooke::compute_C_T4(lambda, Fix::shear_m)};
    T4_t tangent_ref{
      Matrices::ddot<Dim>(C,
                          (Matrices::outer_under(F.transpose(), T2_t::Identity())))};

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
    if (not(error < tol)) {
      std::cout << "stress_ref =" << std::endl << stress_ref << std::endl;
      std::cout << "stress =" << std::endl << stress << std::endl;
    }

    auto stress_tgt = mat.evaluate_stress_tangent(F,
                                                  *Fp_map.begin(),
                                                  *gamma_dot_map.begin(),
                                                  *tau_y_map.begin(),
                                                  *Euler_map.begin());

    auto stress_tgt_ref = mat_ref.evaluate_stress_tangent(E);

    T2_t & stress_2{get<0>(stress_tgt)};
    T4_t & tangent{get<1>(stress_tgt)};

    T2_t stress_2_lin{get<0>(stress_tgt_ref)};
    T4_t C_lin{get<1>(stress_tgt_ref)};
    T2_t I2{T2_t::Identity()};
    T4_t tangent_lin{C_lin*Matrices::outer_under(F.transpose(), I2)};

    error = (stress_2-stress_ref).norm()/stress_ref.norm();
    BOOST_CHECK_LT(error, tol);
    if (not (error < tol)) {
      std::cout << "stress_ref =" << std::endl << stress_ref << std::endl;
      std::cout << "stress_2 =" << std::endl << stress_2 << std::endl;
    }

    error = (stress_2 - stress_2_lin).norm() / stress_2_lin.norm();
    BOOST_CHECK_LT(error, tol);
    if (not (error < tol)) {
      std::cout << "stress_2 from linear elastic =" << std::endl << stress_2_lin << std::endl;
      std::cout << "stress_2 =" << std::endl << stress_2 << std::endl;
    }

    

    error = (tangent-tangent_ref).norm()/tangent_ref.norm();
    BOOST_CHECK_LT(error, tol);
    if (not (error < tol)) {
       std::cout << "tangent_ref =" << std::endl << tangent_ref << std::endl;
       std::cout << "tangent =" << std::endl << tangent << std::endl;
    }


    error = (tangent_lin-tangent_ref).norm()/tangent_ref.norm();
    BOOST_CHECK_LT(error, tol);
    if (not (error < tol)) {
       std::cout << "tangent_ref =" << std::endl << tangent_ref << std::endl;
       std::cout << "tangent from linear elastic =" << std::endl << tangent << std::endl;
    }
  }


  BOOST_FIXTURE_TEST_CASE_TEMPLATE(elastic_bimaterial, Fix, elast_mat_list, Fix) {
    constexpr Dim_t Dim{Fix::get_DimS()};
    using Ccoord = Ccoord_t<Dim>;
    using Rcoord = Rcoord_t<Dim>;


    constexpr Ccoord resolutions{CcoordOps::get_cube<Dim>(3)};
    constexpr Rcoord lengths{CcoordOps::get_cube<Dim>(1.)};
    constexpr Formulation form{Formulation::finite_strain};
    // number of layers in the hard material
    constexpr Uint nb_lays{1};

    static_assert(nb_lays < resolutions[0],
                  "the number or layers in the hard material must be smaller "
                  "than the total number of layers in dimension 0");

    auto sys{make_cell(resolutions, lengths, form)};
    auto sys_ref{make_cell(resolutions, lengths, form)};

    using Mat_t = typename Fix::Mat_t;

    Mat_t & hard = Mat_t::make(sys, "hard",
                               2*Fix::bulk_m,
                               2*Fix::shear_m,
                               Fix::gamma_dot0,
                               Fix::m_par,
                               Fix::tau_y0,
                               Fix::h0,
                               Fix::delta_tau_y,
                               Fix::a_par,
                               Fix::q_n,
                               Fix::slip0,
                               Fix::normals,
                               Fix::delta_t);

    Mat_t & soft = Mat_t::make(sys, "soft",
                               Fix::bulk_m,
                               Fix::shear_m,
                               Fix::gamma_dot0,
                               Fix::m_par,
                               Fix::tau_y0,
                               Fix::h0,
                               Fix::delta_tau_y,
                               Fix::a_par,
                               Fix::q_n,
                               Fix::slip0,
                               Fix::normals,
                               Fix::delta_t);


    using MatElast_t = MaterialLinearElastic1<Dim, Dim>;
    Real Young{MatTB::convert_elastic_modulus<ElasticModulus::Young,
                                              ElasticModulus::Bulk,
                                              ElasticModulus::Shear>
        (Fix::bulk_m, Fix::shear_m)};
    Real Poisson{MatTB::convert_elastic_modulus<ElasticModulus::Poisson,
                                                ElasticModulus::Bulk,
                                                ElasticModulus::Shear>
        (Fix::bulk_m, Fix::shear_m)};
    MatElast_t & hard_ref = MatElast_t::make(sys_ref, "hard", 2*Young, Poisson);
    MatElast_t & soft_ref = MatElast_t::make(sys_ref, "soft",   Young, Poisson);

    using Euler_t = Eigen::Array<Real, Fix::Mat_t::NbEuler, 1>;
    Euler_t angles = Euler_t::Zero();

    for (const auto & pixel: sys) {
      if (pixel[0] < Dim_t(nb_lays)) {
        hard.add_pixel(pixel, angles);
        hard_ref.add_pixel(pixel);
      } else {
        soft.add_pixel(pixel, angles);
        soft_ref.add_pixel(pixel);
      }
    }

    using Grad_t = Eigen::MatrixXd;
    Grad_t F_bar{Grad_t::Identity(Dim, Dim) + Grad_t::Ones(Dim, Dim)*.1};

    constexpr Real cg_tol{1e-4}, newton_tol{1e-4}, equil_tol{1e-4};
    constexpr Uint maxiter{Dim*10};
    constexpr Dim_t verbose{0};

    SolverCG solver{sys, cg_tol, maxiter, bool(verbose)};
    auto result = newton_cg(sys, F_bar, solver, newton_tol, equil_tol);
    std::cout << result.message << std::endl;

    SolverCG solver_ref{sys_ref, cg_tol, maxiter, bool(verbose)};
    auto result_ref = newton_cg(sys_ref, F_bar, solver_ref, newton_tol, equil_tol);

    Real error = (result.stress - result_ref.stress).matrix().norm() / result_ref.stress.matrix().norm();

    BOOST_CHECK_LT(error, tol);

  }
  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
