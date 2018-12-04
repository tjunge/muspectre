/**
 * @file   test_material_hyper_elasto_plastic1_comparison.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   30 Oct 2018
 *
 * @brief  simple wrapper around the MaterialHyperElastoPlastic1 to test it
 *         with arbitrary input
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "materials/stress_transformations_Kirchhoff.hh"
#include "materials/material_hyper_elasto_plastic1.hh"
#include "materials/materials_toolbox.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <tuple>

namespace py = pybind11;
using namespace pybind11::literals;


namespace muSpectre {

  template <Dim_t Dim>
  py::tuple test_fun(Real K, Real mu, Real H, Real tau_y0,
                     py::EigenDRef<const Eigen::MatrixXd> F,
                     py::EigenDRef<const Eigen::MatrixXd> F_prev,
                     py::EigenDRef<const Eigen::MatrixXd> be_prev,
                     Real eps_prev) {
    const Real Young{MatTB::convert_elastic_modulus<ElasticModulus::Young,
                                              ElasticModulus::Bulk,
                                              ElasticModulus::Shear>(K, mu)};
    const Real Poisson{
      MatTB::convert_elastic_modulus<ElasticModulus::Poisson,
                                     ElasticModulus::Bulk,
                                     ElasticModulus::Shear>(K, mu)};

    using Mat_t = MaterialHyperElastoPlastic1<Dim, Dim>;
    Mat_t mat("Name", Young, Poisson, tau_y0, H);

    auto & coll{mat.get_collection()};
    coll.add_pixel({0});
    coll.initialise();

    auto & F_{mat.get_F_prev_field()};
    auto & be_{mat.get_be_prev_field()};
    auto & eps_{mat.get_plast_flow_field()};

    F_.get_map()[0].current() = F_prev;
    be_.get_map()[0].current() = be_prev;
    eps_.get_map()[0].current() = eps_prev;
    mat.save_history_variables();

    auto && tup{mat.evaluate_stress_tangent(F, F_.get_map()[0],
                                            be_.get_map()[0],
                                            eps_.get_map()[0])};

    auto && tau{std::get<0>(tup)};
    auto && C{std::get<1>(tup)};

    return py::make_tuple(std::move(tau), std::move(C));
  }

  PYBIND11_MODULE(material_hyper_elasto_plastic1, mod) {
    mod.doc() = "Comparison provider for MaterialHyperElastoPlastic1";
    auto adder{[&mod](auto name, auto & fun) {
        mod.def(name, &fun,
                "K"_a,
                "mu"_a,
                "H"_a,
                "tau_y0"_a,
                "F"_a,
                "F_prev"_a,
                "be_prev"_a,
                "eps_prev"_a);
      }};
    adder("test_fun_2d", test_fun<  twoD>);
    adder("test_fun_3d", test_fun<threeD>);
  }


}  // muSpectre
