/**
 * @file   bind_py_material_crystal_plasticity_finite.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 Oct 2018
 *
 * @brief  python binding for MaterialCrystalPlasticityFinite
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

#include "common/common.hh"
#include "materials/stress_transformations_PK2.hh"
#include "materials/material_crystal_plasticity_finite.hh"
#include "cell/cell_base.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using namespace muSpectre;
namespace py = pybind11;
using namespace pybind11::literals;

/**
 * python binding for the standard crystal plasticity model of Francesco
 */
template <Dim_t dim, Dim_t NbSlip>
void add_material_crystal_plasticity_finite1_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialCrystalPlasticityFinite_" << dim << "d_"
              << NbSlip << "slip";
  const auto name {name_stream.str()};

  using Mat_t = MaterialCrystalPlasticityFinite<dim, dim, NbSlip>;
  using Sys_t = CellBase<dim, dim>;

  py::class_<Mat_t, MaterialBase<dim, dim>>(mod, name.c_str())
    .def_static("make",
                [](Sys_t & sys,
                   std::string n,
                   Real bulk_modulus,
                   Real shear_modulus,
                   Real gamma_dot_0,
                   Real m,
                   Real tau_y0,
                   Real h0,
                   Real delta_tau_y_max,
                   Real a,
                   Real q_n,
                   py::EigenDRef<Eigen::MatrixXd> slip,
                   py::EigenDRef<Eigen::MatrixXd> normals,
                   Real delta_t,
                   Real tolerance,
                   Int maxiter) -> Mat_t & {

                  auto check = [] (auto && mat, auto && name ) {
                    if (not(
                            (mat.rows() == NbSlip) and
                            (mat.cols() == dim))) {
                      std::stringstream err{};
                      err << "The " << name << " need to be given in the form of a "
                      << NbSlip << "×" << dim << " matrix, but you gave a "
                      << mat.rows() << "×" << mat.cols() << " matrix.";
                      throw std::runtime_error(err.str());
                    }
                  };

                  check(slip, "slip directions");
                  check(normals, "normals to slip planes");
                  Eigen::Matrix<Real, NbSlip, dim> real_slip{slip};
                  Eigen::Matrix<Real, NbSlip, dim> real_normals{normals};
                  return Mat_t::make(sys,
                                     n,
                                     bulk_modulus,
                                     shear_modulus,
                                     gamma_dot_0,
                                     m,
                                     tau_y0,
                                     h0,
                                     delta_tau_y_max,
                                     a,
                                     q_n,
                                     real_slip,
                                     real_normals,
                                     delta_t,
                                     tolerance,
                                     maxiter);
                },
                "cell"_a,
                "name"_a,
                "bulk_modulus"_a,
                "shear_modulus"_a,
                "dγ/dt₀"_a,
                "m_exonent"_a,
                "tau_y₀"_a,
                "h₀"_a,
                "Δτ_y_max"_a,
                "a_exponent"_a,
                "qₙ"_a,
                "slip_directions"_a,
                "normals"_a,
                "Δt"_a,
                "tolerance"_a=1.e-4,
                "max_iter"_a= 20,
                py::return_value_policy::reference, py::keep_alive<1, 0>())
    .def("add_pixel",
         [] (Mat_t & mat, Ccoord_t<dim> pix, py::EigenDRef<Eigen::MatrixXd> euler) {
           auto check = [] (auto && mat, auto && name ) {
             if (not(
                     (mat.rows() == NbSlip) and
                     (mat.cols() == 1))) {
               std::stringstream err{};
               err << "The " << name << " need to be given in the form of a "
               << Mat_t::NbEuler << "×" << 1 << " matrix, but you gave a "
               << mat.rows() << "×" << mat.cols() << " matrix.";
               throw std::runtime_error(err.str());
             }
           };
           check(euler, "euler angles");
           Eigen::Matrix<Real, Mat_t::NbEuler, 1> real_euler{euler};
           mat.add_pixel(pix, real_euler);},
         "pixel"_a,
         "euler_angles"_a);
}


template
void add_material_crystal_plasticity_finite1_helper<  twoD,  1>(py::module &);

template
void add_material_crystal_plasticity_finite1_helper<  twoD,  3>(py::module &);

template
void add_material_crystal_plasticity_finite1_helper<threeD, 12>(py::module &);
