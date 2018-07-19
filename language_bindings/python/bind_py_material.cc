/**
 * @file   bind_py_material.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   09 Jan 2018
 *
 * @brief  python bindings for µSpectre's materials
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

#include "common/common.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/material_linear_elastic2.hh"
#include "materials/material_linear_elastic3.hh"
#include "materials/material_linear_elastic4.hh"
#include "materials/material_crystal_plasticity_finite.hh"
#include "materials/material_hyper_elasto_plastic1.hh"
#include "cell/cell_base.hh"
#include "common/field_collection.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using namespace muSpectre;
namespace py = pybind11;
using namespace pybind11::literals;

/**
 * python binding for the optionally objective form of Hooke's law
 */
template <Dim_t dim>
void add_material_linear_elastic1_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElastic1_" << dim << 'd';
  const auto name {name_stream.str()};

  using Mat_t = MaterialLinearElastic1<dim, dim>;
  using Sys_t = CellBase<dim, dim>;
  py::class_<Mat_t, MaterialBase<dim, dim>>(mod, name.c_str())
    .def_static("make",
                [](Sys_t & sys, std::string n, Real e, Real p) -> Mat_t & {
                  return Mat_t::make(sys, n, e, p);
                },
                "cell"_a, "name"_a, "Young"_a, "Poisson"_a,
                py::return_value_policy::reference, py::keep_alive<1, 0>());
}

/* ---------------------------------------------------------------------- */
template <Dim_t dim>
void add_material_linear_elastic2_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElastic2_" << dim << 'd';
  const auto name {name_stream.str()};

  using Mat_t = MaterialLinearElastic2<dim, dim>;
  using Sys_t = CellBase<dim, dim>;

  py::class_<Mat_t, MaterialBase<dim, dim>>(mod, name.c_str())
     .def_static("make",
                [](Sys_t & sys, std::string n, Real e, Real p) -> Mat_t & {
                  return Mat_t::make(sys, n, e, p);
                },
                "cell"_a, "name"_a, "Young"_a, "Poisson"_a,
                py::return_value_policy::reference, py::keep_alive<1, 0>())
    .def("add_pixel",
         [] (Mat_t & mat, Ccoord_t<dim> pix, py::EigenDRef<Eigen::ArrayXXd>& eig) {
           Eigen::Matrix<Real, dim, dim> eig_strain{eig};
           mat.add_pixel(pix, eig_strain);},
         "pixel"_a,
         "eigenstrain"_a);
}


/* ---------------------------------------------------------------------- */
template <Dim_t dim>
void add_material_linear_elastic3_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElastic3_" << dim << 'd';
  const auto name {name_stream.str()};

  using Mat_t = MaterialLinearElastic3<dim, dim>;
  using Sys_t = CellBase<dim, dim>;

  py::class_<Mat_t, MaterialBase<dim, dim>>(mod, name.c_str())
    .def_static("make",
                [](Sys_t & sys, std::string n) -> Mat_t & {
                  return Mat_t::make(sys, n);
                },
                "cell"_a, "name"_a,
                py::return_value_policy::reference, py::keep_alive<1, 0>())
    .def("add_pixel",
         [] (Mat_t & mat, Ccoord_t<dim> pix, Real Young, Real Poisson) {
	   mat.add_pixel(pix, Young, Poisson);},
         "pixel"_a,
         "Young"_a,
         "Poisson"_a);
}

/* ---------------------------------------------------------------------- */
template <Dim_t dim>
void add_material_linear_elastic4_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElastic4_" << dim << 'd';
  const auto name {name_stream.str()};

  using Mat_t = MaterialLinearElastic4<dim, dim>;
  using Sys_t = CellBase<dim, dim>;

  py::class_<Mat_t, MaterialBase<dim, dim>>(mod, name.c_str())
    .def_static("make",
                [](Sys_t & sys, std::string n) -> Mat_t & {
                  return Mat_t::make(sys, n);
                },
                "cell"_a, "name"_a,
                py::return_value_policy::reference, py::keep_alive<1, 0>())
    .def("add_pixel",
         [] (Mat_t & mat, Ccoord_t<dim> pix, Real Young, Real Poisson) {
	   mat.add_pixel(pix, Young, Poisson);},
         "pixel"_a,
         "Young"_a,
         "Poisson"_a);
}

/* ---------------------------------------------------------------------- */
template <Dim_t Dim>
void add_material_hyper_elasto_plastic1_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialHyperElastoPlastic1_" << Dim << "d";
  const auto name {name_stream.str()};

  using Mat_t = MaterialHyperElastoPlastic1<Dim, Dim>;
  using Cell_t = CellBase<Dim, Dim>;

  py::class_<Mat_t, MaterialBase<Dim, Dim>>(mod, name.c_str())
    .def_static
    ("make",
     [](Cell_t & cell, std::string name, Real Young, Real Poisson, Real tau_y0,
        Real h) -> Mat_t & {
      return Mat_t::make(cell, name, Young, Poisson, tau_y0, h);
    },
     "cell"_a,
     "name"_a,
     "YoungModulus"_a,
     "PoissonRatio"_a,
     "τ_y₀"_a,
     "h"_a,
     py::return_value_policy::reference, py::keep_alive<1, 0>());
}

/* ---------------------------------------------------------------------- */
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


template <Dim_t Dim>
class PyMaterialBase : public MaterialBase<Dim,  Dim>  {
public:
  /* Inherit the constructors */
  using Parent = MaterialBase<Dim,  Dim>;
  using Parent::Parent;

  /* Trampoline (need one for each virtual function) */
  void save_history_variables() override {
    PYBIND11_OVERLOAD_PURE
      (void, /* Return type */
       Parent,      /* Parent class */
       save_history_variables          /* Name of function in C++ (must match Python name) */
       );
  }

  /* Trampoline (need one for each virtual function) */
  void initialise() override {
    PYBIND11_OVERLOAD_PURE
      (void, /* Return type */
       Parent,      /* Parent class */
       initialise          /* Name of function in C++ (must match Python name) */
       );
  }
  virtual void compute_stresses(const typename Parent::StrainField_t & F,
                                typename Parent::StressField_t & P,
                                Formulation form) override {
    PYBIND11_OVERLOAD_PURE
      (void, /* Return type */
       Parent,      /* Parent class */
       compute_stresses,          /* Name of function in C++ (must match Python name) */
       F,P,form
       );
  }

  virtual void compute_stresses_tangent(const typename Parent::StrainField_t & F,
                                        typename Parent::StressField_t & P,
                                        typename Parent::TangentField_t & K,
                                        Formulation form) override {
    PYBIND11_OVERLOAD_PURE
      (void, /* Return type */
       Parent,      /* Parent class */
       compute_stresses,          /* Name of function in C++ (must match Python name) */
       F,P,K, form
       );
  }

};



template <Dim_t dim>
void add_material_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialBase_" << dim << "d";
  std::string name{name_stream.str()};
  using Material = MaterialBase<dim, dim>;
  using MaterialTrampoline = PyMaterialBase<dim>;
  using FC_t = LocalFieldCollection<dim>;
  using FCBase_t = FieldCollectionBase<dim, FC_t>;

  py::class_<Material, MaterialTrampoline /* <--- trampoline*/>(mod, name.c_str())
    .def(py::init<std::string>())
    .def("save_history_variables", &Material::save_history_variables)
    .def("list_fields", &Material::list_fields)
    .def("get_real_field", &Material::get_real_field, "field_name"_a,
         py::return_value_policy::reference_internal)
    .def("size", &Material::size)
    .def("add_pixel",
         [] (Material & mat, Ccoord_t<dim> pix) {
           mat.add_pixel(pix);},
         "pixel"_a)
    .def_property_readonly("collection",
                          [](Material & material) -> FCBase_t &{
                            return material.get_collection();},
                          "returns the field collection containing internal "
                          "fields of this material");

  add_material_linear_elastic1_helper<dim>(mod);
  add_material_linear_elastic2_helper<dim>(mod);
  add_material_linear_elastic3_helper<dim>(mod);
  add_material_linear_elastic4_helper<dim>(mod);
  add_material_hyper_elasto_plastic1_helper<dim>(mod);
}

void add_material(py::module & mod) {
  auto material{mod.def_submodule("material")};
  material.doc() = "bindings for constitutive laws";
  add_material_helper<twoD  >(material);
  add_material_helper<threeD>(material);

  add_material_crystal_plasticity_finite1_helper<  twoD,  1>(material);
  add_material_crystal_plasticity_finite1_helper<  twoD,  3>(material);
  add_material_crystal_plasticity_finite1_helper<threeD, 12>(material);

}
