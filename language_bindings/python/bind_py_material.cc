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
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include "common/common.hh"
#include "materials/material_base.hh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using namespace muSpectre;
namespace py = pybind11;
using namespace pybind11::literals;

/* ---------------------------------------------------------------------- */
template <Dim_t dim>
void add_material_linear_elastic1_helper(py::module & mod);
template <Dim_t dim>
void add_material_linear_elastic2_helper(py::module & mod);
template <Dim_t dim>
void add_material_linear_elastic3_helper(py::module & mod);
template <Dim_t dim>
void add_material_linear_elastic4_helper(py::module & mod);
template <Dim_t Dim>
void add_material_hyper_elasto_plastic1_helper(py::module & mod);
template <Dim_t dim, Dim_t NbSlip>
void add_material_crystal_plasticity_finite1_helper(py::module & mod);

/* ---------------------------------------------------------------------- */
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
