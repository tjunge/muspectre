/**
 * file   bind_py_field_collection.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Jul 2018
 *
 * @brief  Python bindings for µSpectre field collections
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

#include "common/common.hh"
#include "common/field.hh"
#include "common/field_collection.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using namespace muSpectre;
namespace py = pybind11;
using namespace pybind11::literals;

template <Dim_t Dim, class FieldCollectionDerived>
void add_field_collection(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "_" << (FieldCollectionDerived::Global ? "Global" : "Local")
              << "FieldCollection_" << Dim << 'd';
  const auto name {name_stream.str()};
  using FC_t = FieldCollectionBase<Dim, FieldCollectionDerived>;
  py::class_<FC_t>(mod, name.c_str())
    .def("get_real_field",
         [](FC_t & field_collection, std::string unique_name)
         -> typename FC_t::template TypedField_t<Real> & {
           return field_collection.template get_typed_field<Real>(unique_name);
         },
         "unique_name"_a)
    .def("get_int_field",
         [](FC_t & field_collection, std::string unique_name)
         -> typename FC_t::template TypedField_t<Int> & {
           return field_collection.template get_typed_field<Int>(unique_name);
         },
         "unique_name"_a)
    .def("get_current_int",
         [](FC_t & field_collection, std::string unique_prefix)
         -> typename FC_t::template TypedField_t<Int> & {
           return field_collection.template get_current<Int>(unique_prefix);
         },
         "unique_prefix"_a)
    .def("get_current_int",
         [](FC_t & field_collection, std::string unique_prefix)
         -> typename FC_t::template TypedField_t<Int> & {
           return field_collection.template get_current<Int>(unique_prefix);
         },
         "unique_prefix"_a);
}

template <typename T, class FieldCollection>
void add_field(py::module & mod, std::string dtype_name) {
  using Field_t = TypedField<FieldCollection, T>;
  std::stringstream name_stream{};
  name_stream << (FieldCollection::Global ? "Global" : "Local")
              << "Field" << dtype_name << "_" << FieldCollection::spatial_dim();
  std::string name{name_stream.str()};
  py::class_<Field_t, typename Field_t::Parent>(mod, name.c_str())
    .def("eigen", &Field_t::eigen, "array of stored data")
    .def("eigenvec", &Field_t::eigen, "flattened array of stored data");
}

template <Dim_t Dim, class FieldCollection>
void add_field_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << (FieldCollection::Global ? "Global" : "Local")
              << "Field" << "_" << Dim;
  std::string name{name_stream.str()};
  using Field_t = internal::FieldBase<FieldCollection>;
  py::class_<Field_t>(mod, name.c_str())
    .def_property_readonly("name", &Field_t::get_name, "field name")
    .def_property_readonly("collection", &Field_t::get_collection,
                           "Collection containing this field")
    .def_property_readonly("nb_components", &Field_t::get_nb_components,
                           "number of scalars stored per pixel in this field")
    .def_property_readonly("stored_type",
                           [](const Field_t & field) {
                             return field.get_stored_typeid().name();
                           },
                           "fundamental type of scalars stored in this field")
    .def_property_readonly("size", &Field_t::size, "number of pixels in this field")
    .def("set_zero", &Field_t::set_zero,
         "Set all components in the field to zero");

  add_field<Real, FieldCollection>(mod, "Real");
  add_field<Int,  FieldCollection>(mod, "Int");
}

template <typename T, class FieldCollection>
void add_statefield(py::module & mod, std::string dtype_name) {
  using StateField_t = TypedStateField<FieldCollection, T>;
  std::stringstream name_stream{};
  name_stream << (FieldCollection::Global ? "Global" : "Local")
              << "StateField" << dtype_name
              << "_" << FieldCollection::spatial_dim();
  std::string name{name_stream.str()};
  py::class_<StateField_t, typename StateField_t::Parent>(mod, name.c_str())
    .def("get_current_field", &StateField_t::get_current_field,
         "returns the current field value")
    .def("get_old_field", &StateField_t::get_old_field,
         "nb_steps_ago"_a = 1,
         "returns the value this field held 'nb_steps_ago' steps ago");
}

template <Dim_t Dim, class FieldCollection>
void add_statefield_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << (FieldCollection::Global ? "Global" : "Local")
              << "StateField" << "_" << Dim;
  std::string name{name_stream.str()};
  using StateField_t = StateFieldBase<FieldCollection>;
  py::class_<StateField_t>(mod, name.c_str())
    .def_property_readonly("prefix", &StateField_t::get_prefix, "state field prefix")
    .def_property_readonly("collection", &StateField_t::get_collection,
                           "Collection containing this field")
    .def_property_readonly("nb_memory", &StateField_t::get_nb_memory,
                           "number of old states stored")
    .def_property_readonly("stored_type",
                           [](const StateField_t & field) {
                             return field.get_stored_typeid().name();
                           },
                           "fundamental type of scalars stored in this field");

  add_statefield<Real, FieldCollection>(mod, "Real");
  add_statefield<Int,  FieldCollection>(mod, "Int");
}

template <Dim_t Dim>
void add_field_collection_helper(py::module & mod) {
  add_field_helper<Dim, GlobalFieldCollection<Dim>>(mod);
  add_field_helper<Dim,  LocalFieldCollection<Dim>>(mod);

  add_field_collection<Dim, GlobalFieldCollection<Dim>>(mod);
  add_field_collection<Dim,  LocalFieldCollection<Dim>>(mod);
}

void add_field_collections (py::module & mod) {
  add_field_collection_helper<  twoD>(mod);
  add_field_collection_helper<threeD>(mod);
}
