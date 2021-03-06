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
#include "materials/material_linear_elastic1.hh"
#include "materials/material_linear_elastic2.hh"
#include "materials/material_linear_elastic3.hh"
#include "materials/material_linear_elastic4.hh"
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
                py::return_value_policy::reference, py::keep_alive<1, 0>())
    .def("add_pixel",
         [] (Mat_t & mat, Ccoord_t<dim> pix) {
           mat.add_pixel(pix);},
         "pixel"_a)
    .def("size", &Mat_t::size);
}

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
         "eigenstrain"_a)
    .def("size", &Mat_t::size);
}


template <Dim_t dim>
void add_material_linear_elastic3_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElastic3_" << dim << 'd';
  const auto name {name_stream.str()};

  using Mat_t = MaterialLinearElastic3<dim, dim>;
  using Sys_t = CellBase<dim, dim>;

  py::class_<Mat_t, MaterialBase<dim, dim>>(mod, name.c_str())
    .def(py::init<std::string>(), "name"_a)
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
	 "Poisson"_a)
    .def("size", &Mat_t::size);
}

template <Dim_t dim>
void add_material_linear_elastic4_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElastic4_" << dim << 'd';
  const auto name {name_stream.str()};

  using Mat_t = MaterialLinearElastic4<dim, dim>;
  using Sys_t = CellBase<dim, dim>;

  py::class_<Mat_t, MaterialBase<dim, dim>>(mod, name.c_str())
    .def(py::init<std::string>(), "name"_a)
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
	 "Poisson"_a)
    .def("size", &Mat_t::size);
}


template <Dim_t dim>
void add_material_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "Material_" << dim << 'd';
  const std::string name{name_stream.str()};
  using Mat_t = MaterialBase<dim, dim>;
  using FC_t = LocalFieldCollection<dim>;
  using FCBase_t = FieldCollectionBase<dim, FC_t>;

  py::class_<Mat_t>(mod, name.c_str()).
    def_property_readonly
    ("collection",
     [](Mat_t & material) -> FCBase_t &{
      return material.get_collection();},
     "returns the field collection containing internal "
     "fields of this material",
     py::return_value_policy::reference_internal);

  add_material_linear_elastic1_helper<dim>(mod);
  add_material_linear_elastic2_helper<dim>(mod);
  add_material_linear_elastic3_helper<dim>(mod);
  add_material_linear_elastic4_helper<dim>(mod);
}

void add_material(py::module & mod) {
  auto material{mod.def_submodule("material")};
  material.doc() = "bindings for constitutive laws";
  add_material_helper<twoD  >(material);
  add_material_helper<threeD>(material);
}
