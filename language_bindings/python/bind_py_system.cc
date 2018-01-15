/**
 * file   bind_py_system.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   09 Jan 2018
 *
 * @brief  Python bindings for the system factory function
 *
 * @section LICENCE
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
#include "common/ccoord_operations.hh"
#include "system/system_factory.hh"
#include "system/system_base.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <memory>

using namespace muSpectre;
namespace py=pybind11;
using namespace pybind11::literals;

/**
 * the system factory is only bound for default template parameters
 */
template <Dim_t dim>
void add_system_factory_helper(py::module & mod) {
  using Ccoord = Ccoord_t<dim>;
  using Rcoord = Rcoord_t<dim>;
  using Form = Formulation;

  mod.def
    ("SystemFactory",
     [](Ccoord res, Rcoord lens, Form form) {
      return make_system(std::move(res), std::move(lens), std::move(form));
     },
     "resolutions"_a,
     "lengths"_a=CcoordOps::get_cube<dim>(1.),
     "formulation"_a=Formulation::finite_strain);
}

void add_system_factory(py::module & mod) {
  add_system_factory_helper<twoD  >(mod);
  add_system_factory_helper<threeD>(mod);
}

/**
 * SystemBase for which the material and spatial dimension are identical
 */
template <Dim_t dim>
void add_system_base_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "SystemBase" << dim << 'd';
  const std::string name = name_stream.str();
  using sys_t = SystemBase<dim, dim>;
  py::class_<sys_t>(mod, name.c_str())
    .def("__len__", &sys_t::size)
    .def("__iter__", [](sys_t & s) {
        return py::make_iterator(s.begin(), s.end());
      })
    .def("initialise", &sys_t::initialise, "flags"_a=FFT_PlanFlags::estimate);
}

void add_system_base(py::module & mod) {
  add_system_base_helper<twoD>  (mod);
  add_system_base_helper<threeD>(mod);
}

void add_system(py::module & mod) {
  add_system_factory(mod);

  auto system{mod.def_submodule("system")};
  system.doc() = "bindings for systems and system factories";
  add_system_base(system);
}