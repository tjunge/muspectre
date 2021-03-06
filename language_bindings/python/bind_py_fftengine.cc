/**
 * @file   bind_py_fftengine.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   17 Jan 2018
 *
 * @brief  Python bindings for the FFT engines
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

#include "fft/fftw_engine.hh"
#ifdef WITH_FFTWMPI
#include "fft/fftwmpi_engine.hh"
#endif
#ifdef WITH_PFFT
#include "fft/pfft_engine.hh"
#endif
#include "bind_py_declarations.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using namespace muSpectre;
namespace py=pybind11;
using namespace pybind11::literals;

template <class Engine, Dim_t dim>
void add_engine_helper(py::module & mod, std::string name) {
  using Ccoord = Ccoord_t<dim>;
  using ArrayXXc = Eigen::Array<Complex, Eigen::Dynamic,
                                Eigen::Dynamic>;
  py::class_<Engine>(mod, name.c_str())
#ifdef WITH_MPI
    .def(py::init([](Ccoord res, Dim_t nb_components, size_t comm) {
          return new Engine(res, nb_components,
                            std::move(Communicator(MPI_Comm(comm))));
        }),
      "resolutions"_a,
      "nb_components"_a,
      "communicator"_a=size_t(MPI_COMM_SELF))
#else
    .def(py::init<Ccoord, Dim_t>())
#endif
    .def("fft",
         [](Engine & eng, py::EigenDRef<Eigen::ArrayXXd> v) {
           using Coll_t = typename Engine::GFieldCollection_t;
           using Field_t = typename Engine::Field_t;
           Coll_t coll{};
           coll.initialise(eng.get_subdomain_resolutions(),
                           eng.get_subdomain_locations());
           Field_t & temp{make_field<Field_t>("temp_field", coll,
                                              eng.get_nb_components())};
           temp.eigen() = v;
           return ArrayXXc{eng.fft(temp).eigen()};
         },
         "array"_a)
    .def("ifft",
         [](Engine & eng,
            py::EigenDRef<ArrayXXc> v) {
           using Coll_t = typename Engine::GFieldCollection_t;
           using Field_t = typename Engine::Field_t;
           Coll_t coll{};
           coll.initialise(eng.get_subdomain_resolutions(),
                           eng.get_subdomain_locations());
           Field_t & temp{make_field<Field_t>("temp_field", coll,
                                              eng.get_nb_components())};
           eng.get_work_space().eigen() = v;
           eng.ifft(temp);
           return Eigen::ArrayXXd{temp.eigen()};
         },
         "array"_a)
    .def("initialise", &Engine::initialise,
         "flags"_a=FFT_PlanFlags::estimate)
    .def("normalisation", &Engine::normalisation)
    .def("get_subdomain_resolutions", &Engine::get_subdomain_resolutions)
    .def("get_subdomain_locations", &Engine::get_subdomain_locations)
    .def("get_fourier_resolutions", &Engine::get_fourier_resolutions)
    .def("get_fourier_locations", &Engine::get_fourier_locations)
    .def("get_domain_resolutions", &Engine::get_domain_resolutions);
}

void add_fft_engines(py::module & mod) {
  auto fft{mod.def_submodule("fft")};
  fft.doc() = "bindings for µSpectre's fft engines";
  add_engine_helper<FFTWEngine<  twoD>,  twoD>(fft, "FFTW_2d");
  add_engine_helper<FFTWEngine<threeD>, threeD>(fft, "FFTW_3d");
#ifdef WITH_FFTWMPI
  add_engine_helper<FFTWMPIEngine<  twoD>,   twoD>(fft, "FFTWMPI_2d");
  add_engine_helper<FFTWMPIEngine<threeD>, threeD>(fft, "FFTWMPI_3d");
#endif
#ifdef WITH_PFFT
  add_engine_helper<PFFTEngine<  twoD>,   twoD>(fft, "PFFT_2d");
  add_engine_helper<PFFTEngine<threeD>, threeD>(fft, "PFFT_3d");
#endif
  add_projections(fft);
}
