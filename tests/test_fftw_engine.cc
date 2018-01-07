/**
 * file   test_fftw_engine.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Dec 2017
 *
 * @brief  tests for the fftw fft engine implementation
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include <boost/mpl/list.hpp>

#include "tests.hh"
#include "fft/fftw_engine.hh"
#include "common/ccoord_operations.hh"
#include "common/field_collection.hh"
#include "common/field_map.hh"
#include "common/iterators.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(fftw_engine);

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  struct FFTW_fixture {
    constexpr static Dim_t box_resolution{3};
    constexpr static Real box_length{4.5};
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};
    FFTW_fixture() :engine(CcoordOps::get_cube<DimS>(box_resolution),
                           CcoordOps::get_cube<DimS>(box_length)){}
    FFTW_Engine<DimS, DimM> engine;
  };

  using fixlist = boost::mpl::list<FFTW_fixture<twoD, twoD>,
                                   FFTW_fixture<twoD, threeD>,
                                   FFTW_fixture<threeD, threeD>>;


  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Constructor_test, Fix, fixlist, Fix) {
    BOOST_CHECK_NO_THROW(Fix::engine.initialise(FFT_PlanFlags::estimate));
    BOOST_CHECK_EQUAL(Fix::engine.size(), ipow(Fix::box_resolution, Fix::sdim));
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(fft_test, Fix, fixlist, Fix) {
    Fix::engine.initialise(FFT_PlanFlags::estimate);
    constexpr Dim_t order{2};
    using FC_t = FieldCollection<Fix::sdim, Fix::mdim>;
    FC_t fc;
    auto & input{make_field<TensorField<FC_t, Real, order, Fix::mdim>>("input", fc)};
    auto & ref  {make_field<TensorField<FC_t, Real, order, Fix::mdim>>("reference", fc)};
    auto & result{make_field<TensorField<FC_t, Real, order, Fix::mdim>>("result", fc)};
    fc.initialise(CcoordOps::get_cube<Fix::sdim>(Fix::box_resolution));

    using map_t = MatrixFieldMap<FC_t, Real, Fix::mdim, Fix::mdim>;
    map_t inmap{input};
    auto refmap{map_t{ref}};
    auto resultmap{map_t{result}};
    size_t cntr{0};
    for (auto tup: akantu::zip(inmap, refmap)) {
      cntr++;
      auto & in_{std::get<0>(tup)};
      auto & ref_{std::get<1>(tup)};
      in_.setRandom();
      ref_ = in_;
    }
    auto & complex_field = Fix::engine.fft(input);
    using cmap_t = MatrixFieldMap<FieldCollection<Fix::sdim, Fix::mdim, false>, Complex, Fix::mdim, Fix::mdim>;
    cmap_t complex_map(complex_field);
    Real error = complex_map[0].imag().norm();
    BOOST_CHECK_LT(error, tol);

    /* make sure, the engine has not modified input (which is
       unfortunately const-casted internally, hence this test) */
    for (auto && tup: akantu::zip(inmap, refmap)) {
      Real error{(std::get<0>(tup) - std::get<1>(tup)).norm()};
      BOOST_CHECK_LT(error, tol);
    }

    /* make sure that the ifft of fft returns the original*/
    Fix::engine.ifft(result);
    for (auto && tup: akantu::zip(resultmap, refmap)) {
      Real error{(std::get<0>(tup)*Fix::engine.normalisation() - std::get<1>(tup)).norm()};
      BOOST_CHECK_LT(error, tol);
      if (error > tol) {
        std::cout << std::get<0>(tup).array()/std::get<1>(tup).array() << std::endl << std::endl;
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre