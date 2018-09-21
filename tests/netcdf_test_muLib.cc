/**
 * @file   netcdf_test_muLib.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   20 Sep 2018
 *
 * @brief  test the mulib input reader
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


#include <io/mulib_input.hh>
#include "solver/mulib_solver.hh"

#include "tests.hh"



namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(mulib_input);

  BOOST_AUTO_TEST_CASE(read_input) {

    const filesystem::path nc_path{"/tmp/testBWimage.nc"};

    MuLibInput input(nc_path);
  }

  BOOST_AUTO_TEST_CASE(setup_cell) {
    mulib("/tmp/testBWimage.nc", 1e-6, 1e-6, 1e-6, 100, 1);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
