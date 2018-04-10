/**
 * @file   test_fields.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   20 Sep 2017
 *
 * @brief  Test Fields that are used in FieldCollections
 *
 * Copyright © 2017 Till Junge
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

#include "tests.hh"
#include "common/field_collection.hh"
#include "common/field.hh"
#include "common/ccoord_operations.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(field_test);

  BOOST_AUTO_TEST_CASE(simple_creation) {
    constexpr Dim_t sdim{twoD};
    constexpr Dim_t mdim{twoD};
    constexpr Dim_t order{fourthOrder};
    using FC_t = GlobalFieldCollection<sdim>;
    FC_t fc;

    using TF_t = TensorField<FC_t, Real, order, mdim>;
    auto & field{make_field<TF_t>("TensorField 1", fc)};

    // check that fields are initialised with empty vector
    BOOST_CHECK_EQUAL(field.size(), 0);
    Dim_t len{2};
    fc.initialise(CcoordOps::get_cube<sdim>(len), {});
    // check that returned size is correct
    BOOST_CHECK_EQUAL(field.size(), ipow(len, sdim));
    // check that setting pad size won't change logical size
    field.set_pad_size(24);
    BOOST_CHECK_EQUAL(field.size(), ipow(len, sdim));
  }

  BOOST_AUTO_TEST_CASE(dynamic_field_creation) {
    constexpr Dim_t sdim{threeD};
    Dim_t nb_components{2};

    using FC_t = GlobalFieldCollection<sdim>;
    FC_t fc{};
    make_field<TypedField<FC_t, Real>>("Dynamic Field", fc, nb_components);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
