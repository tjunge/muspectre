/**
 * @file   header_test_t4_map.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Nov 2017
 *
 * @brief  Test the fourth-order map on second-order tensor implementation
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <boost/mpl/list.hpp>

#include "common/common.hh"
#include "tests.hh"
#include "test_goodies.hh"
#include "common/T4_map_proxy.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(T4map_tests);

  /**
   * Test fixture for construction of T4Map for the time being, symmetry is not
   * exploited
   */
  template<typename T, Dim_t Dim>
  struct T4_fixture
  {
    T4_fixture():matrix{}, tensor(matrix.data()){}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using M4 = T4Mat<T, Dim>;
    using T4 = T4MatMap<T, Dim>;
    constexpr static Dim_t dim{Dim};
    M4 matrix;
    T4 tensor;
  };

  using fix_collection = boost::mpl::list<T4_fixture<Real, twoD>,
                                          T4_fixture<Real, threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Simple_construction_test, F, fix_collection, F) {
    BOOST_CHECK_EQUAL(F::tensor.cols(), F::dim*F::dim);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(write_access_test, F, fix_collection, F) {
    auto & t4 = F::tensor;
    constexpr Dim_t dim{F::dim};
    Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim, dim, dim>> t4c;
    Eigen::Map<typename F::M4> t4c_map(t4c.data());
    for (Dim_t i = 0; i < F::dim; ++i) {
      for (Dim_t j = 0; j < F::dim; ++j) {
        for (Dim_t k = 0; k < F::dim; ++k) {
          for (Dim_t l = 0; l < F::dim; ++l) {
            get(t4,i,j,k,l) = 1000*(i+1) + 100*(j+1) + 10*(k+1) + l+1;
            t4c(i,j,k,l) = 1000*(i+1) + 100*(j+1) + 10*(k+1) + l+1;
          }
        }
      }
    }
    for (Dim_t i = 0; i < ipow(dim,4); ++i) {
      BOOST_CHECK_EQUAL(F::matrix.data()[i], t4c.data()[i]);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(numpy_compatibility, F, fix_collection, F) {
    auto & t4 = F::tensor;
    typename F::M4 numpy_ref{};
    if (F::dim == twoD) {
      numpy_ref <<
        1111., 1112., 1121., 1122.,
        1211., 1212., 1221., 1222.,
        2111., 2112., 2121., 2122.,
        2211., 2212., 2221., 2222.;
    } else {
      numpy_ref <<
        1111., 1112., 1113., 1121., 1122., 1123., 1131., 1132., 1133.,
        1211., 1212., 1213., 1221., 1222., 1223., 1231., 1232., 1233.,
        1311., 1312., 1313., 1321., 1322., 1323., 1331., 1332., 1333.,
        2111., 2112., 2113., 2121., 2122., 2123., 2131., 2132., 2133.,
        2211., 2212., 2213., 2221., 2222., 2223., 2231., 2232., 2233.,
        2311., 2312., 2313., 2321., 2322., 2323., 2331., 2332., 2333.,
        3111., 3112., 3113., 3121., 3122., 3123., 3131., 3132., 3133.,
        3211., 3212., 3213., 3221., 3222., 3223., 3231., 3232., 3233.,
        3311., 3312., 3313., 3321., 3322., 3323., 3331., 3332., 3333.;
    }
    for (Dim_t i = 0; i < F::dim; ++i) {
      for (Dim_t j = 0; j < F::dim; ++j) {
        for (Dim_t k = 0; k < F::dim; ++k) {
          for (Dim_t l = 0; l < F::dim; ++l) {
            get(t4,i,j,k,l) = 1000*(i+1) + 100*(j+1) + 10*(k+1) + l+1;
          }
        }
      }
    }

    Real error{(t4-testGoodies::from_numpy(numpy_ref)).norm()};
    BOOST_CHECK_EQUAL(error, 0);
    if (error != 0 ) {
      std::cout << "T4:" << std::endl << t4 << std::endl;
      std::cout << "reshuffled np:" << std::endl << testGoodies::from_numpy(numpy_ref) << std::endl;
      std::cout << "original np:" << std::endl << numpy_ref << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(numpy_right_trans, F, fix_collection, F) {
    auto & t4 = F::tensor;
    typename F::M4 numpy_ref{};
    if (F::dim == twoD) {
      numpy_ref <<
        1111., 1121., 1112., 1122.,
        1211., 1221., 1212., 1222.,
        2111., 2121., 2112., 2122.,
        2211., 2221., 2212., 2222.;
    } else {
      numpy_ref <<
        1111., 1121., 1131., 1112., 1122., 1132., 1113., 1123., 1133.,
        1211., 1221., 1231., 1212., 1222., 1232., 1213., 1223., 1233.,
        1311., 1321., 1331., 1312., 1322., 1332., 1313., 1323., 1333.,
        2111., 2121., 2131., 2112., 2122., 2132., 2113., 2123., 2133.,
        2211., 2221., 2231., 2212., 2222., 2232., 2213., 2223., 2233.,
        2311., 2321., 2331., 2312., 2322., 2332., 2313., 2323., 2333.,
        3111., 3121., 3131., 3112., 3122., 3132., 3113., 3123., 3133.,
        3211., 3221., 3231., 3212., 3222., 3232., 3213., 3223., 3233.,
        3311., 3321., 3331., 3312., 3322., 3332., 3313., 3323., 3333.;
    }
    for (Dim_t i = 0; i < F::dim; ++i) {
      for (Dim_t j = 0; j < F::dim; ++j) {
        for (Dim_t k = 0; k < F::dim; ++k) {
          for (Dim_t l = 0; l < F::dim; ++l) {
            get(t4,i,j,k,l) = 1000*(i+1) + 100*(j+1) + 10*(k+1) + l+1;
          }
        }
      }
    }

    Real error{(t4-testGoodies::right_transpose(numpy_ref)).norm()};
    BOOST_CHECK_EQUAL(error, 0);
    if (error != 0 ) {
      std::cout << "T4:" << std::endl << t4 << std::endl;
      std::cout << "reshuffled np:" << std::endl << testGoodies::from_numpy(numpy_ref) << std::endl;
      std::cout << "original np:" << std::endl << numpy_ref << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(assign_matrix_test, F, fix_collection, F) {
    decltype(F::matrix) matrix;
    matrix.setRandom();
    F::tensor = matrix;
    for (Dim_t i = 0; i < ipow(F::dim,4); ++i) {
      BOOST_CHECK_EQUAL(F::matrix.data()[i], matrix.data()[i]);
    }
  }

  BOOST_AUTO_TEST_CASE(Return_ref_from_const_test) {
    constexpr Dim_t dim{2};
    using T = int;
    using M4 = Eigen::Matrix<T, dim*dim, dim*dim>;
    using M4c = const Eigen::Matrix<T, dim*dim, dim*dim>;
    using T4 = T4MatMap<T, dim>;
    using T4c = T4MatMap<T, dim, true>;

    M4 mat;
    mat.setRandom();
    M4c cmat{mat};
    T4 tensor{mat.data()};
    T4c ctensor{mat.data()};

    T a = get(tensor,0,0,0,1);
    T b = get(ctensor,0,0,0,1);
    BOOST_CHECK_EQUAL(a, b);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre