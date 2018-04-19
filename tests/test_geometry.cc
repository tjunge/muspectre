/**
 * file   test_geometry.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   19 Apr 2018
 *
 * @brief  Tests for tensor rotations
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

#include "common/geometry.hh"
#include "tests.hh"
#include "test_goodies.hh"
#include "common/T4_map_proxy.hh"

#include <Eigen/Dense>

#include <cmath>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(geometry);

  BOOST_AUTO_TEST_CASE(twoD_rotation_test) {
    using Vec_t = Eigen::Vector2d;
    using Mat_t = Eigen::Matrix2d;
    using Ten_t = T4Mat<Real, twoD>;

    testGoodies::RandRange<Real> rr{};
    Real angle{rr.randval(0, 2 * pi)};
    Mat_t R{}; R <<
      std::cos(angle), -std::sin(angle),
      std::sin(angle),  std::cos(angle);

    Vec_t v{}; v.setRandom();
    Mat_t m{}; m.setRandom();
    Ten_t t{}; t.setRandom();

    Vec_t v_ref{R * v};
    Mat_t m_ref{R * m * R.transpose()};
    Ten_t t_ref{}; t_ref.setZero();
    for (int i = 0; i < twoD; ++i) {
      for (int a = 0; a < twoD; ++a) {
        for (int l = 0; l < twoD; ++l) {
          for (int b = 0; b < twoD; ++b) {
            for (int m = 0; m < twoD; ++m) {
              for (int n = 0; n < twoD; ++n) {
                for (int o = 0; o < twoD; ++o) {
                  for (int p = 0; p < twoD; ++p) {
                    get(t_ref, a, b, o, p) =
                      R(a, i) * R(b, l) * get(t, i, l, m, n) * R(o, m) * R(n, p);
                  }
                }
              }
            }
          }
        }
      }
    }

    Eigen::Matrix<Real, 1, 1> angle_vec{}; angle_vec << angle;
    Rotator<twoD> rotator(angle_vec);
    Vec_t v_rotator(rotator.rotate_into(v));
    Mat_t m_rotator(rotator.rotate_into(m));
    Ten_t t_rotator(rotator.rotate_into(t));
  }



  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
