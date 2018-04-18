/**
 * @file   geometry.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   18 Apr 2018
 *
 * @brief  Geometric calculation helpers
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
#include "common/tensor_algebra.hh"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>

namespace muSpectre {

  /**
   * The rotation matrices depend on the order in which we rotate
   * around different axes. See [[
   * https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix ]] to
   * find the matrices
   */
  class enum RotationOrder {Z,
      XZXEuler, XYXEuler, YXYEuler, YZYEuler, ZYZEuler, ZXZEuler,
      XZYTaitBryan, XYZTaitBryan, YXZTaitBryan, YZXTaitBryan, ZYXTaitBryan,
      ZXYTaitBryan};

  namespace internal {

    template <Dim_t Dim>
    struct DefaultOrder
    {
      constexpr static RotationOrder value{RotationOrder::ZXYTaitBryan};
    };

    template <>
    struct DefaultOrder<twoD>
    {
      constexpr static RotationOrder value{RotationOrder::Z};
    };

  }  // internal

  template <Dim_t Dim, RotationOrder Order=internal::DefaultOrder<Dim>::value>
  class Rotator
  {
  public:
    static_assert(((Dim == twoD) and (Order == RotationOrder::Z)) or
                  ((Dim == threeD) and (Order != RotationOrder::Z)),
                  "In 2d, only order 'Z' makes sense. In 3d, it doesn't");
    using Angles_t = Eigen::Matrix<Real, (Dim == TwoD) ? 1 :3, 1>;
    using RotMat_t = Eigen::Matrix<Real, Dim, Dim>;

    //! Default constructor
    Rotator() = delete;

    Rotator(Eigen::Ref<Angles_t> angles):
      angles{angles}, rot_mat{this->compute_rotation_matrix()} {}

    //! Copy constructor
    Rotator(const Rotator &other) = default;

    //! Move constructor
    Rotator(Rotator &&other) = default;

    //! Destructor
    virtual ~Rotator() = default;

    //! Copy assignment operator
    Rotator& operator=(const Rotator &other) = default;

    //! Move assignment operator
    Rotator& operator=(Rotator &&other) = default;

    /**
     * Applies the rotation @param input is a first-, second-, or
     * fourth-rank tensor (column vector, square matrix, or T4Matrix,
     * or a Eigen::Map of either of these, or an expression that
     * evaluates into any of these)
     */
    template<class In_t>
    inline decltype(auto) rotate_into(In_t && input);

    template<class In_t>
    inline decltype(auto) rotate_outof(In_t && input);

  protected:

    inline RotMat_t compute_rotation_matrix();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Angles_t angles;
    RotMat_t rot_mat;
  private:
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, RotationOrder Order>
  auto Rotator<Dim, Order>::compute_rotation_matrix() -> RotMat_t {
    if (Dim == twoD) {
      return Eigen::Rotation2Dd(this->angles(0));
    } else {
      switch (Order) {
      case RotationOrder::ZXZEuler: {
        return (Eigen::AngleAxisd(angles(0), Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(angles(1), Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(angles(2), Eigen::Vector3d::UnitZ()));
        break;
      }
      }
    }
  }

  namespace internal {

    /* ---------------------------------------------------------------------- */
    template <class In_t, Dim_t Dim>
    inline decltype(auto) rotate_T1(In_t && input,
                                    const Eigen::Matrix<Real, Dim, Dim> & R) {
      return R*std::forward<In_t>(input);
    }

    /* ---------------------------------------------------------------------- */
    template <class In_t, Dim_t Dim>
    inline decltype(auto) rotate_T2(In_t && input,
                                    const Eigen::Matrix<Real, Dim, Dim> & R) {
      return R*std::forward<In_t>(input)*R.transpose();
    }

    /* ---------------------------------------------------------------------- */
    template <class In_t, Dim_t Dim>
    inline decltype(auto) rotate_T4(In_t && input,
                                    const Eigen::Matrix<Real, Dim, Dim> & R) {
      auto && rotator_forward = Matrices::outer_under(R, R);
      auto && rotator_back = Matrices::outer_under(R.transpose(), R.transpose());
      return rotator_back * std::forward<In_t>(intput) * rotator_forward;
    }

  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim, RotationOrder Order>
  template <class In_t>
  auto Rotator<Dim, Order>::rotate_into(In_t && input) decltype(auto) {
    
  }

}  // muSpectre
