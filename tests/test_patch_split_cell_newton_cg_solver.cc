/**
 * @file   test_solver_newton_cg.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief  Tests for the standard Newton-Raphson + Conjugate Gradient solver
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
#include "solver/solvers.hh"
#include "solver/solver_cg.hh"
#include "solver/solver_cg_eigen.hh"
#include "fft/fftw_engine.hh"
#include "fft/projection_finite_strain_fast.hh"
#include "materials/material_linear_elastic1.hh"
#include "common/iterators.hh"
#include "common/ccoord_operations.hh"
#include "cell/cell_factory.hh"
#include "common/common.hh"
#include "cell/cell_split.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(split_cell_newton_cg_tests);

  BOOST_AUTO_TEST_CASE(manual_construction_test) {

    // constexpr Dim_t dim{twoD};
    constexpr Dim_t dim{threeD};
    
    using Mat_t = MaterialLinearElastic1<dim, dim>;
    //const Real Young{210e9}, Poisson{.33};
    const Real contrast {10};
    const Real Young_soft{1.0030648180242636}, Poisson_soft{0.29930675909878679};
    const Real Young_hard{contrast * Young_soft}, Poisson_hard{0.29930675909878679};
    const Real Young_mix{(Young_soft+Young_hard)/2}, Poisson_mix{0.29930675909878679};
    // const Real lambda{Young*Poisson/((1+Poisson)*(1-2*Poisson))};
    // const Real mu{Young/(2*(1+Poisson))};


    // constexpr Ccoord_t<dim> resolutions{3, 3};
    // constexpr Rcoord_t<dim> lengths{2.3, 2.7};
    constexpr Ccoord_t<dim> resolutions_split{3, 3, 3};
    constexpr Rcoord_t<dim> lengths_split{3, 3, 3};
    auto fft_ptr_split{std::make_unique<FFTWEngine<dim, dim>>(resolutions_split)};
    auto proj_ptr_split{std::make_unique<ProjectionFiniteStrainFast<dim, dim>>(std::move(fft_ptr_split), lengths_split)};
    CellSplit<dim, dim> sys_split(std::move(proj_ptr_split), SplittedCell::yes);

    constexpr Ccoord_t<dim> resolutions_base{3, 3, 3};
    constexpr Rcoord_t<dim> lengths_base{3, 3, 3};
    auto fft_ptr_base{std::make_unique<FFTWEngine<dim, dim>>(resolutions_base)};
    auto proj_ptr_base{std::make_unique<ProjectionFiniteStrainFast<dim, dim>>(std::move(fft_ptr_base), lengths_base)};
    CellBase<dim, dim> sys_base(std::move(proj_ptr_base));


    auto& Material_hard_split = Mat_t::make(sys_split, "hard", Young_hard, Poisson_hard);
    auto& Material_soft_split = Mat_t::make(sys_split, "soft", Young_soft, Poisson_soft);

    for (auto && tup: akantu::enumerate(sys_split)) {
      auto && pixel = std::get<1>(tup);
      if (pixel[0] < 2) {
        Material_hard_split.add_pixel_split(pixel,1);
      }else {
        Material_hard_split.add_pixel_split(pixel,0.5);
        Material_soft_split.add_pixel_split(pixel,0.5);
      }
    }
    sys_split.initialise();




    auto& Material_hard_base = Mat_t::make(sys_base, "hard", Young_hard, Poisson_hard);
    auto& Material_mix_base =  Mat_t::make(sys_base, "mix",   Young_mix, Poisson_mix);

    for (auto && tup: akantu::enumerate(sys_base)) {
      auto && pixel = std::get<1>(tup);
      if (pixel[0] < 2) {
        Material_hard_base.add_pixel(pixel);
      } else {
        Material_mix_base.add_pixel(pixel);
      }
    }
    sys_base.initialise();

    Grad_t<dim> delF0;
    delF0 << 0, 1, 0, 0, 0, 0, 0, 0, 0;
    constexpr Real cg_tol{1e-8}, newton_tol{1e-5};
    constexpr Uint maxiter{CcoordOps::get_size(resolutions_split)*ipow(dim, secondOrder)*10};
    constexpr bool verbose{false};

    GradIncrements<dim> grads; grads.push_back(delF0);
    
    SolverCG<dim> cg2{sys_base, cg_tol, maxiter, bool(verbose)};
    Eigen::ArrayXXd res2{newton_cg(sys_base, grads, cg2, newton_tol, verbose)[0].grad};

    SolverCG<dim> cg1{sys_split, cg_tol, maxiter, bool(verbose)};
    Eigen::ArrayXXd res1{newton_cg(sys_split, grads, cg1, newton_tol, verbose)[0].grad};

    BOOST_CHECK_LE(abs(res1-res1).mean(), cg_tol);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
