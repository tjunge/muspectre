/**
 * file   solvers.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief  implementation of solver functions
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

#include <iomanip>
#include <cmath>

#include "solvers.hh"
#include "solver/solver_cg.hh"
#include "common/iterators.hh"

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  typename SystemBase<DimS, DimM>::StrainField_t &
  de_geus (SystemBase<DimS, DimM> & sys, const GradIncrements<DimM> & delFs,
           const Real cg_tol, const Real newton_tol, Uint maxiter,
           Dim_t verbose) {
    using Field_t = typename MaterialBase<DimS, DimM>::StrainField_t;
    auto solver_fields{std::make_unique<GlobalFieldCollection<DimS, DimM>>()};
    solver_fields->initialise(sys.get_resolutions());

    // Corresponds to symbol δF or δε
    auto & incrF{make_field<Field_t>("δF", *solver_fields)};

    // Corresponds to symbol ΔF or Δε
    auto & DeltaF{make_field<Field_t>("ΔF", *solver_fields)};

    // field to store the rhs for cg calculations
    auto & rhs{make_field<Field_t>("rhs", *solver_fields)};

    SolverCG<DimS, DimM> cg(sys.get_resolutions(),
                            cg_tol, maxiter, verbose-1>0);
    cg.initialise();


    if (maxiter == 0) {
      maxiter = sys.size()*DimM*DimM*10;
    }

    size_t count_width{};
    if (verbose > 0) {
      //setup of algorithm 5.2 in Nocedal, Numerical Optimization (p. 111)
      std::cout << "Algo 5.2 with newton_tol = " << newton_tol << ", cg_tol = "
                << cg_tol << " maxiter = " << maxiter << " and ΔF =" <<std::endl;
      for (auto&& tup: akantu::enumerate(delFs)) {
        auto && counter{std::get<0>(tup)};
        auto && grad{std::get<1>(tup)};
        std::cout << "Step " << counter + 1 << ":" << std::endl
                  << grad << std::endl;
      }
      count_width = size_t(std::log10(maxiter))+1;
    }

    // initialise F = I
    auto & F{sys.get_strain()};
    F.get_map() = Matrices::I2<DimM>();

    // initialise materials
    constexpr bool need_tangent{true};
    sys.initialise_materials(need_tangent);

    Grad_t<DimM> previous_grad{Grad_t<DimM>::Zero()};
    for (const auto & delF: delFs) { //incremental loop

      Real incrNorm{2*newton_tol}, gradNorm{1};
      for (Uint newt_iter{0};
           (newt_iter < maxiter) && ((incrNorm/gradNorm > newton_tol) ||
                                     (newt_iter==1));
           ++newt_iter) {

        // obtain material response
        auto res_tup{sys.evaluate_stress_tangent(F)};
        auto & P{std::get<0>(res_tup)};
        auto & K{std::get<1>(res_tup)};

        auto tangent_effect = [&sys, &K] (const Field_t & delF, Field_t & delP) {
          sys.directional_stiffness(K, delF, delP);
        };


        if (newt_iter == 0) {
          DeltaF.get_map() = -(delF-previous_grad); // neg sign because rhs
          tangent_effect(DeltaF, rhs);
          cg.solve(tangent_effect, rhs, incrF);
          F.eigen() -= DeltaF.eigen();
        } else {
          rhs.eigen() = -P.eigen();
          sys.project(rhs);
          cg.solve(tangent_effect, rhs, incrF);
        }

        F.eigen() += incrF.eigen();

        incrNorm = incrF.eigen().matrix().norm();
        gradNorm = F.eigen().matrix().norm();
        if (verbose>0) {
          std::cout << "at Newton step " << std::setw(count_width) << newt_iter
                    << ", |δF|/|ΔF| = " << std::setw(17) << incrNorm/gradNorm
                    << ", tol = " << newton_tol << std::endl;
          std::cout << "<F> =" << std::endl << F.get_map().mean() << std::endl;
        }
      }
      // update previous gradient
      previous_grad = delF;

      //store history variables here

    }

    return F;

  }

  template typename SystemBase<twoD, twoD>::StrainField_t &
  de_geus (SystemBase<twoD, twoD> & sys, const GradIncrements<twoD>& delF0,
           const Real cg_tol, const Real newton_tol, Uint maxiter,
           Dim_t verbose);

  // template typename SystemBase<twoD, threeD>::StrainField_t &
  // de_geus (SystemBase<twoD, threeD> & sys, const GradIncrements<threeD>& delF0,
  //            const Real cg_tol, const Real newton_tol, Uint maxiter,
  //            Dim_t verbose);

  template typename SystemBase<threeD, threeD>::StrainField_t &
  de_geus (SystemBase<threeD, threeD> & sys, const GradIncrements<threeD>& delF0,
           const Real cg_tol, const Real newton_tol, Uint maxiter,
           Dim_t verbose);

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename SystemBase<DimS, DimM>::StrainField_t &
  newton_cg (SystemBase<DimS, DimM> & sys, const GradIncrements<DimM> & delFs,
             const Real cg_tol, const Real newton_tol, Uint maxiter,
             Dim_t verbose) {
    using Field_t = typename MaterialBase<DimS, DimM>::StrainField_t;
    auto solver_fields{std::make_unique<GlobalFieldCollection<DimS, DimM>>()};
    solver_fields->initialise(sys.get_resolutions());

    // Corresponds to symbol δF or δε
    auto & incrF{make_field<Field_t>("δF", *solver_fields)};

    // field to store the rhs for cg calculations
    auto & rhs{make_field<Field_t>("rhs", *solver_fields)};

    SolverCG<DimS, DimM> cg(sys.get_resolutions(),
                            cg_tol, maxiter, verbose-1>0);
    cg.initialise();


    if (maxiter == 0) {
      maxiter = sys.size()*DimM*DimM*10;
    }

    size_t count_width{};
    if (verbose > 0) {
      //setup of algorithm 5.2 in Nocedal, Numerical Optimization (p. 111)
      std::cout << "Algo 5.2 with newton_tol = " << newton_tol << ", cg_tol = "
                << cg_tol << " maxiter = " << maxiter << " and ΔF =" <<std::endl;
      for (auto&& tup: akantu::enumerate(delFs)) {
        auto && counter{std::get<0>(tup)};
        auto && grad{std::get<1>(tup)};
        std::cout << "Step " << counter + 1 << ":" << std::endl
                  << grad << std::endl;
      }
      count_width = size_t(std::log10(maxiter))+1;
    }

    // initialise F = I
    auto & F{sys.get_strain()};
    F.get_map() = Matrices::I2<DimM>();

    // initialise materials
    constexpr bool need_tangent{true};
    sys.initialise_materials(need_tangent);

    Grad_t<DimM> previous_grad{Grad_t<DimM>::Zero()};
    for (const auto & delF: delFs) { //incremental loop
      // apply macroscopic strain increment
      for (auto && grad: F.get_map()) {
        grad += delF - previous_grad;
      }

      Real incrNorm{2*newton_tol}, gradNorm{1};
      for (Uint newt_iter{0};
           newt_iter < maxiter && incrNorm/gradNorm> newton_tol;
           ++newt_iter) {

        // obtain material response
        auto res_tup{sys.evaluate_stress_tangent(F)};
        auto & P{std::get<0>(res_tup)};
        auto & K{std::get<1>(res_tup)};

        auto fun = [&sys, &K] (const Field_t & delF, Field_t & delP) {
          sys.directional_stiffness(K, delF, delP);
        };

        rhs.eigen() = -P.eigen();
        sys.project(rhs);
        cg.solve(fun, rhs, incrF);

        F.eigen() += incrF.eigen();

        incrNorm = incrF.eigen().matrix().norm();
        gradNorm = F.eigen().matrix().norm();
        if (verbose > 0) {
          std::cout << "at Newton step " << std::setw(count_width) << newt_iter
                    << ", |δF|/|ΔF| = " << std::setw(17) << incrNorm/gradNorm
                    << ", tol = " << newton_tol << std::endl;
          std::cout << "<F> =" << std::endl << F.get_map().mean() << std::endl;
        }

      }
      // update previous gradient
      previous_grad = delF;

      //store history variables here

    }

    return F;

  }

  template typename SystemBase<twoD, twoD>::StrainField_t &
  newton_cg (SystemBase<twoD, twoD> & sys, const GradIncrements<twoD>& delF0,
             const Real cg_tol, const Real newton_tol, Uint maxiter,
             Dim_t verbose);

  // template typename SystemBase<twoD, threeD>::StrainField_t &
  // newton_cg (SystemBase<twoD, threeD> & sys, const GradIncrements<threeD>& delF0,
  //            const Real cg_tol, const Real newton_tol, Uint maxiter,
  //            Dim_t verbose);

  template typename SystemBase<threeD, threeD>::StrainField_t &
  newton_cg (SystemBase<threeD, threeD> & sys, const GradIncrements<threeD>& delF0,
             const Real cg_tol, const Real newton_tol, Uint maxiter,
             Dim_t verbose);


}  // muSpectre
