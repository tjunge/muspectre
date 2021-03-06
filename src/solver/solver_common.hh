/**
 * @file   solver_common.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   28 Dec 2017
 *
 * @brief  Errors raised by solvers and other common utilities
 *
 * Copyright © 2017 Till Junge
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

#ifndef SOLVER_COMMON_H
#define SOLVER_COMMON_H

#include "common/common.hh"
#include "common/tensor_algebra.hh"

#include <Eigen/Dense>

#include <stdexcept>

namespace muSpectre {

  /**
   * emulates scipy.optimize.OptimizeResult
   */
  struct OptimizeResult
  {
    //! Strain ε or Gradient F at solution
    Eigen::ArrayXXd grad;
    //! Cauchy stress σ or first Piola-Kirchhoff stress P at solution
    Eigen::ArrayXXd stress;
    //! whether or not the solver exited successfully
    bool success;
    //! Termination status of the optimizer. Its value depends on the
    //! underlying solver. Refer to message for details.
    Int status;
    //! Description of the cause of the termination.
    std::string message;
    //! number of iterations
    Uint nb_it;
    //! number of cell evaluations
    Uint nb_fev;
  };

  /**
   * Field type that solvers expect gradients to be expressed in
   */
  template <Dim_t Dim>
  using Grad_t = Matrices::Tens2_t<Dim>;
  /**
   * multiple increments can be submitted at once (useful for
   * path-dependent materials)
   */
  template <Dim_t Dim>
  using GradIncrements = std::vector<Grad_t<Dim>,
                                     Eigen::aligned_allocator<Grad_t<Dim>>>;


  /* ---------------------------------------------------------------------- */
  class SolverError: public std::runtime_error {
    using runtime_error::runtime_error;
  };

  /* ---------------------------------------------------------------------- */
  class ConvergenceError: public SolverError {
    using SolverError::SolverError;
  };

  /* ---------------------------------------------------------------------- */
  /**
   * check whether a strain is symmetric, for the purposes of small
   * strain problems
   */
  bool check_symmetry(const Eigen::Ref<const Eigen::ArrayXXd>& eps,
                      Real rel_tol = 1e-8);

}  // muSpectre


#endif /* SOLVER_COMMON_H */
