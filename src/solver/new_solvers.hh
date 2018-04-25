/**
 * file   new_solvers.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Apr 2018
 *
 * @brief  Free functions for solving rve problems
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


#ifndef NEW_SOLVERS_H
#define NEW_SOLVERS_H

#include "solvers.hh"
#include "solver/new_solver_base.hh"

#include <Eigen/Dense>

#include <vector>
#include <string>

namespace muSpectre {

  std::vector<OptimizeResult>
  newton_cg_dyn(Cell & cell,
                const std::vector<Eigen::MatrixXd> load_increments,
                SolverBaseDyn & solver, Real newton_tol,
                Real equil_tol,
                Dim_t verbose = 0);

}  // muSpectre

#endif /* NEW_SOLVERS_H */
