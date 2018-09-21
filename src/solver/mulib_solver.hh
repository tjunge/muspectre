/**
 * @file   muLib_solver.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   21 Sep 2018
 *
 * @brief  Solver taking a mulib input file and returning homogenised properties
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

#ifndef MULIB_SOLVER_H
#define MULIB_SOLVER_H

#include "io/mulib_input.hh"

namespace muSpectre {

  void mulib(const filesystem::path & path,
             Real newton_tol,
             Real equil_tol,
             Real cg_tol,
             Uint maxiter,
             Dim_t verbose = 0);



}  // muSpectre

#endif /* MULIB_SOLVER_H */
