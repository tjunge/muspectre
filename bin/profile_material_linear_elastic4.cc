/**
 * @file   profile_material_linear_elastic4.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   19 Apr 2018
 *
 * @brief  program to profile material_linear_elastic4 for possible bottlenecks
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

#include "cell/cell_factory.hh"
#include "materials/material_linear_elastic4.hh"
#include "solver/solvers.hh"
#include "solver/solver_cg.hh"

#include <iostream>
#include <random>

using namespace muSpectre;

int main()
{
  std::cout << "startig program\n";

  const Ccoord_t<threeD> resolution{25, 25, 25};
  const Rcoord_t<threeD> lengths{5.0, 5.0, 5.0};
  constexpr Formulation formulation{Formulation::finite_strain};

  auto cell{make_cell<threeD, threeD>(resolution, lengths, formulation)};

  //initialize my material
  using Material_t = MaterialLinearElastic4<threeD, threeD>;
  auto mat{std::make_unique<Material_t>("material")};

  //introduce random number generators for Young and Poisson
  //make random numbers as in std::uniform_real_distribution cppreference.com
  std::random_device rd;
  std::mt19937 gen(rd());
  //set a seed for the random number generation to make reproducable results
  gen.seed(15);
  // for Young [5,10)
  std::uniform_real_distribution<> dis_Y(5, 10);
  //for Poisson [0.1, 0.4)
  std::uniform_real_distribution<> dis_P(0.1, 0.4);

  //fill the pixels
  for (const auto && pixel:cell) {
    mat->add_pixel(pixel, dis_Y(gen), dis_P(gen));
  }

  cell.add_material(std::move(mat));       //add the material to the cell
  cell.initialise(FFT_PlanFlags::measure); //fft initialization, make faster fft

  //set the deformation
  //DelF = np.array([[0, 0.01, 0],
  //                 [0, 0   , 0],
  //                 [0, 0   , 0]])
  Grad_t<threeD> DelF{Grad_t<threeD>::Zero()};
  DelF(0, 1) = 0.01;

  //set solver constants
  const Real newton_tol {1e-6}; //tolerance for newton algo
  const Real cg_tol     {1e-6}; //tolerance for cg algo
  const Real equil_tol  {1e-6}; //tolerance for equilibrium / tol for div(P) ?
  const Uint maxiter    {100};  //maximum cg iterations
  const Uint verbose    {0};    //verbosity of solver

  GradIncrements<threeD> grads{DelF};
  SolverCG<threeD> cg{cell, cg_tol, maxiter, bool(verbose)};
  auto newton_result {newton_cg(cell, grads, cg,
				newton_tol, equil_tol, verbose)[0]};

  //print some messages of the solver

  //std::cout << "gradient F:\n" << newton_result.grad << "\n";
  //std::cout << "\n\nstress:\n" << newton_result.stress << "\n";
  std::cout << "convergence_test: " << newton_result.success << "\n";
  std::cout << "status: " << newton_result.status << "\n";
  std::cout << "message: " << newton_result.message << "\n";
  std::cout << "# iterations: " << newton_result.nb_it << "\n";
  std::cout << "# cell evaluations: " << newton_result.nb_fev << "\n";

  std::cout << "program finished correct\n";
}
