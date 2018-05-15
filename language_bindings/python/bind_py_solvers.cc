/**
 * @file   bind_py_solver.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   09 Jan 2018
 *
 * @brief  python bindings for the muSpectre solvers
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
#include "solver/new_solvers.hh"
#include "solver/new_solver_cg.hh"
#include "solver/new_solver_eigen.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using namespace muSpectre;
namespace py=pybind11;
using namespace pybind11::literals;

/**
 * Solvers instanciated for cells with equal spatial and material dimension
 */

template <class Solver>
void add_iterative_solver_helper(py::module & mod, std::string name) {
  py::class_<Solver, typename Solver::Parent>(mod, name.c_str())
    .def(py::init<Cell&, Real, Uint, bool>(),
         "cell"_a,
         "tol"_a,
         "maxiter"_a,
         "verbose"_a=false)
    .def("name", &Solver::get_name);
  // mod.def(name.c_str(),
  //         [](Cell& cell, Real tol, Uint maxiter, bool verbose) {
  //           return std::make_unique<Solver>(cell, tol, maxiter, verbose);
  //         },
  //         "cell"_a,
  //         "tol"_a,
  //         "maxiter"_a,
  //         "verbose"_a=false);
}

void add_iterative_solver(py::module & mod) {
  std::stringstream name{};
  name << "SolverBase";
  py::class_<SolverBaseDyn>(mod, name.str().c_str());
  add_iterative_solver_helper<SolverCGDyn>(mod, "SolverCG");
  add_iterative_solver_helper<SolverCGEigenDyn>(mod, "SolverCGEigen");
  add_iterative_solver_helper<SolverGMRESEigenDyn>(mod, "SolverGMRESEigen");
  add_iterative_solver_helper<SolverBiCGSTABEigenDyn>(mod, "SolverBiCGSTABEigen");
  add_iterative_solver_helper<SolverDGMRESEigenDyn>(mod, "SolverDGMRESEigen");
  add_iterative_solver_helper<SolverMINRESEigenDyn>(mod, "SolverMINRESEigen");
}

void add_newton_cg_helper(py::module & mod) {

  const char name []{"newton_cg"};
  using solver = SolverBaseDyn;
  using grad = py::EigenDRef<Eigen::MatrixXd>;
  using grad_vec = LoadSteps_t;

  mod.def(name,
          [](Cell & s, const grad & g, solver & so, Real nt,
             Real eqt, Dim_t verb) -> OptimizeResult {
            Eigen::MatrixXd tmp{g};
            return newton_cg_dyn(s, tmp, so, nt, eqt, verb);

          },
          "cell"_a,
          "ΔF₀"_a,
          "solver"_a,
          "newton_tol"_a,
          "equil_tol"_a,
          "verbose"_a=0);
  mod.def(name,
          [](Cell & s, const grad_vec & g, solver & so, Real nt,
             Real eqt, Dim_t verb) -> std::vector<OptimizeResult> {
            return newton_cg_dyn(s, g, so, nt, eqt, verb);
          },
          "cell"_a,
          "ΔF₀"_a,
          "solver"_a,
          "newton_tol"_a,
          "equilibrium_tol"_a,
          "verbose"_a=0);
}

void add_de_geus_helper(py::module & mod) {
  const char name []{"de_geus"};
  using solver = SolverBaseDyn;
  using grad = py::EigenDRef<Eigen::MatrixXd>;
  using grad_vec = LoadSteps_t;

  mod.def(name,
          [](Cell & s, const grad & g, solver & so, Real nt,
             Real eqt, Dim_t verb) -> OptimizeResult {
            Eigen::MatrixXd tmp{g};
            return de_geus_dyn(s, tmp, so, nt, eqt, verb);

          },
          "cell"_a,
          "ΔF₀"_a,
          "solver"_a,
          "newton_tol"_a,
          "equilibrium_tol"_a,
          "verbose"_a=0);
  mod.def(name,
          [](Cell & s, const grad_vec & g, solver & so, Real nt,
             Real eqt, Dim_t verb) -> std::vector<OptimizeResult> {
            return de_geus_dyn(s, g, so, nt, eqt, verb);
          },
          "cell"_a,
          "ΔF₀"_a,
          "solver"_a,
          "newton_tol"_a,
          "equilibrium_tol"_a,
          "verbose"_a=0);
}

void add_solver_helper(py::module & mod) {
  add_newton_cg_helper(mod);
  add_de_geus_helper  (mod);
}

void add_solvers(py::module & mod) {
  auto solvers{mod.def_submodule("solvers")};
  solvers.doc() = "bindings for solvers";

  py::class_<OptimizeResult>(mod, "OptimizeResult")
    .def_readwrite("grad", &OptimizeResult::grad)
    .def_readwrite("stress", &OptimizeResult::stress)
    .def_readwrite("success", &OptimizeResult::success)
    .def_readwrite("status", &OptimizeResult::status)
    .def_readwrite("message", &OptimizeResult::message)
    .def_readwrite("nb_it", &OptimizeResult::nb_it)
    .def_readwrite("nb_fev", &OptimizeResult::nb_fev);

  add_iterative_solver(solvers);

  add_solver_helper(solvers);
}
