/**
 * file   new_solver_eigen.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 May 2018
 *
 * @brief  Bindings to Eigen's iterative solvers
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

#ifndef NEW_SOLVER_EIGEN_H
#define NEW_SOLVER_EIGEN_H

#include "solver/new_solver_base.hh"
#include "cell/cell_base.hh"

#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace muSpectre {

  template <class SolverType>
  class SolverEigenDyn;

  class SolverCGEigenDyn;

  class SolverGMRESEigenDyn;

  class SolverBiCGSTABEigenDyn;

  class SolverDGMRESEigenDyn;

  class SolverMINRESEigenDyn;

  namespace internal {

    template <class Solver>
    struct SolverDyn_traits {
    };

    //! traits for the Eigen conjugate gradient solver
    template<>
    struct SolverDyn_traits<SolverCGEigenDyn> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::ConjugateGradient<typename Cell::Adaptor,
                                 Eigen::Lower|Eigen::Upper,
                                 Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen GMRES solver
    template<>
    struct SolverDyn_traits<SolverGMRESEigenDyn> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::GMRES<typename Cell::Adaptor,
                     Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen BiCGSTAB solver
    template<>
    struct SolverDyn_traits<SolverBiCGSTABEigenDyn> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::BiCGSTAB<typename Cell::Adaptor,
                        Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen DGMRES solver
    template<>
    struct SolverDyn_traits<SolverDGMRESEigenDyn> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::DGMRES<typename Cell::Adaptor,
                      Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen MINRES solver
    template<>
    struct SolverDyn_traits<SolverMINRESEigenDyn> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::MINRES<typename Cell::Adaptor,
                      Eigen::Lower|Eigen::Upper,
                      Eigen::IdentityPreconditioner>;
    };

  }  // internal

  /**
   * base class for iterative solvers from Eigen
   */
  template <class SolverType>
  class SolverEigenDyn: public SolverBaseDyn
  {
  public:
    using Parent = SolverBaseDyn; //!< base class
    //! traits obtained from CRTP
    using Solver = typename internal::SolverDyn_traits<SolverType>::Solver;
    //! Input vectors for solver
    using ConstVector_ref = Parent::ConstVector_ref;
    //! Output vector for solver
    using Vector_map = Parent::Vector_map;
    //! storage for output vector
    using Vector_t = Parent::Vector_t;

    //! Default constructor
    SolverEigenDyn() = delete;

    //! Constructor with domain resolutions, etc,
    SolverEigenDyn(Cell& cell, Real tol, Uint maxiter=0, bool verbose =false);

    //! Copy constructor
    SolverEigenDyn(const SolverEigenDyn &other) = delete;

    //! Move constructor
    SolverEigenDyn(SolverEigenDyn &&other) = default;

    //! Destructor
    virtual ~SolverEigenDyn() = default;

    //! Copy assignment operator
    SolverEigenDyn& operator=(const SolverEigenDyn &other) = delete;

    //! Move assignment operator
    SolverEigenDyn& operator=(SolverEigenDyn &&other) = default;

    //! Allocate fields used during the solution
    void initialise() override final;

    //! executes the solver
    Vector_map solve(const ConstVector_ref rhs) override final;


  protected:
    Cell::Adaptor adaptor; //!< cell handle
    Solver solver; //!< Eigen's Iterative solver
    Vector_t result; //!< storage for result
  };

  /**
   * Binding to Eigen's conjugate gradient solver
   */
  class SolverCGEigenDyn:
    public SolverEigenDyn<SolverCGEigenDyn> {
  public:
    using SolverEigenDyn<SolverCGEigenDyn>::SolverEigenDyn;
    std::string get_name() const override final {return "CG";}
  };

  /**
   * Binding to Eigen's GMRES solver
   */
  class SolverGMRESEigenDyn:
    public SolverEigenDyn<SolverGMRESEigenDyn> {
  public:
    using SolverEigenDyn<SolverGMRESEigenDyn>::SolverEigenDyn;
    std::string get_name() const override final {return "GMRES";}
  };

  /**
   * Binding to Eigen's BiCGSTAB solver
   */
  class SolverBiCGSTABEigenDyn:
    public SolverEigenDyn<SolverBiCGSTABEigenDyn> {
  public:
    using SolverEigenDyn<SolverBiCGSTABEigenDyn>::SolverEigenDyn;
    //! Solver's name
    std::string get_name() const override final {return "BiCGSTAB";}
  };

  /**
   * Binding to Eigen's DGMRES solver
   */
  class SolverDGMRESEigenDyn:
    public SolverEigenDyn<SolverDGMRESEigenDyn> {
  public:
    using SolverEigenDyn<SolverDGMRESEigenDyn>::SolverEigenDyn;
    //! Solver's name
    std::string get_name() const override final {return "DGMRES";}
  };

  /**
   * Binding to Eigen's MINRES solver
   */
  class SolverMINRESEigenDyn:
    public SolverEigenDyn<SolverMINRESEigenDyn> {
  public:
    using SolverEigenDyn<SolverMINRESEigenDyn>::SolverEigenDyn;
    //! Solver's name
    std::string get_name() const override final {return "MINRES";}
  };

}  // muSpectre

#endif /* NEW_SOLVER_EIGEN_H */
