/**
 * file   solver_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   18 Dec 2017
 *
 * @brief  Base class for solvers
 *
 * @section LICENSE
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

#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include "solver/solver_error.hh"
#include "common/common.hh"
#include "system/system_base.hh"
#include "common/tensor_algebra.hh"

#include <Eigen/Dense>

#include <vector>

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM=DimS>
  class SolverBase
  {
  public:
    enum class TangentRequirement{NoNeed, NeedEffect, NeedTangents};
    using Sys_t = SystemBase<DimS, DimM>;
    using Ccoord = Ccoord_t<DimS>;
    using Collection_t = GlobalFieldCollection<DimS, DimM>;
    using SolvVectorIn = Eigen::Ref<Eigen::VectorXd>;
    using SolvVectorInC = Eigen::Ref<const Eigen::VectorXd>;
    using SolvVectorOut = Eigen::VectorXd;


    //! Default constructor
    SolverBase() = delete;

    //! Constructor with domain resolutions
    SolverBase(Sys_t & sys, Real tol, Uint maxiter=0, bool verbose =false);

    //! Copy constructor
    SolverBase(const SolverBase &other) = delete;

    //! Move constructor
    SolverBase(SolverBase &&other) = default;

    //! Destructor
    virtual ~SolverBase() = default;

    //! Copy assignment operator
    SolverBase& operator=(const SolverBase &other) = delete;

    //! Move assignment operator
    SolverBase& operator=(SolverBase &&other) = default;

    //! Allocate fields used during the solution
    virtual void initialise() {this->collection.initialise(this->sys.get_resolutions());}

    bool need_tangents() const {
      return (this->get_tangent_req() == TangentRequirement::NeedTangents);}

    bool need_effect() const {
      return (this->get_tangent_req() == TangentRequirement::NeedEffect);}

    bool no_need_tangent() const {
      return (this->get_tangent_req() == TangentRequirement::NoNeed);}

    virtual bool has_converged() const = 0;

    //! reset the iteration counter to zero
    void reset_counter();

    //! get the count of how many solve steps have been executed since
    //! construction of most recent counter reset
    Uint get_counter() const;

    virtual SolvVectorOut solve(const SolvVectorInC rhs, SolvVectorIn x_0) = 0;

    Sys_t & get_system() {return sys;}

    Uint get_maxiter() const {return this->maxiter;}
    void set_maxiter(Uint val) {this->maxiter = val;}

    Real get_tol() const {return this->tol;}
    void set_tol(Real val) {this->tol = val;}

    virtual std::string name() const = 0;

  protected:
    virtual TangentRequirement get_tangent_req() const = 0;
    Sys_t & sys;
    Real tol;
    Uint maxiter;
    bool verbose;
    Uint counter{0};
    //! storage for internal fields to avoid reallocations between calls
    Collection_t collection{};
  private:
  };

}  // muSpectre

#endif /* SOLVER_BASE_H */
