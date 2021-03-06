/**
 * @file   projection_finite_strain_fast.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   12 Dec 2017
 *
 * @brief  Faster alternative to ProjectionFinitestrain
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

#ifndef PROJECTION_FINITE_STRAIN_FAST_H
#define PROJECTION_FINITE_STRAIN_FAST_H

#include "fft/projection_base.hh"
#include "common/common.hh"
#include "common/field_collection.hh"
#include "common/field_map.hh"

namespace muSpectre {

  /**
   * replaces `muSpectre::ProjectionFiniteStrain` with a faster and
   * less memory-hungry alternative formulation. Use this if you don't
   * have a very good reason not to (and tell me (author) about it,
   * I'd be interested to hear it).
   */
  template <Dim_t DimS, Dim_t DimM>
  class ProjectionFiniteStrainFast: public ProjectionBase<DimS, DimM>
  {
  public:
    using Parent = ProjectionBase<DimS, DimM>; //!< base class
    //! polymorphic pointer to FFT engines
    using FFTEngine_ptr = typename Parent::FFTEngine_ptr;
    using Ccoord = typename Parent::Ccoord; //!< cell coordinates type
    using Rcoord = typename Parent::Rcoord; //!< spatial coordinates type
    //! global field collection (for real-space representations)
    using GFieldCollection_t = GlobalFieldCollection<DimS>;
    //! local field collection (for Fourier-space representations)
    using LFieldCollection_t = LocalFieldCollection<DimS>;
    //! Real space second order tensor fields (to be projected)
    using Field_t = TypedField<GFieldCollection_t, Real>;
    //! Fourier-space field containing the projection operator itself
    using Proj_t = TensorField<LFieldCollection_t, Real, firstOrder, DimM>;
    //! iterable form of the operator
    using Proj_map = MatrixFieldMap<LFieldCollection_t, Real, DimM, 1>;
    //! iterable Fourier-space second-order tensor field
    using Grad_map = MatrixFieldMap<LFieldCollection_t, Complex, DimM, DimM>;



    //! Default constructor
    ProjectionFiniteStrainFast() = delete;

    //! Constructor with fft_engine
    ProjectionFiniteStrainFast(FFTEngine_ptr engine, Rcoord lengths);

    //! Copy constructor
    ProjectionFiniteStrainFast(const ProjectionFiniteStrainFast &other) = delete;

    //! Move constructor
    ProjectionFiniteStrainFast(ProjectionFiniteStrainFast &&other) = default;

    //! Destructor
    virtual ~ProjectionFiniteStrainFast() = default;

    //! Copy assignment operator
    ProjectionFiniteStrainFast& operator=(const ProjectionFiniteStrainFast &other) = delete;

    //! Move assignment operator
    ProjectionFiniteStrainFast& operator=(ProjectionFiniteStrainFast &&other) = default;

    //! initialises the fft engine (plan the transform)
    virtual void initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate) override final;

    //! apply the projection operator to a field
    void apply_projection(Field_t & field) override final;

    Eigen::Map<Eigen::ArrayXXd> get_operator() override final;

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetriy storage, it is a column
     * vector)
     */
    std::array<Dim_t, 2> get_strain_shape() const override final;

    constexpr static Dim_t NbComponents(){return ipow(DimM, 2);}

  protected:
    Proj_t & xiField; //!< field of normalised wave vectors
    Proj_map xis;     //!< iterable normalised wave vectors
  private:
  };

}  // muSpectre

#endif /* PROJECTION_FINITE_STRAIN_FAST_H */
