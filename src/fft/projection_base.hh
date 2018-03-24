/**
 * @file   projection_base.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  Base class for Projection operators
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

#ifndef PROJECTION_BASE_H
#define PROJECTION_BASE_H

#include "common/common.hh"
#include "common/field_collection.hh"
#include "common/field.hh"
#include "fft/fft_engine_base.hh"

#include <memory>

namespace muSpectre {

  /**
   * base class for projection related exceptions
   */
  class ProjectionError: public std::runtime_error {
  public:
    //! constructor
    explicit ProjectionError(const std::string& what)
      :std::runtime_error(what){}
    //! constructor
    explicit ProjectionError(const char * what)
      :std::runtime_error(what){}
  };

  template<class Projection>
  struct Projection_traits {
  };

  /**
   * defines the interface which must be implemented by projection operators
   */
  template <Dim_t DimS, Dim_t DimM>
  class ProjectionBase
  {
  public:
    //! type of fft_engine used
    using FFTEngine = FFTEngineBase<DimS, DimM>;
    //! reference to fft engine is safely managed through a `std::unique_ptr`
    using FFTEngine_ptr = std::unique_ptr<FFTEngine>;
    //! cell coordinates type
    using Ccoord = typename FFTEngine::Ccoord;
    //! spatial coordinates type
    using Rcoord = typename FFTEngine::Rcoord;
    //! global FieldCollection
    using GFieldCollection_t = typename FFTEngine::GFieldCollection_t;
    //! local FieldCollection (for Fourier-space pixels)
    using LFieldCollection_t = typename FFTEngine::LFieldCollection_t;
    //! Field type on which to apply the projection
    using Field_t = typename FFTEngine::Field_t;
    /**
     * iterator over all pixels. This is taken from the FFT engine,
     * because depending on the real-to-complex FFT employed, only
     * roughly half of the pixels are present in Fourier space
     * (because of the hermitian nature of the transform)
     */
    using iterator = typename FFTEngine::iterator;

    //! Default constructor
    ProjectionBase() = delete;

    //! Constructor with cell sizes
    ProjectionBase(FFTEngine_ptr engine, Formulation form);

    //! Copy constructor
    ProjectionBase(const ProjectionBase &other) = delete;

    //! Move constructor
    ProjectionBase(ProjectionBase &&other) = default;

    //! Destructor
    virtual ~ProjectionBase() = default;

    //! Copy assignment operator
    ProjectionBase& operator=(const ProjectionBase &other) = delete;

    //! Move assignment operator
    ProjectionBase& operator=(ProjectionBase &&other) = default;

    //! initialises the fft engine (plan the transform)
    virtual void initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate);

    //! apply the projection operator to a field
    virtual void apply_projection(Field_t & field) = 0;

    //! returns the process-local resolutions of the cell
    const Ccoord & get_resolutions() const {
      return this->fft_engine->get_resolutions();}
    //! returns the process-local locations of the cell
    const Ccoord & get_locations() const {
      return this->fft_engine->get_locations();}
    //! returns the resolutions of the cell
    const Ccoord & get_domain_resolutions() const {
      return this->fft_engine->get_domain_resolutions();}
    //! returns the physical sizes of the cell
    const Rcoord & get_lengths() const {
      return this->fft_engine->get_lengths();}

    /**
     * return the `muSpectre::Formulation` that is used in solving
     * this cell. This allows tho check whether a projection is
     * compatible with the chosen formulation
     */
    const Formulation & get_formulation() const {return this->form;}

    //! return the raw projection operator. This is mainly intended
    //! for maintenance and debugging and should never be required in
    //! regular use
    virtual Eigen::Map<Eigen::ArrayXXd> get_operator() = 0;

    //! return the communicator object
    const Communicator & get_communicator() const {
      return this->fft_engine->get_communicator();
    }

  protected:
    //! handle on the fft_engine used
    FFTEngine_ptr fft_engine;
    /**
     * formulation this projection can be applied to (determines
     * whether the projection enforces gradients, small strain tensor
     * or symmetric smal strain tensor
     */
    const Formulation form;
    /**
     * A local `muSpectre::FieldCollection` to store the projection
     * operator per k-space point. This is a local rather than a
     * global collection, since the pixels considered depend on the
     * FFT implementation. See
     * http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
     * for an example
     */
    LFieldCollection_t & projection_container{};

  private:
  };

}  // muSpectre



#endif /* PROJECTION_BASE_H */
