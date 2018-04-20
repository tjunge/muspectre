/**
 * @file   cell_split.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   19 Apr 2018
 *
 * @brief Base class representing a unit cell able to handle 
 *        split material assignments
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

#ifndef CELL_SPLIT_H
#define CELL_SPLIT_H

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "common/field.hh"
#include "common/utilities.hh"
#include "materials/material_base.hh"
#include "fft/projection_base.hh"
#include "cell/cell_traits.hh"
#include "cell/cell_base.hh"

#include <vector>
#include <memory>
#include <tuple>
#include <functional>
namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  /* This class handles the cells that has splitly assigned material to their pixels */
  template <Dim_t DimS, Dim_t DimM=DimS>
  class CellSplit: public CellBase<DimS, DimM> {
  public:
    //! global field collection
    using FieldCollection_t = GlobalFieldCollection<DimS>;
    using Projection_t = ProjectionBase<DimS, DimM>;
    //! projections handled through `std::unique_ptr`s
    using Projection_ptr = std::unique_ptr<Projection_t>;
    using StrainField_t =
      TensorField<FieldCollection_t, Real, secondOrder, DimM>;
    //! expected type for stress fields
    using StressField_t =
      TensorField<FieldCollection_t, Real, secondOrder, DimM>;
    //! expected type for tangent stiffness fields
    using TangentField_t =
      TensorField<FieldCollection_t, Real, fourthOrder, DimM>;
    //! combined stress and tangent field
    using FullResponse_t =
      std::tuple<const StressField_t&, const TangentField_t&>;


    //! Default constructor
    CellSplit() = delete;

    //! constructor using sizes and resolution
    CellSplit(Projection_ptr projection);

    //! Copy constructor
    CellSplit(const CellSplit &other) = delete;

    //! Move constructor
    CellSplit(CellSplit &&other) = default;

    //! Destructor
    virtual ~CellSplit() = default;

    //! Copy assignment operator
    CellSplit& operator=(const CellSplit &other) = delete;

    //! Move assignment operator
    CellSplit& operator=(CellSplit &&other) = default;

  protected:
    void check_material_coverage();
    void set_p_k_zero();
    //full resppnse is consisted of the stresses and tangent matrix
    FullResponse_t evaluate_stress_tangent(StrainField_t & F) override final;
    FullResponse_t evaluate_split_stress_tangent(StrainField_t & F);
  private:
  };



}

#endif /* CELL_SPLIT_H */
