/**
 * @file   cell_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Nov 2017
 *
 * @brief Base class representing a unit cell cell with single
 *        projection operator
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

#ifndef CELL_BASE_H
#define CELL_BASE_H

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "common/field.hh"
#include "common/utilities.hh"
#include "materials/material_base.hh"
#include "fft/projection_base.hh"
#include "cell/cell_traits.hh"

#include <vector>
#include <memory>
#include <tuple>
#include <functional>
#include <array>

namespace muSpectre {
  /**
   * Cell adaptors implement the matrix-vector multiplication and
   * allow the system to be used like a sparse matrix in
   * conjugate-gradient-type solvers
   */
  template <class Cell>
  class CellAdaptor;

  /**
   * Base class for cells that is not templated and therefore can be
   * in solvers that see cells as runtime-polymorphic objects. This
   * allows the use of standard
   * (i.e. spectral-method-implementation-agnostic) solvers, as for
   * instance the scipy solvers
   */

  class Cell
  {
  public:

    //! dynamic vector type for interactions with numpy/scipy/solvers etc.
    using Vector_t = Eigen::VectorXd;

    //! ref to constant vector
    using ConstVector_ref = Eigen::Map<const Vector_t>;

    //! output vector reference for solvers
    using Vector_ref = Eigen::Map<Vector_t>;

    //! Default constructor
    Cell() = default;

    //! Copy constructor
    Cell(const Cell &other) = default;

    //! Move constructor
    Cell(Cell &&other) = default;

    //! Destructor
    virtual ~Cell()  = default;

    //! Copy assignment operator
    Cell& operator=(const Cell &other) = default;

    //! Move assignment operator
    Cell& operator=(Cell &&other) = default;

    //! for handling double initialisations right
    bool is_initialised() const {return this->initialised;}

    //! return the communicator object
    virtual const Communicator & get_communicator() const = 0;

    /**
     * returns a writable map onto the strain field of this cell. This
     * corresponds to the unknowns in a typical solve cycle.
     */
    virtual Vector_ref get_strain_vector() = 0;

    /**
     * returns a read-only map onto the stress field of this
     * cell. This corresponds to the intermediate (and finally, total)
     * solution in a typical solve cycle
     */
    virtual ConstVector_ref get_stress_vector() const = 0;


    /**
     * evaluates and returns the stress for the currently set strain
     */
    virtual ConstVector_ref evaluate_stress() = 0;

    /**
     * evaluates and returns the stress and stiffness for the currently set strain
     */
    virtual std::array<ConstVector_ref, 2> evaluate_stress_tangent() = 0;

    /**
     * evaluates the directional and projected stiffness (this
     * corresponds to G:K:δF in de Geus 2017,
     * http://dx.doi.org/10.1016/j.cma.2016.12.032). It seems that
     * this operation needs to be implemented with a copy in oder to
     * be compatible with scipy and EigenCG etc (At the very least,
     * the copy is only made once)
     */
    virtual Vector_ref evaluate_projected_directional_stiffness
      (Eigen::Ref<Vector_t> delF) = 0;


  protected:
    bool initialised{false}; //!< to handle double initialisation right

  private:
  };
  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  template <Dim_t DimS, Dim_t DimM=DimS>
  class CellBase: public Cell
  {
  public:
    using Parent = Cell;
    using Ccoord = Ccoord_t<DimS>; //!< cell coordinates type
    using Rcoord = Rcoord_t<DimS>; //!< physical coordinates type
    //! global field collection
    using FieldCollection_t = GlobalFieldCollection<DimS>;
    //! the collection is handled in a `std::unique_ptr`
    using Collection_ptr = std::unique_ptr<FieldCollection_t>;
    //! polymorphic base material type
    using Material_t = MaterialBase<DimS, DimM>;
    //! materials handled through `std::unique_ptr`s
    using Material_ptr = std::unique_ptr<Material_t>;
    //! polymorphic base projection type
    using Projection_t = ProjectionBase<DimS, DimM>;
    //! projections handled through `std::unique_ptr`s
    using Projection_ptr = std::unique_ptr<Projection_t>;
    //! expected type for strain fields
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
    //! iterator type over all cell pixel's
    using iterator = typename CcoordOps::Pixels<DimS>::iterator;

    //! dynamic vector type for interactions with numpy/scipy/solvers etc.
    using Vector_t = typename Parent::Vector_t;

    //! ref to constant vector
    using ConstVector_ref = typename Parent::ConstVector_ref;

    //! output vector reference for solvers
    using Vector_ref = typename Parent::Vector_ref;


    //! sparse matrix emulation
    using Adaptor = CellAdaptor<CellBase>;

    //! Default constructor
    CellBase() = delete;

    //! constructor using sizes and resolution
    CellBase(Projection_ptr projection);

    //! Copy constructor
    CellBase(const CellBase &other) = delete;

    //! Move constructor
    CellBase(CellBase &&other) = default;

    //! Destructor
    virtual ~CellBase() = default;

    //! Copy assignment operator
    CellBase& operator=(const CellBase &other) = delete;

    //! Move assignment operator
    CellBase& operator=(CellBase &&other) = default;

    /**
     * Materials can only be moved. This is to assure exclusive
     * ownership of any material by this cell
     */
    Material_t & add_material(Material_ptr mat);


    /**
     * returns a writable map onto the strain field of this cell. This
     * corresponds to the unknowns in a typical solve cycle.
     */
    virtual Vector_ref get_strain_vector() override;

    /**
     * returns a read-only map onto the stress field of this
     * cell. This corresponds to the intermediate (and finally, total)
     * solution in a typical solve cycle
     */
    virtual ConstVector_ref get_stress_vector() const override;

    /**
     * evaluates and returns the stress for the currently set strain
     */
    virtual ConstVector_ref evaluate_stress() override;

    /**
     * evaluates and returns the stress and stiffness for the currently set strain
     */
    virtual std::array<ConstVector_ref, 2> evaluate_stress_tangent() override;


    /**
     * evaluates the directional and projected stiffness (this
     * corresponds to G:K:δF in de Geus 2017,
     * http://dx.doi.org/10.1016/j.cma.2016.12.032). It seems that
     * this operation needs to be implemented with a copy in oder to
     * be compatible with scipy and EigenCG etc. (At the very least,
     * the copy is only made once)
     */
    virtual Vector_ref evaluate_projected_directional_stiffness
      (Eigen::Ref<Vector_t> delF) override;


    /**
     * evaluates all materials
     */
    FullResponse_t evaluate_stress_tangent(StrainField_t & F);

    /**
     * evaluate directional stiffness (i.e. G:K:δF or G:K:δε)
     */

    StressField_t & directional_stiffness(const TangentField_t & K,
                                          const StrainField_t & delF,
                                          StressField_t & delP);
    /**
     * vectorized version for eigen solvers, no copy, but only works
     * when fields have ArrayStore=false
     */
    Vector_ref directional_stiffness_vec(const Eigen::Ref<const Vector_t> & delF);
    /**
     * Evaluate directional stiffness into a temporary array and
     * return a copy. This is a costly and wasteful interface to
     * directional_stiffness and should only be used for debugging or
     * in the python interface
     */
    Eigen::ArrayXXd directional_stiffness_with_copy
      (Eigen::Ref<Eigen::ArrayXXd> delF);

    /**
     * Convenience function circumventing the neeed to use the
     * underlying projection
     */
    StressField_t & project(StressField_t & field);

    //! returns a ref to the cell's strain field
    StrainField_t & get_strain();

    //! returns a ref to the cell's stress field
    const StressField_t & get_stress() const;

    //! returns a ref to the cell's tangent stiffness field
    const TangentField_t & get_tangent(bool create = false);

    //! returns a ref to a temporary field managed by the cell
    StrainField_t & get_managed_field(std::string unique_name);

    /**
     * general initialisation; initialises the projection and
     * fft_engine (i.e. infrastructure) but not the materials. These
     * need to be initialised separately
     */
    void initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate);
    /**
     * initialise materials (including resetting any history variables)
     */
    void initialise_materials(bool stiffness=false);

    /**
     * for materials with state variables, these typically need to be
     * saved/updated an the end of each load increment, this function
     * calls this update for each material in the cell
     */
    void save_history_variables();

    iterator begin(); //!< iterator to the first pixel
    iterator end();  //!< iterator past the last pixel
    //! number of pixels in the cell
    size_t size() const {return pixels.size();}

    //! return the subdomain resolutions of the cell
    const Ccoord & get_subdomain_resolutions() const {
      return this->subdomain_resolutions;}
    //! return the subdomain locations of the cell
    const Ccoord & get_subdomain_locations() const {
      return this->subdomain_locations;}
    //! return the domain resolutions of the cell
    const Ccoord & get_domain_resolutions() const {
      return this->domain_resolutions;}
    //! return the sizes of the cell
    const Rcoord & get_domain_lengths() const {return this->domain_lengths;}

    /**
     * formulation is hard set by the choice of the projection class
     */
    const Formulation & get_formulation() const {
      return this->projection->get_formulation();}

    /**
     * get a reference to the projection object. should only be
     * required for debugging
     */
    Eigen::Map<Eigen::ArrayXXd> get_projection() {
      return this->projection->get_operator();}

    //! returns the spatial size
    constexpr static Dim_t get_sdim() {return DimS;};

    //! return a sparse matrix adaptor to the cell
    Adaptor get_adaptor();
    //! returns the number of degrees of freedom in the cell
    Dim_t nb_dof() const {return this->size()*ipow(DimS, 2);};

    //! return the communicator object
    virtual const Communicator & get_communicator() const override {
      return this->projection->get_communicator();
    }

  protected:
    //! make sure that every pixel is assigned to one and only one material
    void check_material_coverage();

    const Ccoord & subdomain_resolutions; //!< the cell's subdomain resolutions
    const Ccoord & subdomain_locations; //!< the cell's subdomain resolutions
    const Ccoord & domain_resolutions; //!< the cell's domain resolutions
    CcoordOps::Pixels<DimS> pixels; //!< helper to iterate over the pixels
    const Rcoord & domain_lengths; //!< the cell's lengths
    Collection_ptr fields; //!< handle for the global fields of the cell
    StrainField_t & F; //!< ref to strain field
    StressField_t & P; //!< ref to stress field
    //! Tangent field might not even be required; so this is an
    //! optional ref_wrapper instead of a ref
    optional<std::reference_wrapper<TangentField_t>> K{};
    //! container of the materials present in the cell
    std::vector<Material_ptr> materials{};
    Projection_ptr projection; //!< handle for the projection operator
    const Formulation form; //!< formulation for solution
  private:
  };


  /**
   * lightweight resource handle wrapping a `muSpectre::CellBase` or
   * a subclass thereof into `Eigen::EigenBase`, so it can be
   * interpreted as a sparse matrix by Eigen solvers
   */
  template <class Cell>
  class CellAdaptor: public Eigen::EigenBase<CellAdaptor<Cell>> {

  public:
    using Scalar = double;     //!< sparse matrix traits
    using RealScalar = double; //!< sparse matrix traits
    using StorageIndex = int;  //!< sparse matrix traits
    enum {
      ColsAtCompileTime = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic,
      RowsAtCompileTime = Eigen::Dynamic,
      MaxRowsAtCompileTime = Eigen::Dynamic,
      IsRowMajor = false
    };

    //! constructor
    CellAdaptor(Cell & cell):cell{cell}{}
    //!returns the number of logical rows
    Eigen::Index rows() const {return this->cell.nb_dof();}
    //!returns the number of logical columns
    Eigen::Index cols() const {return this->rows();}

    //! implementation of the evaluation
    template<typename Rhs>
    Eigen::Product<CellAdaptor,Rhs,Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs>& x) const {
      return Eigen::Product<CellAdaptor,Rhs,Eigen::AliasFreeProduct>
        (*this, x.derived());
    }
    Cell & cell; //!< ref to the cell
  };

}  // muSpectre


namespace Eigen {
  namespace internal {
    //! Implementation of `muSpectre::CellAdaptor` * `Eigen::DenseVector` through a
    //! specialization of `Eigen::internal::generic_product_impl`:
    template<typename Rhs, class CellAdaptor>
    struct generic_product_impl<CellAdaptor, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
      : generic_product_impl_base<CellAdaptor,Rhs,generic_product_impl<CellAdaptor,Rhs> >
    {
      //! undocumented
      typedef typename Product<CellAdaptor,Rhs>::Scalar Scalar;

      //! undocumented
      template<typename Dest>
      static void scaleAndAddTo(Dest& dst, const CellAdaptor& lhs, const Rhs& rhs, const Scalar& /*alpha*/)
      {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
        dst.noalias() += const_cast<CellAdaptor&>(lhs).cell.directional_stiffness_vec(rhs);
      }
    };
  }
}


#endif /* CELL_BASE_H */
