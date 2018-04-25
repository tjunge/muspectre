/**
 * file   field_typed.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   10 Apr 2018
 *
 * @brief  Typed Field for dynamically sized fields and base class for fields 
 *         of tensors, matrices, etc
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

#ifndef FIELD_TYPED_H
#define FIELD_TYPED_H

#include "field_base.hh"

#include <sstream>

namespace muSpectre {

  /**
   * Dummy intermediate class to provide a run-time polymorphic
   * typed field. Mainly for binding Python. TypedField specifies methods
   * that return typed Eigen maps and vectors in addition to pointers to the
   * raw data.
   */
  template <class FieldCollection, typename T>
  class TypedField: public internal::FieldBase<FieldCollection>
  {
  public:
    using Parent = internal::FieldBase<FieldCollection>; //!< base class
    //! for type checks when mapping this field
    using collection_t = typename Parent::collection_t;
    using Scalar = T; //!< for type checks
    using Base = Parent; //!< for uniformity of interface
    //! Plain Eigen type to map
    using EigenRep_t = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
    //! map returned when iterating over field
    using EigenMap_t = Eigen::Map<EigenRep_t>;
    //! Plain eigen vector to map
    using EigenVec_t = Eigen::Map<Eigen::VectorXd>;
    //! vector map returned when iterating over field
    using EigenVecConst_t = Eigen::Map<const Eigen::VectorXd>;

    /**
     * type stored (unfortunately, we can't statically size the second
     * dimension due to an Eigen bug,i.e., creating a row vector
     * reference to a column vector does not raise an error :(
     */
    using Stored_t = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
    //! storage container
    using Storage_t = std::vector<T,Eigen::aligned_allocator<T>>;

    //! Default constructor
    TypedField() = delete;

    //! constructor
    TypedField(std::string unique_name,
                   FieldCollection& collection,
                   size_t nb_components);

    //! Copy constructor
    TypedField(const TypedField &other) = delete;

    //! Move constructor
    TypedField(TypedField &&other) = delete;

    //! Destructor
    virtual ~TypedField() = default;

    //! Copy assignment operator
    TypedField& operator=(const TypedField &other) = delete;

    //! Move assignment operator
    TypedField& operator=(TypedField &&other) = delete;

    //! return type_id of stored type
    virtual const std::type_info & get_stored_typeid() const override final;

    virtual size_t size() const override final;

    //! add a pad region to the end of the field buffer; required for
    //! using this as e.g. an FFT workspace
    void set_pad_size(size_t pad_size_) override final;

    //! initialise field to zero (do more complicated initialisations through
    //! fully typed maps)
    virtual void set_zero() override final;

    //! add a new value at the end of the field
    inline void push_back(const Stored_t & value);


    //! raw pointer to content (e.g., for Eigen maps)
    virtual T* data() {return this->get_ptr_to_entry(0);}
    //! raw pointer to content (e.g., for Eigen maps)
    virtual const T* data() const {return this->get_ptr_to_entry(0);}

    //! return a map representing the entire field as a single `Eigen::Array`
    EigenMap_t eigen();
    //! return a map representing the entire field as a single Eigen vector
    EigenVec_t eigenvec();
    //! return a map representing the entire field as a single Eigen vector
    EigenVecConst_t eigenvec() const;
    //! return a map representing the entire field as a single Eigen vector
    EigenVecConst_t const_eigenvec() const;

  protected:
    //! returns a raw pointer to the entry, for `Eigen::Map`
    inline T* get_ptr_to_entry(const size_t&& index);

    //! returns a raw pointer to the entry, for `Eigen::Map`
    inline const T*
    get_ptr_to_entry(const size_t&& index) const;

    //! set the storage size of this field
    inline virtual void resize(size_t size) override final;

    //! The actual storage container
    Storage_t values{};
    /**
     * an unregistered typed field can be mapped onto an array of
     * existing values
     */
    optional<Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>>> alt_values{};

    size_t current_size;

    T* data_ptr{};

  private:
  };

  /* ---------------------------------------------------------------------- */
  /* Implementations                                                        */
  /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T>
  TypedField<FieldCollection, T>::
  TypedField(std::string unique_name, FieldCollection & collection,
             size_t nb_components)
    :Parent(unique_name, nb_components, collection), current_size{0}
  {}

  /* ---------------------------------------------------------------------- */
  //! return type_id of stored type
  template <class FieldCollection, typename T>
  const std::type_info & TypedField<FieldCollection, T>::
  get_stored_typeid() const {
    return typeid(T);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::eigen() -> EigenMap_t {
    return EigenMap_t(this->data(), this->get_nb_components(), this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::eigenvec() -> EigenVec_t {
    return EigenVec_t(this->data(), this->get_nb_components() * this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>:: eigenvec() const -> EigenVecConst_t {
    return EigenVecConst_t(this->data(), this->get_nb_components() * this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>:: const_eigenvec() const -> EigenVecConst_t {
    return EigenVecConst_t(this->data(), this->get_nb_components() * this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  void TypedField<FieldCollection, T>::resize(size_t size) {
    if (this->alt_values) {
      throw FieldError("Field proxies can't resize.");
    }
    this->current_size = size;
    this->values.resize(size*this->get_nb_components() + this->pad_size);
    this->data_ptr = &this->values.front();
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  void TypedField<FieldCollection, T>::set_zero() {
    std::fill(this->values.begin(), this->values.end(), T{});
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  size_t TypedField<FieldCollection, T>::
  size() const {
    return this->current_size;
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  void TypedField<FieldCollection, T>::
  set_pad_size(size_t pad_size) {
    if (this->alt_values) {
      throw FieldError("You can't set the pad size of a field proxy.");
    }
    this->pad_size = pad_size;
    this->resize(this->size());
    this->data_ptr = &this->values.front();
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  T* TypedField<FieldCollection, T>::
  get_ptr_to_entry(const size_t&& index) {
    return &this->values[this->get_nb_components()*std::move(index)];
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  const T* TypedField<FieldCollection, T>::
  get_ptr_to_entry(const size_t&& index) const {
    return &this->values[this->get_nb_components()*std::move(index)];
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  void TypedField<FieldCollection, T>::
  push_back(const Stored_t & value) {
    static_assert (not FieldCollection::Global,
                   "You can only push_back data into local field collections");
    if (value.cols() != 1) {
      std::stringstream err{};
      err << "Expected a column vector, but received and array with "
          << value.cols() <<" colums.";
      throw FieldError(err.str());
    }
    if (value.rows() != static_cast<Int>(this->get_nb_components())) {
      std::stringstream err{};
      err << "Expected a column vector of length " << this->get_nb_components()
          << ", but received one of length " << value.rows() <<".";
      throw FieldError(err.str());
    }
    for (size_t i = 0; i < this->get_nb_components(); ++i) {
      this->values.push_back(value(i));
    }
    ++this->current_size;
    this->data_ptr = &this->values.front();
  }

}  // muSpectre

#endif /* FIELD_TYPED_H */
