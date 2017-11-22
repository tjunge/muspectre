/**
 * file   field_collection_local.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  FieldCollection base-class for local fields
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#ifndef FIELD_COLLECTION_LOCAL_H
#define FIELD_COLLECTION_LOCAL_H

#include "common/field_collection_base.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  //! DimS spatial dimension (dimension of problem)
  //! DimM material_dimension (dimension of constitutive law)
  template<Dim_t DimS, Dim_t DimM>
  class LocalFieldCollection:
    public FieldCollectionBase<DimS, DimM, LocalFieldCollection<DimS, DimM>>
  {
  public:
    using Parent = FieldCollectionBase<DimS, DimM,
                                       LocalFieldCollection<DimS, DimM>>;
    using Ccoord = typename Parent::Ccoord;
    using Field_p = typename Parent::Field_p;

    //! Default constructor
    LocalFieldCollection();

    //! Copy constructor
    LocalFieldCollection(const LocalFieldCollection &other) = delete;

    //! Move constructor
    LocalFieldCollection(LocalFieldCollection &&other) noexcept = delete;

    //! Destructor
    virtual ~LocalFieldCollection() noexcept = default;

    //! Copy assignment operator
    LocalFieldCollection& operator=(const LocalFieldCollection &other) = delete;

    //! Move assignment operator
    LocalFieldCollection& operator=(LocalFieldCollection &&other) noexcept = delete;

    //! add a pixel/voxel to the field collection
    inline void add_pixel(const Ccoord & local_ccoord);

    /** allocate memory, etc. at this point, the field collection
        knows how many entries it should have from the size of the
        coords containes (which grows by one every time add_pixel is
        called. The job of initialise is to make sure that all fields
        are either of size 0, in which case they need to be allocated,
        or are of the same size as the product of 'sizes' any field of
        a different size is wrong TODO: check whether it makes sense
        to put a runtime check here
     **/
    inline void initialise();


    //! returns the linear index corresponding to cell coordinates
    inline size_t get_index(Ccoord && ccoord) const;
    //! returns the cell coordinates corresponding to a linear index
    inline Ccoord get_ccoord(size_t index) const;

  protected:
    //! container of pixel coords for non-global collections
    std::vector<Ccoord> ccoords{};
    //! container of indices for non-global collections (slow!)
    std::map<Ccoord, size_t> indices{};
  private:
  };


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  LocalFieldCollection<DimS, DimM>::LocalFieldCollection()
    :Parent(){}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void LocalFieldCollection<DimS, DimM>::
  add_pixel(const Ccoord & local_ccoord) {
    if (this->is_initialised) {
      throw FieldCollectionError
        ("once a field collection has been initialised, you can't add new "
         "pixels.");
    }
    this->indices[local_ccoord] = this->ccoords.size();
    this->ccoords.push_back(local_ccoord);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void LocalFieldCollection<DimS, DimM>::
  initialise() {
    this->size_ = this->ccoords.size();
    std::for_each(std::begin(this->fields), std::end(this->fields),
                  [this](auto && item) {
                    auto && field = *item.second;
                    const auto field_size = field.size();
                    if (field_size == 0) {
                      field.resize(this->size());
                    } else if (field_size != this->size()) {
                      std::stringstream err_stream;
                      err_stream << "Field '" << field.get_name()
                                 << "' contains " << field_size
                                 << " entries, but the field collection "
                                 << "has " << this->size() << " pixels";
                      throw FieldCollectionError(err_stream.str());
                    }
                  });
    this->is_initialised = true;
  }


  //----------------------------------------------------------------------------//
  //! returns the linear index corresponding to cell coordinates
  template <Dim_t DimS, Dim_t DimM>
  size_t
  LocalFieldCollection<DimS, DimM>::get_index(Ccoord && ccoord) const {
    return this->indices[std::move(ccoord)];
  }


  //----------------------------------------------------------------------------//
  //! returns the cell coordinates corresponding to a linear index
  template <Dim_t DimS, Dim_t DimM>
  typename LocalFieldCollection<DimS, DimM>::Ccoord
  LocalFieldCollection<DimS, DimM>::get_ccoord(size_t index) const {
    return this->ccoords[std::move(index)];
  }


}  // muSpectre

#endif /* FIELD_COLLECTION_LOCAL_H */
