/**
 * file   material_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   25 Oct 2017
 *
 * @brief  Base class for materials (constitutive models)
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

#ifndef MATERIAL_BASE_H
#define MATERIAL_BASE_H

#include "common/common.hh"
#include "common/field.hh"
#include "common/field_collection.hh"

#include <string>

namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  template<Dim_t DimS, Dim_t DimM>
  class MaterialBase
  {
  public:
    //! typedefs for data handled by this interface
    //! global field collection for system-wide fields, like stress, strain, etc
    using GFieldCollection_t = FieldCollection<DimS, DimM, true>;
    //! field collection for internal variables, such as eigen-strains,
    //! plastic strains, damage variables, etc, but also for managing which
    //! pixels the material is responsible for
    using MFieldCollection_t = FieldCollection<DimS, DimM, false>;
    using iterator = typename MFieldCollection_t::iterator;
    using Field_t = internal::FieldBase<GFieldCollection_t>;
    using StressField_t = TensorField<GFieldCollection_t, Real, secondOrder, DimM>;
    using StrainField_t = StressField_t;
    using TangentField_t = TensorField<GFieldCollection_t, Real, fourthOrder, DimM>;
    using Ccoord = Ccoord_t<DimS>;
    //! Default constructor
    MaterialBase() = delete;

    //! Construct by name
    MaterialBase(std::string name);

    //! Copy constructor
    MaterialBase(const MaterialBase &other) = delete;

    //! Move constructor
    MaterialBase(MaterialBase &&other) = delete;

    //! Destructor
    virtual ~MaterialBase() = default;

    //! Copy assignment operator
    MaterialBase& operator=(const MaterialBase &other) = delete;

    //! Move assignment operator
    MaterialBase& operator=(MaterialBase &&other) = delete;


    /**
     *  take responsibility for a pixel identified by its cell coordinates
     *  WARNING: this won't work for materials with additional info per pixel
     *  (as, e.g. for eigenstrain), we need to pass more parameters. Materials
     *  of this tye need to overload add_pixel
     */
    void add_pixel(const Ccoord & ccooord);

    //! allocate memory, etc, but also: wipe history variables!
    virtual void initialise(bool stiffness = false) = 0;

    //! return the materil's name
    const std::string & get_name() const;

    //! for static inheritance stuff
    constexpr static Dim_t sdim() {return DimS;}
    constexpr static Dim_t mdim() {return DimM;}
    //! computes stress
    virtual void compute_stresses(const StrainField_t & F,
                                  StressField_t & P,
                                  Formulation form) = 0;
    /**
     * Convenience function to compute stresses, mostly for debugging and
     * testing. Has runtime-cost associated with compatibility-checking and
     * conversion of the Field_t arguments that can be avoided by using the
     * version with strongly typed field references
     */
    void compute_stresses(const Field_t & F,
                          Field_t & P,
                          Formulation form);
    //! computes stress and tangent moduli
    virtual void compute_stresses_tangent(const StrainField_t & F,
                                          StressField_t & P,
                                          TangentField_t & K,
                                          Formulation form) = 0;
    /**
     * Convenience function to compute stresses and tangent moduli, mostly for
     * debugging and testing. Has runtime-cost associated with
     * compatibility-checking and conversion of the Field_t arguments that can
     * be avoided by using the version with strongly typed field references
     */
    void compute_stresses_tangent(const Field_t & F,
                                  Field_t & P,
                                  Field_t & K,
                                  Formulation form);

    inline iterator begin() {return this->internal_fields.begin();}
    inline iterator end()  {return this->internal_fields.end();}
    inline size_t size() const {return this->internal_fields.size();}
  protected:

    //! members
    const std::string name;
    MFieldCollection_t internal_fields{};


  private:
  };
}  // muSpectre

#endif /* MATERIAL_BASE_H */
