/**
 * @file   material_crystal_plasticity_finite.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Francesco Maresca <francesco.maresca@epfl.ch>
 *
 * @date   23 Feb 2018
 *
 * @brief finite strain crystal plasticity implementation
 *
 * Copyright © 2018 Till Junge, Francesco Maresca
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

#ifndef MATERIAL_CRYSTAL_PLASTICITY_FINITE_H
#define MATERIAL_CRYSTAL_PLASTICITY_FINITE_H

#include "materials/material_muSpectre_base.hh"
#include "common/field.hh"
#include "common/geometry.hh"
#include "common/statefield.hh"
#include "common/eigen_tools.hh"

#include <Eigen/Dense>

#include <cmath>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  class MaterialCrystalPlasticityFinite;

  /**
   * traits for objective linear elasticity with eigenstrain
   */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  struct MaterialMuSpectre_traits<MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>>
  {
    //! global field collection
    using GFieldCollection_t = typename
      MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for strain fields
    using StrainMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t = T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::Gradient};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};

    //! declare internal variables
    //! local field_collections used for internals
    using LFieldColl_t = LocalFieldCollection<DimS>;
    //! plastic deformation gradient Fₚ(t)
    using FpMap_t = StateFieldMap<MatrixFieldMap<LFieldColl_t, Real, DimM, DimM, false>>;
    //! plastic slip rates dγᵅ/dt
    using GammadotMap_t = StateFieldMap<MatrixFieldMap<LFieldColl_t, Real, NbSlip, 1, false>>;
    //! critical resolved shear stresses (CRSS) τᵅy(t)
    using TauyMap_t = GammadotMap_t;
    //! plastic slips γᵅ(t)
    using GammaMap_t = MatrixFieldMap<LFieldColl_t, Real, NbSlip, 1, false>;
    //! euler angles
    using EulerMap_t = ArrayFieldMap<LFieldColl_t, Real, (DimM==3) ? 3 : 1, 1, true>;

    using InternalVariables = std::tuple<FpMap_t, GammadotMap_t, TauyMap_t, GammaMap_t, EulerMap_t>;
  };

  /**
   * implements finite strain crystal plasticity
   */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  class MaterialCrystalPlasticityFinite:
    public MaterialMuSpectre<MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>, DimS, DimM>
  {
  public:
    constexpr static Int NbEuler{(DimM==3) ? 3 : 1};

    //! base class
    using Parent = MaterialMuSpectre<MaterialCrystalPlasticityFinite, DimS, DimM>;

    //! type for stiffness tensor construction
    using Stiffness_t = Eigen::TensorFixedSize
      <Real, Eigen::Sizes<DimM, DimM, DimM, DimM>>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialCrystalPlasticityFinite>;

    //! Type of container used for storing eigenstrain
    using InternalVariables = typename traits::InternalVariables;

    //! Hooke's law implementation
    using Hooke = typename
      MatTB::Hooke<DimM,
                   typename traits::StrainMap_t::reference,
                   typename traits::TangentMap_t::reference>;

    //! Type in which plastic deformation gradient is referenced
    using Fp_ref = typename traits::FpMap_t::reference;
    //! Type in which slip rates are referenced
    using Gammadot_ref = typename traits::GammadotMap_t::reference;
    //! Type in which CRSS are referenced
    using Tauy_ref = typename traits::TauyMap_t::reference;
    //! Type in which accumulated slips are referenced
    using Gamma_ref = typename traits::GammaMap_t::reference;
    //! Type in which Euler angles are referenced
    using Euler_ref = typename traits::EulerMap_t::reference;
    //! Type in which slip directions and normals are given
    using SlipVecs = Eigen::Matrix<Real, NbSlip, DimM>;
    using SlipVecs_ref = Eigen::Ref<SlipVecs>;
    //! Type for second rank tensors
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    //! Type for fourth rank tensors
    using T4_t = T4Mat<Real, DimM>;

    //! Default constructor
    MaterialCrystalPlasticityFinite() = delete;

    //! Construct by name, Bulk modulus, Shear modulus, Reference slip rate, Strain rate sensitivity, Initial CRSS, Initial hardening modulus, CRSS saturation value, Hardening exponent, Latent/Self-hardening ratio, Slip directions, Slip normals, Plastic slip rate tolerance, Plastic slip rate loop maximum iterations
    MaterialCrystalPlasticityFinite(std::string name, Real bulk_m, Real shear_m, Real gammadot_0, Real m_par, Real tauy0, Real h0, Real s_infty, Real a_par, Real q_n, SlipVecs_ref Slip0, SlipVecs_ref Normal0, Real DeltaT, Real tolerance=1.e-4, Int maxiter=20);

    //! Copy constructor
    MaterialCrystalPlasticityFinite(const MaterialCrystalPlasticityFinite &other) = delete;

    //! Move constructor
    MaterialCrystalPlasticityFinite(MaterialCrystalPlasticityFinite &&other) = delete;

    //! Destructor
    virtual ~MaterialCrystalPlasticityFinite() = default;

    //! Copy assignment operator
    MaterialCrystalPlasticityFinite& operator=(const MaterialCrystalPlasticityFinite &other) = delete;

    //! Move assignment operator
    MaterialCrystalPlasticityFinite& operator=(MaterialCrystalPlasticityFinite &&other) = delete;

    /**
     * evaluates second Piola-Kirchhoff stress given the Deformation Gradient
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress(s_t && F, Fp_ref Fp, Gammadot_ref Gammadot, Tauy_ref Tauy, Gamma_ref Gamma, Euler_ref Euler);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Deformation Gradient
     */
    template <class s_t>
    inline decltype(auto)
    evaluate_stress_tangent(s_t && F, Fp_ref Fp, Gammadot_ref Gammadot, Tauy_ref Tauy, Gamma_ref Gamma, Euler_ref Euler);

    /**
     * return the internals tuple
     */
    InternalVariables & get_internals() {
      return this->internal_variables;};

    /**
     * overload add_pixel to write into internals
     */
    void add_pixel(const Ccoord_t<DimS> & pixel) override final;

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const Ccoord_t<DimS> & pixel,
                   const Eigen::Ref<Eigen::Array<Real, NbEuler, 1>> Euler);

  protected:
    using LColl_t = typename traits::LFieldColl_t;
    //! Storage for Fp
    StateField<TensorField<LColl_t, Real, secondOrder, DimM>> FpField;
    //! Storage for gammadot
    StateField<MatrixField<LColl_t, Real, NbSlip, 1>> GammadotField;
    //! Storage for tauy
    StateField<MatrixField<LColl_t, Real, NbSlip, 1>> TauyField;
    //! Storage for gamma
    MatrixField<LColl_t, Real, NbSlip, 1> & GammaField;
    //! Storage for Euler angles
    MatrixField<LColl_t, Real, NbEuler, 1> & EulerField;
    //! bulk modulus
    Real bulk_m;
    Real shear_m;
    Real gammadot_0;
    Real m_par;
    Real tauy0;
    Real h0;
    Real s_infty;
    Real a_par;
    Real q_n;
    Real DeltaT;
    Real tolerance;
    Int maxiter;
    //! Storage for slip directions
    alignas(16) Eigen::Matrix<Real, NbSlip, DimM> Slip0;
    //! Storage for slip plane normals
    alignas(16) Eigen::Matrix<Real, NbSlip, DimM> Normal0;
    T4_t C_el{};

    //! tuple for iterable internal variables
    InternalVariables internal_variables;
  private:
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  template <class s_t>
  decltype(auto)
  MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  evaluate_stress(s_t && F, Fp_ref Fp, Gammadot_ref Gammadot, Tauy_ref Tauy,
                  Gamma_ref Gamma, Euler_ref Euler) {
    return std::get<0>(this->evaluate_stress_tangent(std::forward<s_t>(F),
                                                     Fp, Gammadot, Tauy, Gamma,
                                                     Euler));
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  template <class s_t>
  decltype(auto)
  MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  evaluate_stress_tangent(s_t && F, Fp_ref Fp, Gammadot_ref Gammadot, Tauy_ref Tauy, Gamma_ref /*Gamma*/, Euler_ref Euler) {
    Rotator<DimM> Rot(Euler);
    T2_t Floc{Rot.rotate(F)};
    std::array<T2_t, NbSlip> SchmidT;
    for (Int i{0}; i < NbSlip; ++i) {
      SchmidT[i] = this->Slip0.row(i).transpose() * this->Normal0.row(i);
    }

    // trial elastic deformation
    T2_t Fe_star{Floc*Fp.old().inverse()};
    T2_t CGe_star{Fe_star.transpose()*Fe_star};
    T2_t GLe_star{.5*(CGe_star - T2_t::Identity())};
    T2_t SPK_star{Matrices::tensmult(C_el,GLe_star)};

    using ColArray_t = Eigen::Array<Real, NbSlip, 1>;
    using ColMatrix_t = Eigen::Matrix<Real, NbSlip, 1>;
    ColArray_t tau_star;
    // pl_corr is the plastic corrector
    std::array<T2_t, NbSlip> pl_corr;
    using SlipMat_t = Eigen::Matrix<Real, NbSlip, NbSlip>;
    SlipMat_t pl_corr_proj;

    for (Int i{0}; i < NbSlip; ++i) {
      tau_star(i) = (CGe_star*SPK_star*SchmidT[i].transpose()).trace();
      pl_corr[i] = Matrices::tensmult(C_el,.5*(CGe_star*SchmidT[i]+SchmidT[i].transpose()*CGe_star));
      for (Int j{0}; j < NbSlip; ++j) {
        pl_corr_proj(i,j)=(Matrices::tensmult(C_el,pl_corr[i]*SchmidT[j].transpose())).trace();
      }
    }

    SlipMat_t I(SlipMat_t::Identity());
    auto && q_matrix{(I + this->q_n*(SlipMat_t::Ones() -I))};

    // residual on plastic slip rates
    Gammadot.current() = Gammadot.old();
    Tauy.current() = Tauy.old();
    ColArray_t tau_inc{tau_star};

    auto compute_gamma_dot = [this, &Gammadot, &tau_inc, &Tauy] () {
      return Gammadot.current().array()-this->gammadot_0*(abs(tau_inc).array()/Tauy.current().array()).pow(this->m_par)*sign(tau_inc.array()); };
    ColArray_t res{compute_gamma_dot()};

    auto compute_h_matrix = [this, &q_matrix] (const ColMatrix_t & Tauy_temp) {
      auto && parens =
      (ColMatrix_t::Ones()-Tauy_temp/this->s_infty).array()
      .pow(this->a_par).matrix();
      return this->h0*parens.asDiagonal()*q_matrix; };
    ColArray_t s_dot_old{(compute_h_matrix(Tauy.current())*Gammadot.old()).array()};

    SlipMat_t drdgammadot{SlipMat_t::Identity()};
    ColArray_t dr_stress{ColArray_t::Zero()};

    Int counter{};

    while (abs(res).maxCoeff()/this->gammadot_0 > tolerance){
      if(counter ++ > this->maxiter){
        throw std::runtime_error("Max. number of iteration for plastic slip reached without convergence");
      }

      dr_stress = abs(tau_inc).pow((1-this->m_par)/this->m_par)*Tauy.current().array().pow(-1/this->m_par)*sign(tau_inc);
      ColArray_t dr_hard{abs(tau_inc).pow(1/this->m_par)*Tauy.current().array().pow(-1-1/this->m_par)*sign(tau_inc)};
      drdgammadot = I+0.5*this->DeltaT*this->gammadot_0/this->m_par*dr_stress.matrix().asDiagonal()*pl_corr_proj.transpose()
        +0.5*this->DeltaT*this->gammadot_0/this->m_par*dr_hard.matrix().asDiagonal()*compute_h_matrix(Tauy.current())*Eigen::sign(Gammadot.current().array()).matrix().asDiagonal();
      Gammadot.current() -= drdgammadot.inverse() * res.matrix();
      // TODO: Check with Francesco whether the transposition of this guy is correct
      tau_inc = tau_star - (0.5*this->DeltaT*(Gammadot.current()+Gammadot.old()).transpose()*pl_corr_proj.transpose()).array().transpose();

      Int counter_h{};
      ColMatrix_t Tauy_temp{};
      do {
        if(counter_h ++ > this->maxiter){
          throw std::runtime_error("Max. number of iteration for hardening reached without convergence");
        }
        Tauy_temp = Tauy.current();
        Tauy.current() = Tauy.old() + 0.5*this->DeltaT*(s_dot_old.matrix() + compute_h_matrix(Tauy_temp)*Gammadot.current());
      } while ((Tauy.current() - Tauy_temp).norm() > tolerance);

      res = compute_gamma_dot();
    }

    T2_t SPK{SPK_star};
    for (Int i{0}; i < NbSlip; ++i) {
      SPK -= .5*DeltaT*(Gammadot.current()(i)+Gammadot.old()(i))*pl_corr[i];
    }

    T2_t Lp{T2_t::Zero()};
    for (Int i{0}; i < NbSlip; ++i) {
      Lp += .5*(Gammadot.current()(i)+Gammadot.old()(i))*SchmidT[i];
    }

    Fp.current() = (T2_t::Identity()+this->DeltaT*Lp)*Fp.old();

    T2_t PK2 = Rot.rotate_back(Fp.current().inverse()*SPK*Fp.current().inverse().transpose());

      // Stiffness matrix calculation begins

      // A4: elastic trial consistent tangent

    auto IRT = Matrices::Itrns<DimM>();
    auto I4 = Matrices::Iiden<DimM>();
    auto odot = [] (auto && T4, auto && T2) {
      T4_t Return_value(T4_t::Zero());
      for (Int i = 0; i < DimM; ++i) {
        for (Int j = 0; j < DimM; ++j) {
          for (Int k = 0; k < DimM; ++k) {
            for (Int l = 0; l < DimM; ++l) {
              for (Int m = 0; m < DimM; ++m) {
                get(Return_value,i,j,k,l) += get(T4,i,m,k,l)*T2(m,j);
              }
            }
          }
        }
      }
      return Return_value;
    };

    auto dot = [] (auto && a, auto && b) {return Matrices::dot<DimM>(a, b);};
    auto ddot = [] (auto && a, auto && b) {return Matrices::ddot<DimM>(a, b);};
    T4_t dAdF{odot(dot(Fp.old().inverse().transpose(),IRT),Fe_star)+odot(dot(Fe_star.transpose(),I4),Fp.old().inverse())};
    T4_t A4{.5*ddot(C_el,dAdF)};

    // E4: Tangent of the projector

    T4_t E4{T4_t::Zero()};
    for (Int i{0}; i < NbSlip; ++i) {
      T4_t dBprojdF{odot(dAdF,SchmidT[i]) + dot(SchmidT[i].transpose(),dAdF)};
      E4 -= .5*this->DeltaT*(Gammadot.current()(i)+Gammadot.old()(i))*ddot(C_el,dBprojdF);
    }

    // G4: Tangent of slip rate

    // dgammadot/dtau
    SlipMat_t dgammadotdtau{-drdgammadot.inverse()*(-this->gammadot_0/this->m_par*dr_stress.matrix().asDiagonal())};

    T4_t G4p1{T4_t::Zero()};

    for (Int k{0}; k < NbSlip; ++k) {
      for (Int mu{0}; mu < NbSlip; ++mu) {

        G4p1 += Matrices::outer(pl_corr[k],dgammadotdtau(k,mu)*SchmidT[mu]);

      }
    }

    T4_t G4p2{odot(dAdF,SPK)+dot(CGe_star,(A4+E4).eval())};
    T4_t G4_RHS{.5*DeltaT*G4p1*G4p2};

    auto xdot = [] (auto && T4, auto && T2) {
      T4_t Return_value(T4_t::Zero());
      for (Int i = 0; i < DimM; ++i) {
        for (Int j = 0; j < DimM; ++j) {
          for (Int k = 0; k < DimM; ++k) {
            for (Int l = 0; l < DimM; ++l) {
              for (Int m = 0; m < DimM; ++m) {
                get(Return_value,i,j,k,l) += get(T4,i,j,m,l)*T2(m,k);
              }
            }
          }
        }
      }
      return Return_value;
    };

    T4_t G4_LHS{I4-.5*DeltaT*xdot(G4p1,CGe_star)};

    T4_t G4{G4_LHS.inverse()*G4_RHS};

    // Plastic contribution to geometric tangent

    T4_t F4p1{T4_t::Zero()};
    T4_t F4p2{T4_t::Zero()};

    for (Int k{0}; k < NbSlip; ++k) {
      for (Int mu{0}; mu < NbSlip; ++mu) {

        F4p1 += Matrices::outer(SchmidT[k],dgammadotdtau(k,mu)*SchmidT[mu]);
        F4p2 += Matrices::outer(SchmidT[k].transpose(),dgammadotdtau(k,mu)*SchmidT[mu]);

      }
    }

    T4_t F4L{-.5*DeltaT*odot(dot(Fp.old(),(F4p1*(A4+E4+G4)).eval()),SPK*Fp.current().inverse().transpose())};
    T4_t F4R{-.5*DeltaT*(dot((Fp.current().inverse()*SPK).eval(),odot((F4p2*(A4+E4+G4)).eval(),Fp.old().inverse().transpose())))};

    T4_t K4{dot(F,Rot.rotate_back(F4L + odot(dot(Fp.current().inverse(),(A4+E4+G4).eval()),Fp.current().inverse().transpose())  + F4R))};

    return std::make_tuple(std::move(PK2), std::move(K4));

  }


}  // muSpectre

#endif /* MATERIAL_CRYSTAL_PLASTICITY_FINITE_H */