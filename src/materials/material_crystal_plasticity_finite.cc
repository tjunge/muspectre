/**
 * @file   material_crystal_plasticity_finite.cc
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

#include "materials/material_crystal_plasticity_finite.hh"
#include "common/iterators.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  MaterialCrystalPlasticityFinite(std::string name, Real bulk_m, Real shear_m, Real gamma_dot0, Real m_par, Real tau_y0, Real h0, Real delta_tau_y_max, Real a_par, Real q_n, SlipVecs_ref Slip0, SlipVecs_ref Normal0, Real delta_t, Real tolerance, Int maxiter)
    : Parent{name},
      FpField("Plastic Deformation Gradient Fₚ(t)",this->internal_fields),
      GammaDotField("Plastic slip rates dγᵅ/dt",this->internal_fields),
      TauYField("Critical resolved shear stress τᵅy(t)",this->internal_fields),
      GammaField{
        make_field<MatrixField<LColl_t, Real, NbSlip, 1>>
          ("Accumulated slips γᵅ(t)",this->internal_fields)},
      EulerField{
        make_field<MatrixField<LColl_t, Real, NbEuler, 1>>
          ("Euler angles", this->internal_fields)},
      bulk_m{bulk_m}, shear_m{shear_m}, gamma_dot0{gamma_dot0}, m_par{m_par},
      tau_y0{tau_y0}, h0{h0}, tau_infty{tau_y0+delta_tau_y_max}, a_par{a_par}, q_n{q_n},
      delta_t{delta_t}, tolerance{tolerance}, maxiter{maxiter}, Slip0{Slip0},
      Normal0{Normal0},
      dummy_gamma_dot{
        make_field<MatrixField<LColl_t, Real, NbSlip, 1>>
          ("dummy γ_dot", this->internal_fields)},
      dummy_tau_inc{
        make_field<MatrixField<LColl_t, Real, NbSlip, 1>>
          ("dummy τ_inc", this->internal_fields)},
      internal_variables{FpField.get_map(),
          ScalarPerSlip_map(GammaDotField),
          ScalarPerSlip_map(TauYField),
          ArrayFieldMap<LColl_t, Real, NbEuler, 1, true>(EulerField),
          MatrixFieldMap<LColl_t, Real, NbSlip, 1>(dummy_gamma_dot),
          MatrixFieldMap<LColl_t, Real, NbSlip, 1>(dummy_tau_inc)}
  {
    // Enforce n₀ and s₀ to be unit vectors!
    auto lambda{MatTB::convert_elastic_modulus<ElasticModulus::lambda,
                                               ElasticModulus::Bulk,
                                               ElasticModulus::Shear>(bulk_m,shear_m)};
    this->C_el=Hooke::compute_C_T4(lambda,shear_m);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  void MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  add_pixel(const Ccoord_t<DimS> & /*pixel*/) {
    throw std::runtime_error
      ("this material needs pixels with a column vector of Euler angles");
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  void MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  add_pixel(const Ccoord_t<DimS> & pixel,
	    const Eigen::Ref<Eigen::Array<Real, NbEuler, 1>> Euler){
    this->internal_fields.add_pixel(pixel);
    this->EulerField.push_back(Euler);
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  void MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  initialise() {
    if (not this->is_initialised) {
      Parent::initialise();
      using T2_t = Eigen::Matrix<Real, DimM, DimM>;
      this->FpField.get_map().current() = T2_t::Identity();

      using ColArray_t = Eigen::Matrix<Real, NbSlip, 1>;
      ScalarPerSlip_map(this->GammaDotField).current() = ColArray_t::Zero();
      ScalarPerSlip_map(this->TauYField).current() = ColArray_t::Constant(this->tau_y0);

      this->GammaField.set_zero();

      this->FpField.cycle();
      this->GammaDotField.cycle();
      this->TauYField.cycle();

    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  void MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  save_history_variables() {
    auto GammaMap = this->GammaField.get_map();
    auto GammaDotMap = this->GammaDotField.get_map();

    for (auto && tup: akantu::zip(GammaMap, GammaDotMap)) {
      auto & gamma = std::get<0>(tup);
      auto & gamma_dot = std::get<1>(tup);
      gamma += .5 * this->delta_t * (gamma_dot.current() + gamma_dot.old());
    }

    this->FpField.cycle();
    this->GammaDotField.cycle();
    this->TauYField.cycle();

  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, Int NbSlip>
  auto
  MaterialCrystalPlasticityFinite<DimS, DimM, NbSlip>::
  evaluate_stress_tangent(const Eigen::Ref<const T2_t>& F, Fp_ref Fp,
                          GammaDot_ref gamma_dot,
                          TauY_ref tau_y, Euler_ref Euler,
                          Dummy_ref dummy_gamma_dot, Dummy_ref dummy_tau_inc
                          ) -> std::tuple<T2_t, T4_t> {
    auto dot = [] (auto && a, auto && b) {return Matrices::dot<DimM>(a, b);};
    auto ddot = [] (auto && a, auto && b) {return Matrices::ddot<DimM>(a, b);};

    Rotator<DimM> Rot(Euler);
    T2_t Floc{Rot.rotate(F)};
    std::array<T2_t, NbSlip> SchmidT;
    for (Int i{0}; i < NbSlip; ++i) {
      SchmidT[i] = this->Slip0.row(i).transpose() * this->Normal0.row(i);
    }

    // trial elastic deformation
    T2_t Fe_star{Floc*Fp.old().inverse()};
    T2_t CGe_star{Fe_star.transpose()*Fe_star}; // elastic Cauchy-Green strain
    T2_t GLe_star{.5*(CGe_star - T2_t::Identity())};
    T2_t SPK_star{Matrices::tensmult(C_el,GLe_star)};

    using ColArray_t = Eigen::Array<Real, NbSlip, 1>;
    using ColMatrix_t = Eigen::Matrix<Real, NbSlip, 1>;
    ColArray_t tau_star;
    // pl_corr is the plastic corrector (Bᵅ)
    std::array<T2_t, NbSlip> pl_corr;
    using SlipMat_t = Eigen::Matrix<Real, NbSlip, NbSlip>;
    SlipMat_t pl_corr_proj;

    for (Int i{0}; i < NbSlip; ++i) {
      // first half of eq (19)
      tau_star(i) = (CGe_star*SPK_star*SchmidT[i].transpose()).trace();

      // eq (19)
      pl_corr[i] = Matrices::tensmult(C_el,.5*(CGe_star*SchmidT[i]+SchmidT[i].transpose()*CGe_star));
      for (Int j{0}; j < NbSlip; ++j) {
        pl_corr_proj(i,j)= ddot(CGe_star*pl_corr[i], SchmidT[j]);
      }
    }

    SlipMat_t I(SlipMat_t::Identity());
    auto && q_matrix{(I + this->q_n*(SlipMat_t::Ones() -I))};

    // residual on plastic slip rates
    gamma_dot.current() = gamma_dot.old();
    tau_y.current() = tau_y.old();
    ColArray_t tau_inc{tau_star};

    auto objective_fun = [this, &gamma_dot, &tau_inc, &tau_y] () {
      return (gamma_dot.current().array() -
              this->gamma_dot0*(abs(tau_inc).array()/tau_y.current().array()).pow(1./this->m_par)*sign(tau_inc.array()));
    };
    ColArray_t res{objective_fun()};

    auto compute_h_matrix = [this, &q_matrix] (const ColMatrix_t & tau_y_temp) {
      auto && parens =
      (ColMatrix_t::Ones()-tau_y_temp/this->tau_infty).array()
      .pow(this->a_par).matrix();
      return this->h0*parens.asDiagonal()*q_matrix; };


    ColArray_t s_dot_old{(compute_h_matrix(tau_y.current())*gamma_dot.old()).array()};

    SlipMat_t drdgammadot{SlipMat_t::Identity()};
    ColArray_t dr_stress{ColArray_t::Zero()};

    Int counter{};

    while (abs(res).maxCoeff()/this->gamma_dot0 > tolerance){
      if(counter ++ > this->maxiter){
        throw std::runtime_error("Max. number of iteration for plastic slip reached without convergence");
      }

      // in eq (27), term |τᵅ(tₙ₊₁)|^(¹⁻ᵐ/ₘ)·(sᵃ(tₙ₊₁))^(-¹/ₘ)·sgn(τᵅ(tₙ₊₁))
      dr_stress = abs(tau_inc).pow((1-this->m_par)/this->m_par)*tau_y.current().array().pow(-1/this->m_par)*sign(tau_inc);

      // in eq (27), term |τᵅ(tₙ₊₁)|^(¹/ₘ)·(sᵅ(tₙ₊₁))^(-1-¹/ₘ)·sgn(τᵅ(tₙ₊₁))
      ColArray_t dr_hard{abs(tau_inc).pow(1/this->m_par)*tau_y.current().array().pow(-1-1/this->m_par)*sign(tau_inc)};

      //! eq (27)?
      drdgammadot = ((I + 0.5*this->delta_t * this->gamma_dot0/this->m_par *
                      dr_stress.matrix().asDiagonal()*pl_corr_proj.transpose()) +

                     (0.5 * this->delta_t * this->gamma_dot0/this->m_par *
                      dr_hard.matrix().asDiagonal() *
                      compute_h_matrix(tau_y.current()) *
                      Eigen::sign(gamma_dot.current().array()).matrix().asDiagonal()));
      gamma_dot.current() -= drdgammadot.inverse() * res.matrix();

      tau_inc = tau_star -
        (0.5*this->delta_t*(gamma_dot.current() + gamma_dot.old()).transpose() *
         pl_corr_proj).array().transpose();

      {
        Int counter_h{};
        ColMatrix_t tau_y_temp{};
        Real tau_y_residual{};
        do {
          if(counter_h ++ > this->maxiter){
            throw std::runtime_error("Max. number of iteration for hardening reached without convergence");
          }
          tau_y_temp = tau_y.current();
          tau_y.current() = tau_y.old() + 0.5*this->delta_t*(s_dot_old.matrix() + compute_h_matrix(tau_y_temp)*gamma_dot.current());
          //! // TODO: rediscuss with francesco why the second version is wrong (commented one)
          //tau_y.current() += 0.5*this->delta_t*(s_dot_old.matrix() + compute_h_matrix(tau_y_temp)*gamma_dot.current());
          tau_y_residual = (tau_y.current() - tau_y_temp).norm()/(tau_y.current().norm() + 1);
        } while (tau_y_residual > tolerance);
      }

      res = objective_fun();
    }
    dummy_gamma_dot = gamma_dot.current();
    dummy_tau_inc = tau_inc;
    T2_t SPK{SPK_star};
    for (Int i{0}; i < NbSlip; ++i) {
      SPK -= .5*delta_t*(gamma_dot.current()(i)+gamma_dot.old()(i))*pl_corr[i];
    }

    T2_t Lp{T2_t::Zero()};
    for (Int i{0}; i < NbSlip; ++i) {
      Lp += .5*(gamma_dot.current()(i)+gamma_dot.old()(i))*SchmidT[i];
    }

    Fp.current() = (T2_t::Identity()+this->delta_t*Lp)*Fp.old();

    T2_t PK2 = Rot.rotate_back(Fp.current().inverse()*SPK*Fp.current().inverse().transpose());

      // Stiffness matrix calculation begins

      // A4: elastic trial consistent tangent

    auto IRT = Matrices::Itrns<DimM>();
    auto I4 = Matrices::Iiden<DimM>();
    auto odot = [] (auto && T4, auto && T2) {
      T4_t ret_val(T4_t::Zero());
      for (Int i = 0; i < DimM; ++i) {
        for (Int j = 0; j < DimM; ++j) {
          for (Int k = 0; k < DimM; ++k) {
            for (Int l = 0; l < DimM; ++l) {
              for (Int m = 0; m < DimM; ++m) {
                get(ret_val,i,j,k,l) += get(T4,i,m,k,l)*T2(m,j);
              }
            }
          }
        }
      }
      return ret_val;
    };
    T4_t dAdF1{odot(dot(Fp.old().inverse().transpose(),IRT),Fe_star)};
    T4_t dAdF2{odot(dot(Fe_star.transpose(),I4),Fp.old().inverse()) };
    T4_t dAdF{dAdF1 + dAdF2};
    T4_t A4{.5*ddot(C_el,dAdF)};

    // E4: Tangent of the projector

    T4_t E4{T4_t::Zero()};
    for (Int i{0}; i < NbSlip; ++i) {
      T4_t dBprojdF{odot(dAdF,SchmidT[i]) + dot(SchmidT[i].transpose(),dAdF)};
      E4 -= .5*this->delta_t*(gamma_dot.current()(i)+gamma_dot.old()(i))*ddot(C_el,dBprojdF);
    }

    // G4: Tangent of slip rate

    // dgammadot/dtau
    SlipMat_t dgammadotdtau{-drdgammadot.inverse()*(-this->gamma_dot0/this->m_par*dr_stress.matrix().asDiagonal())};

    T4_t G4p1{T4_t::Zero()};

    for (Int k{0}; k < NbSlip; ++k) {
      for (Int mu{0}; mu < NbSlip; ++mu) {

        G4p1 += Matrices::outer(pl_corr[k],dgammadotdtau(k,mu)*SchmidT[mu]);

      }
    }

    T4_t G4p2{odot(dAdF,SPK)+dot(CGe_star,(A4+E4).eval())};
    T4_t G4_RHS{.5*delta_t*G4p1*G4p2};

    auto xdot = [] (auto && T4, auto && T2) {
      T4_t ret_val(T4_t::Zero());
      for (Int i = 0; i < DimM; ++i) {
        for (Int j = 0; j < DimM; ++j) {
          for (Int k = 0; k < DimM; ++k) {
            for (Int l = 0; l < DimM; ++l) {
              for (Int m = 0; m < DimM; ++m) {
                get(ret_val,i,j,k,l) += get(T4,i,j,m,l)*T2(m,k);
              }
            }
          }
        }
      }
      return ret_val;
    };

    T4_t G4_LHS{I4-.5*delta_t*xdot(G4p1,CGe_star)};

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

    T4_t F4L{-.5*delta_t*odot(dot(Fp.old(),(F4p1*(A4+E4+G4)).eval()),SPK*Fp.current().inverse().transpose())};
    T4_t F4R{-.5*delta_t*(dot((Fp.current().inverse()*SPK).eval(),odot((F4p2*(A4+E4+G4)).eval(),Fp.old().inverse().transpose())))};

    T4_t K4{Rot.rotate_back(F4L + odot(dot(Fp.current().inverse(),(A4+E4+G4).eval()),Fp.current().inverse().transpose())  + F4R)};

    return std::make_tuple(std::move(PK2), std::move(K4));

  }


  /* ---------------------------------------------------------------------- */
  template class MaterialCrystalPlasticityFinite<  twoD,   twoD,  3>;
  template class MaterialCrystalPlasticityFinite<  twoD, threeD, 12>;
  template class MaterialCrystalPlasticityFinite<threeD, threeD, 12>;

  // toy materials with a single slip system for testing
  template class MaterialCrystalPlasticityFinite<  twoD,   twoD,  1>;

} // muSpectre
