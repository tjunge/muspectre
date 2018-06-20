#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_exact_reference_test.py

@author Till Junge <till.junge@epfl.ch>

@date   18 Jun 2018

@brief  Tests exactness of each iterate with respect to python reference
        implementation from GooseFFT

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import unittest
import numpy as np
from numpy.linalg import norm
from python_test_imports import µ

import scipy.sparse.linalg as sp
import itertools

# ----------------------------------- GRID ------------------------------------

ndim   = 3   # number of dimensions
N      = 3  # number of voxels (assumed equal for all directions)

Nx = Ny = Nz = N


def deserialise_t4(t4):
    turnaroud = np.arange(ndim**2).reshape(ndim,ndim).T.reshape(-1)
    retval = np.zeros([ndim*ndim, ndim*ndim])
    for i,j in itertools.product(range(ndim**2), repeat=2):
        retval[i,j] = t4[:ndim, :ndim, :ndim, :ndim, 0,0].reshape(ndim**2, ndim**2)[turnaroud[i], turnaroud[j]]
        pass
    return retval

def t2_to_goose(t2_msp):
    t2_goose = np.zeros((ndim, ndim, Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                t2_goose[:,:,i,j,k] = t2_msp[:, i + Nx*j + Nx*Ny*k].reshape(ndim, ndim).T
            pass
        pass
    return t2_goose

def t2_vec_to_goose(t2_msp_vec):
    return t2_to_goose(t2_msp_vec.reshape(ndim*ndim, Nx*Ny*Nz)).reshape(-1)

def t4_to_goose(t4_msp):
    t4_goose = np.zeros((ndim, ndim, ndim, ndim, Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                tmp = t4_msp[:, i + Nx*j + Nx*Ny*k]

                t4_goose[:,:,:,:,i,j,k] = t4_msp[:, i + Nx*j + Nx*Ny*k].reshape(
                    ndim, ndim, ndim, ndim).T
            pass
        pass
    return t4_goose

def t4_vec_to_goose(t4_msp_vec):
    return t4_to_goose(t4_msp_vec.reshape(ndim**4, Nx*Ny*Nz)).reshape(-1)

def t2_from_goose(t2_goose):
    nb_pix = Nx*Ny*Nz
    t2_msp = np.zeros((ndim**2, nb_pix), order='F')
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                view = t2_msp[:, i + Nx*j + Nx*Ny*k].reshape(ndim, ndim).T
                view = t2goose[:,:,i,j,k].T
            pass
        pass
    return t2_msp


# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
trans2 = lambda A2   : np.einsum('ijxyz          ->jixyz  ',A2   )
ddot42 = lambda A4,B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijklxyz,lkmnxyz->ijmnxyz',A4,B4)
dot22  = lambda A2,B2: np.einsum('ijxyz  ,jkxyz  ->ikxyz  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ijxyz  ,jkmnxyz->ikmnxyz',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijklxyz,lmxyz  ->ijkmxyz',A4,B2)
dyad22 = lambda A2,B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz',A2,B2)

# identity tensor                                               [single tensor]
i      = np.eye(ndim)
# identity tensors                                            [grid of tensors]
I      = np.einsum('ij,xyz'           ,                  i   ,np.ones([N,N,N]))
I4     = np.einsum('ijkl,xyz->ijklxyz',np.einsum('il,jk',i,i),np.ones([N,N,N]))
I4rt   = np.einsum('ijkl,xyz->ijklxyz',np.einsum('ik,jl',i,i),np.ones([N,N,N]))
I4s    = (I4+I4rt)/2.
II     = dyad22(I,I)

# projection operator                                         [grid of tensors]
# NB can be vectorized (faster, less readable), see: "elasto-plasticity.py"
# - support function / look-up list / zero initialize
delta  = lambda i,j: np.float(i==j)            # Dirac delta function
freq   = np.arange(-(N-1)/2.,+(N+1)/2.)        # coordinate axis -> freq. axis
Ghat4  = np.zeros([ndim,ndim,ndim,ndim,N,N,N]) # zero initialize
# - compute
for i,j,l,m in itertools.product(range(ndim),repeat=4):
    for x,y,z    in itertools.product(range(N),   repeat=3):
        q = np.array([freq[x], freq[y], freq[z]])  # frequency vector
        if not q.dot(q) == 0:                      # zero freq. -> mean
            Ghat4[i,j,l,m,x,y,z] = delta(i,m)*q[j]*q[l]/(q.dot(q))

# (inverse) Fourier transform (for each tensor component in each direction)
fft    = lambda x  : np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[N,N,N]))
ifft   = lambda x  : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[N,N,N]))

# functions for the projection 'G', and the product 'G : K^LT : (delta F)^T'
G      = lambda A2 : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
K_dF   = lambda dFm: trans2(ddot42(K4,trans2(dFm.reshape(ndim,ndim,N,N,N))))
G_K_dF = lambda dFm: G(K_dF(dFm))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# phase indicator: cubical inclusion of volume fraction (9**3)/(31**3)
phase  = np.zeros([N,N,N]); phase[:2,:2,:2] = 1.
# material parameters + function to convert to grid of scalars
param  = lambda M0,M1: M0*np.ones([N,N,N])*(1.-phase)+M1*np.ones([N,N,N])*phase
K      = param(0.833,8.33)  # bulk  modulus                   [grid of scalars]
mu     = param(0.386,3.86)  # shear modulus                   [grid of scalars]

# constitutive model: grid of "F" -> grid of "P", "K4"        [grid of tensors]
def constitutive(F):
    C4 = K*II+2.*mu*(I4s-1./3.*II)
    S  = ddot42(C4,.5*(dot22(trans2(F),F)-I))
    P  = dot22(F,S)
    K4 = dot24(S,I4)+ddot44(ddot44(I4rt,dot42(dot24(F,C4),trans2(F))),I4rt)
    return P,K4


F     = np.array(I,copy=True)
P,K4  = constitutive(F)



class LinearElastic_Check(unittest.TestCase):
    def setUp(self):
        #---------------------------- µSpectre init -----------------------------------
        resolution = list(phase.shape)
        dim = len(resolution)
        self.dim=dim

        center = np.array([r//2 for r in resolution])
        incl = resolution[0]//5


        ## Domain dimensions
        lengths = [float(r) for r in resolution]
        ## formulation (small_strain or finite_strain)
        formulation = µ.Formulation.finite_strain

        ## build a computational domain
        self.rve = µ.Cell(resolution, lengths, formulation)
        def get_E_nu(bulk, shear):
            Young = 9*bulk*shear/(3*bulk + shear)
            Poisson = Young/(2*shear) - 1
            return Young, Poisson

        mat = µ.material.MaterialLinearElastic1_3d

        E, nu = get_E_nu(.833, .386)
        hard = mat.make(self.rve, 'hard', 10*E, nu)
        soft = mat.make(self.rve, 'soft',    E, nu)

        for pixel in self.rve:
            if pixel[0] < 2 and pixel[1] < 2 and pixel[2] < 2:
                hard.add_pixel(pixel)
            else:
                soft.add_pixel(pixel)

    def test_solve(self):
        tol = 1e-12
        # ----------------------------- NEWTON ITERATIONS ---------------------

        # initialize deformation gradient, and stress/stiffness [tensor grid]
        global K4, P, F
        F     = np.array(I,copy=True)
        F2    = np.array(I,copy=True)*1.1
        P2,K42  = constitutive(F2)
        P,K4  = constitutive(F)
        self.rve.set_uniform_strain(np.array(np.eye(ndim)))
        µF = self.rve.get_strain()

        self.assertLess(norm(t2_vec_to_goose(µF) - F.reshape(-1))/norm(F), tol)
        # set macroscopic loading
        DbarF = np.zeros([ndim,ndim,N,N,N]); DbarF[0,1] += 1.0

        # initial residual: distribute "barF" over grid using "K4"
        b     = -G_K_dF(DbarF)
        F    +=         DbarF
        Fn    = np.linalg.norm(F)
        iiter = 0

        # µSpectre inits
        µbarF    = np.zeros_like(µF)
        µbarF[ndim, :] += 1.
        µF2 = µF.copy()*1.1
        µP2, µK2 = self.rve.evaluate_stress_tangent(µF2)
        err = norm(t2_vec_to_goose(µP2) - P2.reshape(-1))/norm(P2)
        def t2_comparator(µT2, gT2):
            err_sum = 0.
            err_max = 0.
            for counter, (i, j, k) in enumerate(self.rve):
                print((i,j,k))
                µ_arr = µT2[:, counter].reshape(ndim, ndim).T
                g_arr = gT2[:,:,i,j,k]
                print(µ_arr)
                print(g_arr)
                print(µ_arr-g_arr)
                err = norm(µ_arr-g_arr)
                print("error norm = {}".format(err))
                err_sum += err_max
                err_max = max(err_max, err)
                pass
            print("∑(err) = {}, max(err) = {}".format (err_sum, err_max))
            return 

        if not (err < tol):
            t2_comparator(µP2, µK2)
        self.assertLess(err, tol)
        self.rve.set_uniform_strain(np.array(np.eye(ndim)))
        µP, µK = self.rve.evaluate_stress_tangent(µF)
        err = norm(t2_vec_to_goose(µP) - P.reshape(-1))
        if not (err < tol):
            print(µF)
            t2_comparator(µP, P)
        self.assertLess(err, tol)
        err = norm(t4_vec_to_goose(µK) - K4.reshape(-1))/norm(K4)
        if not (err < tol):
            print ("err = {}".format(err))

        self.assertLess(err, tol)
        µF += µbarF
        self.assertLess(norm(t2_vec_to_goose(µF) - F.reshape(-1))/norm(F), tol)
        µG_K_dF = lambda x: self.rve.directional_stiffness(x.reshape(µF.shape)).reshape(-1)
        µG = lambda x: self.rve.project(x).reshape(-1)
        µb = -µG_K_dF(µbarF)


        print("|µb| = {}".format(norm(µb)))
        print("|b| = {}".format(norm(b)))
        err = (norm(t2_vec_to_goose(µb.reshape(µF.shape)) - b) /
               norm(b))
        if not (err < tol):
            print("total error = {}".format(err))
            t2_comparator(µb.reshape(µF.shape), b.reshape(F.shape))
        self.assertLess(err, tol)

        # iterate as long as the iterative update does not vanish
        while True:
            # solve linear system using CG
            dFm,_ = sp.cg(tol=1.e-8,
                          A = sp.LinearOperator(shape=(F.size,F.size),
                                                matvec=G_K_dF,dtype='float'),
                          b = b,
            )

            µdFm,_ = sp.cg(tol=1.e-8,
                         A =  sp.LinearOperator(shape=(F.size,F.size),
                                                matvec=µG_K_dF,dtype='float'),
                         b = µb)

            print("µdFm.shape = {}".format(µdFm.shape))
            print("|µdFm| = {}".format(norm(µdFm)))
            print("|dFm| = {}".format(norm(dFm)))
            self.assertLess(norm(t2_vec_to_goose(µdFm) - dFm)/norm(dFm), tol)
            # update DOFs (array -> tens.grid)
            F    += dFm.reshape(ndim,ndim,N,N,N)
            # new residual stress and tangent
            P,K4  = constitutive(F)
            # convert res.stress to residual
            b     = -G(P)
            # print residual to the screen
            print('%10.2e'%(np.linalg.norm(dFm)/Fn))
            if np.linalg.norm(dFm)/Fn<1.e-5 and iiter>0: break # check convergence
            iiter += 1


if __name__ == '__main__':
    unittest.main()
