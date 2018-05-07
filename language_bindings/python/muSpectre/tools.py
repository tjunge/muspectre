#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   tools.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   20 Apr 2018

@brief  Toolbox for computing and plotting results of muSpectre computations.

@section LICENSE

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

import numpy as np
import itertools


def compute_displacements(F, resolutions, lengths, order=0):
    """
    Method to compute the real space displacements 'u' from the deformation
    gradient 'F' and the lengths 'lengths' of the system.

    Parameters
    ----------
    F: list
        Deformation gradient.
        For a three dimensional problem you need the following input form:
        F=[i,j,x,y,z]; F.shape=[3,3,nx,ny,nz]
    resolutions: list
        Grid resolutions in the Cartesian directions.
        For a three dimensional problem you need the following input form:
        [nx, ny, nz]
    lengths: list
        Physical size of the cell in the Cartesian directions.
        For a three dimensional problem you need the following input form:
        [Lx, Ly, Lz]
    order: integer
        Order of the corrections to the Fourier transformed of the derivative.
        This is necessary because of finite grid spacing and hence errors in the
        computation of the finite difference derivative.
        Corrections are done as in:
        A. Vidyasagar et al., Computer Methods in Applied Mechanics and
        Engineering 106 (2017) 133-151, sec. 3.4 and Appendix C.
        available: order = {0, 1, 2}

    Returns
    -------
    displacements: list
        Displacement vectors 'u' pointing from the initial 'X' to the final
        position 'U' of each gridpoint. To get the displacements 'u' one has to
        integrate F=du/dX by a fourier transformation.
    final_positions: list
        Positions 'U' after applying the deformation, according to the
        deformation gradient F, to the initial positions 'X'. U=u+<F>*X.

    Coution!
    --------
        Up to now only for 3D grids implemented!

    TODO:
    -----
        --> implementation for 2D grid, (1D grid)
        --> implement higher order corrections (select order in function),
            is this necessary?
        --> should we implement that one can give the initial coordinates 'X',
            if the body is already deformed.
            What happens if the grid is not cubic/orthogonal?
        --> prevent error messages from dividing by zero in F/(i*q)
    """
    print('\n\ntools output\n')
    #for 3D grids...

    F    = np.array(F)    #deformation gradient
    res  = np.array(resolutions)
    lens = np.array(lengths)
    ndim = len(res)       #dimension of the problem
    Nx, Ny, Nz = res      #number of gridpoints in each direction
    Lx, Ly, Lz = lens     #lengths in each direction
    dx, dy, dz = lens/res #gridspacing in each direction

    print('gridpoints:   ', res)
    print('lengths:      ', lens)
    print('grid spacing: ', lens/res)

    ### Initialize the undeformed positions and q-vectors
    #TODO: write this shorter/faster
    #      prevent unnecessary increas in dimension of q and X33NNN
    X = np.zeros((ndim, Nx, Ny, Nz))
    for i,j,k in itertools.product(range(Nx), range(Ny), range(Nz)):
        X[:,i,j,k] = np.array([i*dx, j*dy, k*dz])

    print('positions:\n', X[:,25,20,10])

    #increase dim of X for more easy multiplication with F (do this better)
    X_33NNN = np.tensordot(np.array([1,1,1]), X , axes=0)

    print('positions tensor:\n', X_33NNN[:,:,25,20,10])

    # calculate the q vectors and bring them in a (ndim,ndim,Nx,Ny,Nz) shape
    q = np.zeros((ndim, ndim, Nx, Ny, Nz))
    qx_1d = (np.fft.fftfreq(Nx, d=dx))*(2*np.pi)
    qy_1d = (np.fft.fftfreq(Ny, d=dy))*(2*np.pi)
    qz_1d = (np.fft.fftfreq(Nz, d=dz))*(2*np.pi)
    for i,j,k in itertools.product(range(Nx), range(Ny), range(Nz)):
        q[:,:,i,j,k] = np.tensordot(np.array([1,1,1]),
                                    np.array([qx_1d[i], qy_1d[j], qz_1d[k]]),
                                    axes=0)

    print('q-vector:\n', q[:,:,:,20,10])

    # calculate <F>
    # F_av_(ij,klm) = 1/N^3 * Sum_(a,b,c=0)^(N-1){F_ij,abc} * I_(ij,klm)
    # TODO: proof if you can use F_av=F_q(q=0)
    F_av = np.einsum('ij,klm', 1.0/(Nx*Ny*Nz) * F.sum(axis=(2,3,4)),
                     np.ones([Nx,Ny,Nz]))

    print('shape F ', F.shape)
    F_q = np.fft.fftn(F, [Nx,Ny,Nz]) #F_q = DFT(F)
    print('shape F_q ', F_q.shape)
    print(F_q[:,:,0,0,0]*1/(Nx*Ny*Nz))
    print(F_av[:,:,0,0,0])

    ### Calculate u_q from F_q; u_q = F_q/(j*q)
    # u'_i=1/3 * { F'_xi/(j*q_i) + F'_yi/(j*q_i) + F'_zi/(j*q_i) }
    # Note:
    # except for u_q[:,:,0,0,0] one always can read off problematic entries
    # (*/0) along axis=1. Therefore one can reconstruct all values of
    # u_q[:,:,k,l,m] for [k,l,m]!=[0,0,0]

    ### Corrections to the Fourier transformed of the derivative ###
    # Necessary because of finite grid spacing and hence errors in the
    # computation of the finite difference derivative.
    #
    # A. Vidyasagar et al., Computer Methods in Applied Mechanics and
    # Engineering 106 (2017) 133-151, sec. 3.4 and Appendix C.

    if order == 0:
        #zeroth order
        u_q = (F_q / (1j*q))
    elif order == 1:
        #first oder correction
        u_q = (F_q / (1j * np.sin(q*dx)/dx))
    elif order == 2:
        #second oder correction
        u_q = (F_q / (1j * (8*np.sin(q*dx)/(6*dx) - np.sin(2*q*dx)/(6*dx) )))
    else:
        print('\n\n   WARNING!   \n')
        print('The order {} is not supported. Up to now only order={{0,1,2}} is'
              ' supported.'.format(order))
        print('I switch back to default, order = 0.\n\n')
        u_q = (F_q / (1j*q))


    print('u_q before correction:\n', u_q[:,:,41,31,18])

    ## correct the problematic u_q(q_i=0) by setting them:
    #   a) to zero ('0./0 = 0') for u_q[:,:,0,0,0]
    u_q[:,:,0,0,0] = np.zeros([3,3], dtype=complex )
    #   b) reconstruct them from values in F(axis=1)
    u_q[:,0, 0 , 1:, : ] = u_q[:,1, 0 , 1:, : ]   # k=0 correct with l=1...(N-1)
    u_q[:,0, 0 , 0 , : ] = u_q[:,2, 0 , 0 , : ]   # correct for k=l=0
    u_q[:,1, : , 0 , 1:] = u_q[:,2, : , 0 , 1:]   # l=0 correct with m=1...(N-1)
    u_q[:,1, : , 0 , 0 ] = u_q[:,0, : , 0 , 0 ]   # correct for l=m=0
    u_q[:,2, 1:, : , 0 ] = u_q[:,0, 1:, : , 0 ]   # m=0 correct with k=1...(N-1)
    u_q[:,2, 0 , : , 0 ] = u_q[:,1, 0 , : , 0 ]   # correct for m=k=0

    print('u_q after correction:\n', u_q[:,:,41,31,18])

    # get u by inverse fourier transform u_q; u=IFFT(u_q)
    # take the real values and average over axis=1 (which are all the same entries)
    u = (1./3.) * np.real(np.fft.ifftn(u_q, [Nx,Ny,Nz])).sum(axis=1)

    print('u final:\n', u[:,41,31,18])

    # calculate U = u + <F>*X
    fx = (F_av * X_33NNN).sum(axis=1) #fx = <F>*X
    U  = u + fx

    print('final_positions:\n', U[:,41,31,18])

    #give mor meaningfull names to the results
    displacements   = u
    final_positions = U

    return displacements, final_positions


def write_structure_as_vtk(file_name, positions, cellData=None, pointData=None):
    """
    Function to save a structure as vtk file by using pyevtk.hl.
    The file is named file_name.vtk.

    Parameters
    ----------
    file_name: str
        name of the 'vtk' file which will be writen
    positions: list
        grid point positions
    cellData: dictionary
        data associated with voxel properties
    pointData: dictionary
        data associated to each point/point properties

    Returns
    -------
    writes a '.vtk' file with the file name, file_name.vtk
    returns str 'file was writen'

    TODO
    ----
        --> write more about the function...
    """
    from pyevtk.hl import gridToVTK

    cdata = cellData
    pdata = pointData
    x = positions[0,:,:,:].flatten()
    print('x\n', x, x.shape)
    y = positions[1,:,:,:].flatten()
    print('y\n', y, y.shape)
    z = positions[2,:,:,:].flatten()
    print('z\n', z, z.shape)
    gridToVTK(file_name, x, y, z, cellData=cdata, pointData=pdata)

    return 'file was writen'
