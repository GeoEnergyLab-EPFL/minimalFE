#
# This file is part of minimalFE.
#
# Created by Brice Lecampion on 04.05.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

import numpy as np
from numba import jit

# FUNCTIONS Utilities to Handle some unstructured mesh computations

@jit(nopython=True)
def project_surface_elt(xae3D):
    # this function projects the coordinates of a 3D planar surface element (i.e. a triangle) into its plane
    # and return
    R = surface_elt_local(xae3D)
    xae_proj=xae3D-xae3D[0]
    for i in range(3) :
        xae_proj[i] = np.dot(R,xae_proj[i])
    return xae_proj[:,0:2]

@jit(nopython=True)
def surface_elt_local(xae3D):
    # this function returns the local cartesian frame of a 3D planar surface element
    # as a rotation matrix [tangent_1, tangent_2, normal]
    # xae3D has the coordinates of the vertex (we only take the first 3 vertex assuming a triangle)
    # vertex1 = xae3D[0,:], vertex2=xae3D[1,:]
    tangent_1=xae3D[1]-xae3D[0]
    tangent_1=tangent_1/(np.linalg.norm(tangent_1))
    normal =np.cross(tangent_1,xae3D[2]-xae3D[0])
    normal = normal/(np.linalg.norm(normal))
    tangent_2= np.cross(normal,tangent_1)
    res = np.empty((3, 3)) # Instantiating NumPy arrays via a list of NumPy arrays, or even a list of lists, is not supported by numba.njit. Instead, use np.empty and then assign values via NumPy indexing
    res[0]=tangent_1
    res[1]=tangent_2
    res[2]=normal
    return res

#myxx=np.array([[0.,0.,0.],[0.,0.,1.],[0.,1.,0.]])

# 2D segment

@jit(nopython=True)
def project_segment_elt(xae2D):
    # this function projects the coordinates of a 3D planar surface element (i.e. a triangle) into its plane
    # and return
    R = segment_elt_local(xae2D)
    xae_proj=xae2D-xae2D[0]
    for i in range(2) :
        xae_proj[i] = np.dot(R,xae_proj[i])
    return xae_proj[:,0]


@jit(nopython=True)
def segment_elt_local(xae2D):
    # tangent vector
    tangent=xae2D[1]-xae2D[0]
    tangent = tangent / (np.linalg.norm(tangent))
    # normal
    normal=np.array([-tangent[1],tangent[0]])
    res = np.empty((2, 2))
    res[0]=tangent
    res[1]=normal
    return res
