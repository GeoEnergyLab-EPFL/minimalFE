#
# This file is part of minimalFE.
#
# Created by Brice Lecampion on 07.02.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

import numpy as np
from numba import jit

#########################################
# PDE Operators @ the element level directly without using a class, leveraging Numba LLVM
@jit(nopython=True)
def jacobian(xae):
    DNaDxi = np.array([[-1., 1., 0.], [-1., 0., 1.]])
    DxDxi = DNaDxi@xae
    return np.linalg.det(DxDxi)

@jit(nopython=True)
def conductivityMatrix(xae,cond):
    DNaDxi = np.array([[-1., 1., 0.], [-1., 0., 1.]])
    DxDxi = DNaDxi@xae
    jacobian = np.linalg.det(DxDxi)
    DxiDx = np.linalg.inv(DxDxi)
    GradN =  DxiDx@DNaDxi
    # order 1 element - constant gradient - single gauss point
    # assuming isotropy
    Wl=1./2.
    Celt = Wl* (np.transpose(GradN)@GradN)
    return cond * jacobian * Celt

@jit(nopython=True)
def massMatrix(xae,rho):
#    the_elt=TRI3(xae)
    j = jacobian(xae)
    # order 1 element - 3 gauss points
    xeta =np.array([[1/6., 1./6.], [2./3., 1./6.],[1./6., 2./3.]])
    Wl =np.array([1/6., 1./6., 1./6.])
    Melt = np.zeros((3,3))
    for i in range(0,3):
        xil=xeta[i]
        Na=np.array([[1. - xil[0] - xil[1], xil[0], xil[1]]])
        Melt = Melt+Wl[i]*(np.transpose(Na)@Na)
    return rho * j * Melt

@jit(nopython=True)
def massMatrixLumped(xae,rho):
#    the_elt=TRI3(xae)
    j = jacobian(xae)
    # order 1 element - 3 gauss points
    xeta =np.array([[1/6., 1./6.], [2./3., 1./6.],[1./6., 2./3.]])
    Wl =np.array([1/6., 1./6., 1./6.])
    Melt = np.zeros((3,3))
    for i in range(0,3):
        xil=xeta[i]
        Na=np.array([[1. - xil[0] - xil[1], xil[0], xil[1]]])
        Melt = Melt+Wl[i]*(np.transpose(Na)@Na)
    Melt2= np.zeros((3,3))
    for i in range(0, 3):
        Melt2[i,i]=Melt[i,0]+Melt[i,1]+Melt[i,2]
    return rho * j * Melt2


@jit(nopython=True)
def sourceTermConstant(xae,f):
# constant source term f over the element
#    the_elt=TRI3(xae)
    j = jacobian(xae)
    # order 1 element - 3 gauss points
    xeta = np.array([[1./6., 1./6.], [2./3., 1. / 6.], [1./6., 2./3.]])
    Wl = np.array([1./6., 1./6., 1./6.])
    load = np.zeros(3)
    for i in range(0, 3):
        xil = xeta[i]
        Na = np.array([1. - xil[0] - xil[1], xil[0], xil[1]])
        load = load + Wl[i] * f * (Na)
    return j * load

@jit(nopython=True)
def sourceTerm(xae,func):
# source term given by func a function  of space over the element
#
    j = jacobian(xae)
    # order 1 element - 3 gauss points
    xeta = np.array([[1./6., 1./6.], [2./3., 1./6.], [1./6., 2./3.]])
    Wl = np.array([1 / 6., 1./6., 1. / 6.])
    load = np.zeros(3)
    for i in range(0, 3):
        xil = xeta[i]
        Na = np.array([[1 - xil[0] - xil[1], xil[0], xil[1]]])
        xx = np.array([(Na@(xae[:, 0]))[0], (Na@(xae[:, 1]))[0]])
        load = load + Wl[i] * func(xx) * (Na[0,:])
    return j * load

####################################################################
#### Pure python / numpy implementation with a class
class TRI3 :
    dim=2    # surface element
    order=1
    nnodes=3

    DNaDxi = np.asarray([[-1.,1.,0.],[-1.,0.,1.]],dtype=float)

    def __init__(self,xae):
        assert type(xae) is np.ndarray, "xae is Not a numpy array "
        self.xae=xae
        self.DxDxi=np.dot(self.DNaDxi,xae)
        self.jacobian=np.linalg.det(self.DxDxi)
        self.DxiDx=np.linalg.inv(self.DxDxi)

    def Naxi(self,xil) :
        assert type(xil) is np.ndarray,"xil is Not a numpy array  "
        return np.asarray([[1-xil[1]-xil[2], xil[1], xil[2]]],dtype=float)

    def Bmatrix(self):
        # cartesian B matrix
        DNaDx = np.dot(self.DxiDx,self.DNaDxi)
        return np.asarray([[DNaDx[1,1],0.,DNaDx[1,2],0.,DNaDx[1,3],0.],
                        [0.,DNaDx[2,1],0.,DNaDx[2,2],0.,DNaDx[2,3]],
                        [DNaDx[2,1],DNaDx[1,1],DNaDx[2,2],DNaDx[2,3],DNaDx[1,3]]],dtype=float)

    def GradN(self):
        # gradient matrix cartesian
        return np.matmul(self.DxiDx, self.DNaDxi)

    def MapX(self,xil):
        assert type(xil) is np.ndarray, "xil Not a numpy array  "
        aux = self.Naxi(xil)
        return np.asarray([np.dot(aux, self.xae[:, 1]), np.dot(aux, self.xae[:, 2])], dtype=float)
