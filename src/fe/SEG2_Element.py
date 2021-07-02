#
# This file is part of minimalFE.
#
# Created by Brice Lecampion on 07.02.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

## this is a pure python file - needs to be numba @ jit to WORK with assemble.
import numpy as np
from numba import jit

#########################################
# PDE Operators @ the element level directly without using a class, leveraging Numba LLVM
@jit(nopython=True)#,signature_or_function='float64(float64[:])'
def jacobian(xae):
    DNaDxi = np.array([-0.5 , 0.5])
    DxDxi =DNaDxi@xae
    return DxDxi


@jit(nopython=True,signature_or_function='float64[:,:](float64[:],float64)')
def conductivityMatrix(xae,cond):
    DNaDxi = np.array([[-0.5 , 0.5]])
    j = DNaDxi@xae
    GradN =   DNaDxi /j
    # order 1 element - constant gradient - single gauss point
    # assuming isotropy
    Wl=2.
    Celt = Wl* (np.transpose(GradN)@GradN) * j
    return cond  * Celt


@jit(nopython=True,signature_or_function='float64[:,:](float64[:],float64)')
def massMatrix(xae,rho):
    j = jacobian(xae)
    # order 1 element
    Melt= np.array([[2./3.,1./3],[1./3,2./3]])
    # xeta =np.array([-(3./5)**0.5,0., (3./5)**0.5])
    # Wl = [5. / 9, 8 / 9., 5. / 9.]
    # Melt = np.zeros((2, 2), dtype=float)
    # for i in range(0, 3):
    #     xil = xeta[i]
    #     Na=np.array([[0.5 * (1 - xil), 0.5 * (1 + xil)]], dtype=float)
    #     Melt = Melt + Wl[i] * np.dot(np.transpose(Na), Na)
    return rho * j * Melt

@jit(nopython=True)
def massMatrixLumped(xae,rho):
    j = jacobian(xae)
    # order 1 element - 3 gauss points
    xeta =np.array([-(3./5)**0.5,0., (3./5)**0.5])
    Wl = [5. / 9, 8 / 9., 5. / 9.]
    Melt = np.zeros((2, 2), dtype=float)
    for i in range(0, 3):
        xil = xeta[i]
        Na=np.array([[0.5 * (1 - xil), 0.5 * (1 + xil)]], dtype=float)
        Melt = Melt + Wl[i] * np.dot(np.transpose(Na), Na)
    Melt2 = np.zeros((2, 2))
    for i in range(0, 2):
         Melt2[i, i] = Melt[i, 0] + Melt[i, 1]
    return rho * j * Melt2


@jit(nopython=True)
def sourceTermConstant(xae,f):
# constant source term f over the element
    j = jacobian(xae)
    # order 1 element
    load = np.zeros(2)
    load = f *  np.array([1 , 1])
    return j * load





############################# old
class SEG2 :
    dim=1
    order=1
    nnodes=2
    DNaDxi =np.asarray([[-0.5 , 0.5]],dtype=float)

    def __init__(self,xae):
        assert type(xae) is np.ndarray, "xae is Not a numpy array "
        self.xae=xae
        self.DxDxi=np.dot(self.DNaDxi,xae)
        self.jacobian=self.DxDxi

    def Bmatrix(self):
        # cartesian B matrix
        return (1. / self.jacobian) * self.DNaDxi

    def Naxi(self,xil):
        # cartesian shape
        assert type(xil) is float, "xil is Not a float "
        return np.asarray([[0.5 * (1 - xil) , 0.5 * (1 + xil)]],dtype=float)

    def MapX(self,xil):
        Na_xi=self.Naxi(xil)
        return np.dot(Na_xi,self.xae)

#########################################
# PDE Operators @ the element level
#
# def conductivityMatrix(xae,cond) :
#
#     the_elt = SEG2(xae)
#     # order 1 element - constant gradient - single gauss point
#     Celt = 2.*np.dot(np.transpose(the_elt.Bmatrix()),the_elt.Bmatrix())
#     return cond * the_elt.jacobian * Celt
#
# def massMatrix(xae,rho):
#
#     the_elt = SEG2(xae)
#     xeta = [-(3./5)**0.5,0., (3./5)**0.5]  # order 1 element - 3 gauss points
#     Wl = [5./9, 8/9., 5./9.]
#     Melt = np.zeros((2,2),dtype=float)
#     for i in range(0,3):
#         Melt = Melt+Wl[i]*np.dot(np.transpose(the_elt.Naxi(xeta[i])),the_elt.Naxi(xeta[i]))
#     return rho * the_elt.jacobian * Melt
