#
# This file is part of minimalFE.
#
# Created by Brice Lecampion on 07.02.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

import numpy as np

class SEG2 :
    dim=1
    order=1
    nnodes=2
    DNaDxi =np.asarray([[-0.5 , 0.5]],dtype=float)

    def Naxi(self,xil) :
        assert type(xil) is float,"xil Not a float "
        return np.asarray([[0.5 * (1 - xil) , 0.5 * (1 + xil)]],dtype=float)

    def __init__(self,xae):
        assert type(xae) is np.ndarray, "xae is Not a numpy array "
        self.xae=xae
        self.DxDxi=np.dot(self.DNaDxi,xae)
        self.jacobian=self.DxDxi

    def Bmatrix(self):
        # cartesian B matrix
        return (1. / self.jacobian) *  self.DNaDxi

    def Na(self,xil):
        # cartesian shape
        assert type(xil) is float, "xil is Not a float "
        return np.asarray([[0.5 * (1 - xil) , 0.5 * (1 + xil)]],dtype=float)

    def MapX(self,xil):
        Na_xi=self.Na(xil)
        return Na_xi*self.xae

# PDE Operators @ the element level

def conductivityMatrix(xae,cond) :

    the_elt=SEG2(xae)
    xeta = 0.  # order 1 element - single gauss point
    Celt =  2.*np.dot(np.transpose(the_elt.Bmatrix()),the_elt.Bmatrix())
    return  cond * (the_elt.jacobian) * Celt

def massMatrix(xae,rho):

    the_elt=SEG2(xae)
    xeta = [-(3./5)**0.5,0., (3./5)**0.5]  # order 1 element - 3 gauss points
    Wl  = [5./9, 8/9., 5./9.]
    Melt = np.zeros((2,2),dtype=float)
    for i in range(0,3):
        Melt = Melt+Wl[i]*np.dot(np.transpose(the_elt.Na(xeta[i])),the_elt.Na(xeta[i]))

    return rho * (the_elt.jacobian) * Melt

#...
