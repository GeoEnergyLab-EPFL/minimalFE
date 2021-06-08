#
# This file is part of minimalFE.
#
# Created by Brice Lecampion on 07.02.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

from scipy.sparse import coo_matrix
from numba import njit,prange,jit
import numpy as np


#########Matrix FEM Assembly routines
# works for 2D surface only
@njit(parallel=True)
def _assembleRaw(coor,conn,fun,matid,property,eltfunc):
# this function does not "assemble", the summation is performed when converting from COO to CSR format
# nelts:: number of elements in the mesh (for pre-allocation)
# coor :: array containing the nodes coordinates of the mesh
# conn :: connectivity array of the mesh
# fun :: bi-linear form at the element level
# matid :: array of ID of the material in the mesh
# property :: property array (length equal to the number of material)
    nelts = len(conn)
    ndof_e = conn[0].size   # we assume all elements are of the same type
    row=np.zeros(nelts*ndof_e*ndof_e)
    col=np.zeros(nelts*ndof_e*ndof_e)
    data=np.zeros(nelts*ndof_e*ndof_e)
    for e in prange(nelts) :
        xae=eltfunc(coor[conn[e]]) # pass this as a lambda fct as an additional input to the function ?
        dof_id = conn[e]
        prop_e = property[matid[e]]
        celt=fun(xae,prop_e)
        for i in range(ndof_e):
            for j in range (ndof_e):
                k= e*ndof_e*ndof_e+(i*ndof_e+j)
                row[k]=dof_id[i]
                col[k]=dof_id[j]
                data[k]=celt[i,j]
    return row,col,data

# Matrix assembly
def assemble(mesh,fun,matid,property,eltfunc) :
# mesh :: mesh object
# fun :: the bi-linear form function at the element level
# matid :: array of ID of the material in the mesh
# property :: property array (length equal to the number of material)
    (row,col,data)=_assembleRaw(mesh.coor, mesh.conn, fun, matid, property,eltfunc) # we pass np array for speed
    ndof = mesh.nnodes  # that;s ok only for scalar problem
# we use coo matrix - the summation is performed  when converting to CSR format ! great
    Csp= coo_matrix((data,(row,col)),shape=(ndof,ndof), dtype=float)
# we convert to csc format for future algrebra with scipy
    return Csp.tocsc()  # for splu needs csc

######### vector assembly routines
# Vector Assembly routines for the case of a source term given by a spatial function
@njit()
def _assembleVecRawFun(coor, conn, fun, args,eltfunc):
    nelts = len(conn)
    ndof_e = conn[0].size  # we assume all elements are of the same type
    row = np.zeros(nelts * ndof_e)
    col = np.zeros(nelts * ndof_e)
    data = np.zeros(nelts * ndof_e)
    for e in range(nelts):
        xae =eltfunc(coor[conn[e]])
        dof_id = conn[e]
        lelt = fun(xae,args)
        for i in range(ndof_e):
                k = e*ndof_e + i
                row[k] = dof_id[i]
                data[k] = lelt[i]
                col[k] = 0
    return row, col, data

# assemble load vector given by a function argument
def assembleLoadFun(mesh, fun, arg, eltfunc) :
# mesh object
# fun : function for the operator
# arg : argument of the operator function
    (row, col, data) = _assembleVecRawFun(mesh.coor, mesh.conn, fun, arg,eltfunc)  # we pass np array for speed
    ndof = mesh.nnodes  # that;s ok only for scalar problem
    # we use coo matrix - the summation is performed  when converting to CSR format ! great
    Csp = coo_matrix((data, (row, col)), shape=(ndof, 1), dtype=float)
    # we convert to csc format for future algrebra with scipy
    return Csp.tocsc() # for fast splu

# Vector Assembly routines for the case of a source term constant per element
@njit()
def _assembleVecRaw(coor, conn,fun,matid,source,eltfunc):
    # source is an numpy array
    nelts = len(conn)
    ndof_e = conn[0].size  # we assume all elements are of the same type
    row = np.zeros(nelts * ndof_e)
    col = np.zeros(nelts * ndof_e)
    data = np.zeros(nelts * ndof_e)
    for e in range(nelts):
        xae =eltfunc(coor[conn[e]])
        dof_id = conn[e]
        source_e = source[matid[e]]
        lelt = fun(xae,source_e)
        #    Celt[np.ix_(dof_id,dof_id)]=Celt[np.ix_(dof_id,dof_id)]+celt
        for i in range(ndof_e):
                k = e*ndof_e + i
                row[k] = dof_id[i]
                data[k] = lelt[i]
                col[k] = 0
    return row, col, data

# assemble load source term  - constant source per element
def assembleSource(mesh, operator, matid,source, eltfunc) :
# mesh object
# fun : function for the operator
# arg : argument of the operator function
    (row, col, data) = _assembleVecRaw(mesh.coor, mesh.conn, operator,matid, source, eltfunc)  # we pass np array for speed
    ndof = mesh.nnodes  # that;s ok only for scalar problem
    # we use coo matrix - the summation is performed  when converting to CSC format ! great
    Csp = coo_matrix((data, (row, col)), shape=(ndof, 1), dtype=float)
    # we convert to csc format for future algrebra with scipy
    return Csp.tocsc()   #   here to csc for fast LU

