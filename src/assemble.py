#
# This file is part of minimalFE.
#
# Created by Brice Lecampion on 07.02.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

from scipy.sparse import *

def assemble(mesh,fun,property):

    row=[]
    col=[]
    data=[]
    for e in range(mesh.nelts) :

        xae=mesh.coor[mesh.conn[e]]
        dof_id = mesh.conn[e]
        celt=fun(xae,property)
#    Celt[np.ix_(dof_id,dof_id)]=Celt[np.ix_(dof_id,dof_id)]+celt
        for i in range(dof_id.size):
            for j in range (dof_id.size):
                row.append(dof_id[i])
                col.append(dof_id[j])
                data.append(celt[i,j])

    ndof = mesh.nnodes  # that;s ok only for scalar problem
# we use coo matrix - the summation is performed during creation ! great
    Csp= coo_matrix((data,(row,col)),shape=(ndof,ndof), dtype=float)
# we convert to csr format for future algrebra.
    return Csp.tocsr()
