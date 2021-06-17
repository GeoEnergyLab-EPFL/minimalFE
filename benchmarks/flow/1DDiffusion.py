#
# This file is part of PyFracX.
#
# Created by Brice Lecampion on 03.06.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#


# 1D diffusion with imposed constant pressure.


import numpy as np
import numba
import time,sys


import matplotlib
import matplotlib.pyplot as plt

from mesh.usmesh import usmesh
from mesh.mesh_utils import *
from fe.SEG2_Element import *
from fe.assemble import *
from scipy.sparse.linalg import splu
from scipy.sparse import csr_matrix


# simple 1D mesh
Nelts=1000
coor1D=np.linspace(-100.,100.,Nelts+1)
coor = np.transpose(np.array([coor1D,coor1D*0.]))
conn=np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)

me=usmesh(2,coor,conn,1)

# analytical solution for pressure at collocation points
from scipy import special
pressure = lambda x,t,Dpcenter: Dpcenter * special.erfc(np.abs(x)/((4.*t)**0.5))

# -

# xae=project_segment_elt(me.coor[me.conn[0]])
#
# jacobian(xae)
#
# ma=massMatrix(xae,1)
#
# ca=conductivityMatrix(xae,1.)


properties={"Conductivity": 1.,"Storage":1. }

# single material
matid=np.zeros(me.nelts,dtype=int)

#(row,col,data)=fe.assemble._assembleRaw(me.coor,me.conn,conductivityMatrix,matid,np.array([properties["Conductivity"]]),project_segment_elt)
# assemble conductivity matrix.
t = time.process_time()
myC=assemble(me,conductivityMatrix,matid,np.array([properties["Conductivity"]]),project_segment_elt)
elapsed_time = time.process_time() - t
print(elapsed_time)

# assemble  mass matrix.
t = time.process_time()
myM=assemble(me,massMatrix,matid,np.array([properties["Storage"]]),project_segment_elt)
elapsed_time = time.process_time() - t
print(elapsed_time)



# find nodes at mesh center
ij = np.where(me.coor[:,0]==0.)[0][0] # anyway there is a


# we impose the constant pressure by adding a large number to the matrix entry
Mlarge=1.0e10
# inj pressure
Pinj=1.0
### CONTANT TIME STEEPING SOLUTION
dt = 0.02
myK = myM + dt * myC
myK[ij,ij]+=Mlarge
luK = splu(myK)  # splu decomposition needs a csc format

p0 = np.zeros(me.nnodes, dtype=float)
p0[ij]=Pinj
pn = p0.copy()
tt = 0.
k = 0
q0 = np.asarray([0.])
times = np.asarray([0.])

while k < 2000:
    k = k + 1
    tt = tt + dt
    times = np.append(times, [tt])
    ft = dt * np.asarray(-(myC @ pn) )
    dp = luK.solve(ft)
    pn = pn + dp


# compute the radial coordinates of the mesh nodes.
x=me.coor[:,0]

# plots and checks

#x=np.linspace(0.0, 10.0, num=300)
sol=pressure(x,tt,Pinj)

fig, ax = plt.subplots()
ax.plot(x, sol)
ax.plot(x, pn,'.')
plt.show()

abs_err = np.abs(pn-sol)

fig, ax = plt.subplots()
ax.semilogy(x, abs_err,'.')
plt.show()
