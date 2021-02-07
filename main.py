# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

from minimesh import minimesh
from SEG2_Element import *
from assemble import assemble
from scipy.sparse.linalg import splu

nelts=100
coor= np.geomspace(1.,101.,num=nelts+1)-1. #geomspace
conn=np.zeros((nelts,2),dtype=int)
for e in range(nelts):
    conn[e,0]=e
    conn[e,1]=e+1

#coor=np.asarray([0.,1.,2.,3.],dtype=float)
#conn=np.asarray([ [0,1],[1,2],[2,3]],dtype=int)

mmesh = minimesh(1,coor,conn,1)

xae1=coor[conn[0]]
elt0=SEG2(xae1)
b0=(1./elt0.jacobian) * elt0.Bmatrix()



properties={"Conductivity": 1.,"Storage":1. }
#Celt=np.zeros((ndof,ndof))  # will switch to sparse later

# assemble conductivity and mass matrix.
start = time.time()
myC=assemble(mmesh,conductivityMatrix,properties["Conductivity"])
myM=assemble(mmesh,massMatrix,properties["Storage"])
end = time.time()
print(end - start)
# solve for the problem of a constant pressure at the origin

dt = 0.001
myK=myM + dt * myC
luK=splu(myK.tocsc())  # lu decomposition

# analytical solution
from scipy.special import erfc
def pressolution(x,t):
    return erfc(x/(2.*(t**0.5)))
def rate(t):
    return 1./((np.pi*t)**0.5)

p0=np.zeros(mmesh.nnodes,dtype=float)
pinlet=1. # constant pressure at 0.
p0[0]=pinlet

pn=p0.copy()

tt=0.
k=0
q0=np.asarray([0.])
times=np.asarray([0.])

while k < 500:
    k=k+1
    tt=tt+dt
    times=np.append(times, [tt])
    ft = -dt* myC.dot(pn)
    ft[0]=ft[0]+dt*rate(tt) # injection rate as function
    dp=luK.solve(ft)
    pn=pn+dp


start = time.time()
dp=luK.solve(ft)
end = time.time()
print(end - start)

# plots and checks
sol=pressolution(coor,tt)

fig, ax = plt.subplots()
ax.plot(coor, sol)
ax.plot(coor, pn)

plt.show()

fig, ax = plt.subplots()
ax.plot(coor, abs(sol-pn))
plt.show()

fig, ax = plt.subplots()
ax.plot(np.arange(nelts+1), coor)
plt.show()

