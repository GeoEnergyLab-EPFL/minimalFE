#
# This file is part of PyFracX.
#
# Created by Brice Lecampion on 09.02.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#


# shall we do a class hierarchy ?  to be discussed
# we want to be able to accomodate non -planar 3D surfaces in the future
# unstructured as well as cartesian mesh

class SurfaceMesh :
    pass
# x,y z

class PlanarSurface(SurfaceMesh):
    pass
# is this a good idea, or shall we do a single general SurfaceMesh class ?
# passing the third coordinates as scalar variable ?

class CartesianSurfaceMesh(PlanarSurface) :
    pass
#?is this a good idea

