#
# This file is part of minimalFE.
#
# Created by Brice Lecampion on 07.02.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

import numpy as np

# Unstructured mesh class - minimal
# dependencies: meshio package (pip install meshio)

class usmesh :
    """" A basic and minimal mesh class """

    def __init__(self,dimension,coor,conn,order):
        assert type(dimension) is int, "dimension is not an integer "
        assert type(coor) is np.ndarray, "given coordinates is not a numpy array"
        assert type(conn) is np.ndarray, "given connectivity is not a numpy array "
        assert conn.dtype == "int", "given connectivity is not made of integer"

        self.dimension = dimension
        self.coor = coor
        self.conn = conn
        self.order = order
        self.nelts = conn.shape[0]
        self.nnodes = coor.shape[0]

    @classmethod
    def fromMeshio(cls,meshio,order):
        # for now we only accept mesh made  of a single type of element
        lk=list(meshio.cells_dict.keys())
        assert len(lk)==1
        return cls(meshio.points.shape[1],meshio.points,meshio.cells_dict[lk[0]],order)

