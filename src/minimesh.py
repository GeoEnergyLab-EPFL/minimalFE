#
# This file is part of minimalFE.
#
# Created by Brice Lecampion on 07.02.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

import numpy as np
from numba import jit


class minimesh :
    """" A basic and minimal mesh class """

    def __init__(self,dimension,coor,conn,order):

#     need to write checks
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
        self.matid = np.ones(self.nelts)
        self.nmat =np.unique(self.matid).size

    def set_matid(self,matid):
# matid is a vector of length equal to the number of element
# with the integer id of all the element in the mesh - ordered from elt 1 to nelts
        assert type(matid) is np.ndarray," given matid is not a numpy array"
        assert matid.size == self.nelts, "given matid length do not match the number of elts in the mesh"
        self.matid = matid
        self.nmat=np.unique(matid).size
