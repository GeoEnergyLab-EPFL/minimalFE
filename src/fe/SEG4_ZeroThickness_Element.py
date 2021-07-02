#
# This file is part of minimalFE.
#
# Created by Brice Lecampion on 02.07.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

# ZERO THICKNESS 4 NODES segment - following the formulation of Segura & Carol

## this is a pure python file - needs to be numba @ jit to WORK with assemble.
import numpy as np
from numba import jit

from src.fe.SEG2_Element import *



# we need to code up
# the element conductivity matrix for the 4 nodes segment
# the elememt mass/storage matrix for the 4 nodes segment
# using the element matrix of the SEG2 element

# we probably need to have a special case for the "tip" element - where we only have 3 nodes for pressure
# because 
