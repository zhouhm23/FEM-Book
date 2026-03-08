#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides methods to calculate element stiffness matrix

Created on Sat May 9 15:40:00 2020

@author: thurcni@163.com, xzhang@tsinghua.edu.cn
"""

import FEData as model
import numpy as np

def TrussElem(e):
    '''
    calculate element stiffness matrix

    Args:
        e : (int) element number

    Returns:
        ke : (numpy(nen,nen)) element stiffness matrix
    '''
    # constant coefficient for each truss element
    const = model.CArea[e]*model.E[e]/model.leng[e]

    # calculate element stiffness matrix generalized for 1D, 2D, and 3D
    n1, n2 = model.IEN[e, 0] - 1, model.IEN[e, 1] - 1
    
    d_coord = [model.x[n2] - model.x[n1]]
    if model.ndof >= 2: d_coord.append(model.y[n2] - model.y[n1])
    if model.ndof == 3: d_coord.append(model.z[n2] - model.z[n1])
    
    T = np.array(d_coord) / model.leng[e]
    ke_top = const * np.outer(T, T)
    
    ke = np.block([[ke_top, -ke_top],
                   [-ke_top, ke_top]])
    
    return ke
