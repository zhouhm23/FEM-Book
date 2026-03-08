#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides utilities used by FE analysis.
  1. assembly: Global stiffness matrix assembly.
  2. solvedr: Solving the stiffness equations by the reduction approach.

Created on Sat May 9 17:39:00 2020

@author: thurcni@163.com, xzhang@tsinghua.edu.cn
"""

import numpy as np
import FEData as model


def assembly(e, ke):
    """
    Assemble element stiffness matrix.
    
    Args:
        e   : (int) Element number
        ke  : (numpy(nen*ndof,nen*ndof)) element stiffness matrix
    """
    model.K[np.ix_(model.LM[:,e], model.LM[:,e])] += ke

def solvedr(method="penalty", **kwargs):
    """
    Solve K d = f by penalty method or reduction method for displacement BC.
    Assumption:
        - constrained DOFs are [0, nd)
        - prescribed displacements are model.d[0:nd]

    Args:
        method: 'penalty' (default) or 'reduction'.
        penalty method: alpha (float, optional)
    Returns:
        f_E : reaction on constrained DOFs
    """
    nd = model.nd
    neq = model.neq

    # original system backup for reaction recovery
    K0 = np.asarray(model.K, dtype=float).copy()
    f0 = np.asarray(model.f, dtype=float).reshape(-1).copy()
    d0 = np.asarray(model.d, dtype=float).reshape(-1).copy()

    if f0.size != neq:
        raise ValueError(f"model.f size({f0.size}) != model.neq({neq})")
    if d0.size != neq:
        raise ValueError(f"model.d size({d0.size}) != model.neq({neq})")

    if method == "penalty":
        # working system
        K = K0.copy()
        F = f0.copy()

        alpha = kwargs.get("alpha", None)
        # auto penalty
        if alpha is None:
            diag_abs_max = np.max(np.abs(np.diag(K))) if K.size > 0 else 1.0
            alpha = 1e8 * (diag_abs_max if diag_abs_max > 0 else 1.0)

        # apply displacement BC by penalty
        for i in range(nd):
            u_bar = d0[i]
            K[i, i] += alpha
            F[i] += alpha * u_bar

        # solve full system
        d = np.linalg.solve(K, F)
        model.d = d

        # reaction from original system
        r = K0 @ d - f0
        f_E = r[0:nd]

    elif method == "reduction":
        # Partition system: [Kcc Kcf; Kfc Kff] [dc; df] = [fc; ff]
        # where c: constrained (0:nd), f: free (nd:neq)
        Kfc = K0[nd:, :nd]
        Kff = K0[nd:, nd:]
        ff = f0[nd:]
        dc = d0[:nd]  # prescribed

        # Solve for free dofs: Kff df = ff - Kfc*dc
        rhs = ff - Kfc @ dc
        if Kff.shape[0] == 0:
            df = np.array([])
        else:
            df = np.linalg.solve(Kff, rhs)

        # Assemble full d
        d = np.zeros_like(d0)
        d[:nd] = dc
        d[nd:] = df
        model.d = d

        # reaction on constrained dofs
        r = K0 @ d - f0
        f_E = r[:nd]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'penalty' or 'reduction'.")

    print('\nsolution d')
    print(model.d)
    print('\nreaction f =', f_E)

    return f_E