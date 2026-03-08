#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provide methods to setup LM matrices, create FE model for a truss from a json 
file, to plot the truss, to calculate and print stresses of every element.

Created on Sat May 9 15:43:00 2020

@author: thurcni@163.com, xzhang@tsinghua.edu.cn
"""

import FEData as model
import numpy as np
import json
import matplotlib.pyplot as plt


def create_model_json(DataFile):
    """ 
    Initialize the FEM model from file DataFile (in json format)
    """

    # input data from json file
    with open(DataFile) as f_obj:
        FEData = json.load(f_obj)
    
    model.Title= FEData['Title']
    model.nsd  = FEData['nsd']
    model.ndof = FEData['ndof']
    model.nnp  = FEData['nnp']
    model.nel  = FEData['nel']
    model.nen  = FEData['nen']    
    model.neq  = model.ndof*model.nnp
    model.nd   = FEData['nd']

    # initialize K, d and f 
    model.f = np.zeros((model.neq,1))            
    model.d = np.zeros((model.neq,1))        
    model.K = np.zeros((model.neq,model.neq))

    # define the mesh
    model.x = np.array(FEData.get('x', []))
    model.y = np.array(FEData.get('y', np.zeros_like(model.x)))
    model.z = np.array(FEData.get('z', np.zeros_like(model.x)))

    model.IEN = np.array(FEData['IEN'], dtype=int)
    model.LM = np.zeros((model.nen*model.ndof, model.nel), dtype=int)
    set_LM()

    # element and material data (given at the element)
    model.E     = np.array(FEData['E'])
    model.CArea = np.array(FEData['CArea'])
    
    idx0 = model.IEN[:, 0] - 1
    idx1 = model.IEN[:, 1] - 1
    model.leng = np.sqrt(
        (model.x[idx1] - model.x[idx0])**2 +
        (model.y[idx1] - model.y[idx0])**2 +
        (model.z[idx1] - model.z[idx0])**2
    )
    
    model.stress= np.zeros((model.nel,))

    # prescribed forces
    fdof = FEData['fdof']
    force= FEData['force']
    for ind, value in enumerate(fdof):
        model.f[value-1][0] = force[ind]

    # output plots
    model.plot_truss= FEData.get('plot_truss', 'no')
    model.plot_node = FEData.get('plot_node', 'no')
    model.plot_tex  = FEData.get('plot_tex', 'no')
    
    # Optional print here instead of plottruss, because we print stats in plot_deformation now
    # We will just print the summary here.
    print(f"\t{model.ndof}D Truss Params \n")
    print(model.Title + "\n")
    print("No. of Elements  {0}".format(model.nel))
    print("No. of Nodes     {0}".format(model.nnp))
    print("No. of Equations {0}".format(model.neq))


def plot_deformation(scaling_factor=None):
    '''
    Plot the deformed truss structure.
    '''
    if model.plot_truss != "yes":
        return

    # Calculate displacements
    d = model.d.flatten()
    ux = d[0::model.ndof]
    uy = d[1::model.ndof] if model.ndof >= 2 else np.zeros_like(ux)
    uz = d[2::model.ndof] if model.ndof == 3 else np.zeros_like(ux)

    # Automatically determine scaling factor if not provided
    if scaling_factor is None:
        max_dim = max(np.max(model.x) - np.min(model.x), np.max(model.y) - np.min(model.y))
        max_disp = np.max(np.sqrt(ux**2 + uy**2 + uz**2))
        scaling_factor = 0.1 * max_dim / max_disp if max_disp > 0 else 1.0
    
    print(f"Plotting deformation with scaling factor: {scaling_factor:.2e}")

    # Deformed coordinates
    x_def = model.x + scaling_factor * ux
    y_def = model.y + scaling_factor * uy
    z_def = model.z + scaling_factor * uz

    fig = plt.figure()
    is_3d = (model.ndof == 3)
    ax = fig.add_subplot(111, projection='3d') if is_3d else fig.add_subplot(111)

    for i in range(model.nel):
        n1, n2 = model.IEN[i, 0] - 1, model.IEN[i, 1] - 1
        
        pts_orig = [model.x[[n1, n2]], model.y[[n1, n2]]]
        pts_def = [x_def[[n1, n2]], y_def[[n1, n2]]]
        
        if is_3d:
            pts_orig.append(model.z[[n1, n2]])
            pts_def.append(z_def[[n1, n2]])
            
        # Plot original with thicker linewidth when overlapping
        ax.plot(*pts_orig, "b--", alpha=0.3, linewidth=3)
        # Plot deformed with standard linewidth (1.5)
        ax.plot(*pts_def, "r-", linewidth=1.5)
        
        if model.plot_node == "yes":
            ax.text(*[p[0] for p in pts_orig], str(n1 + 1)) # type: ignore
            ax.text(*[p[1] for p in pts_orig], str(n2 + 1)) # type: ignore

    if is_3d:
        ax.set_zlabel("z") # type: ignore
    
    plt.title(f"Deformed Truss (Scale: {scaling_factor:.1f}x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.legend(["Original", "Deformed"])
    plt.savefig("truss_deformation.pdf")
    
    if model.plot_tex == "yes":
        try:
            import tikzplotlib
            tikzplotlib.clean_figure()
            tikzplotlib.save("fe_plot.tex")
        except ImportError as e:
            print(f"Warning: Failed to import tikzplotlib or related module to generate TeX plot ({e}). "
                  "Ignoring TeX plot generation.")
                  
    plt.show()


def set_LM():
    '''
    set up Location Matrix
    '''
    for e in range(model.nel):
        for j in range(model.nen):
            for m in range(model.ndof):
                ind = j*model.ndof + m
                model.LM[ind, e] = model.ndof*(model.IEN[e, j] - 1) + m


def print_stress():
    '''
    Calculate and print stresses of every element
    '''

    # prints the element number and corresponding stresses
    print("Element\t\t\tStress")
    # Compute stress for each element
    for e in range(model.nel):
        de = model.d[model.LM[:, e]].flatten()  # nodal displacements for each element
        const = model.E[e]/model.leng[e]

        n1, n2 = model.IEN[e, 0] - 1, model.IEN[e, 1] - 1
        
        d_coord = [model.x[n2] - model.x[n1]]
        if model.ndof >= 2: d_coord.append(model.y[n2] - model.y[n1])
        if model.ndof == 3: d_coord.append(model.z[n2] - model.z[n1])
        
        c = np.array(d_coord) / model.leng[e]
        T = np.concatenate([-c, c])
        
        model.stress[e] = const * (T @ de)
        print("{0}\t\t\t{1}".format(e+1, model.stress[e]))
        