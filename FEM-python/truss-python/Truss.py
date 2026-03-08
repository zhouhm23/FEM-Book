#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Truss - truss FE analysis program.
  Element types  : 1-, 2- or 3-dimension linear bar element.
  Problem solved : truss whose Young's modulus, cross-sectional area are known within each element
    is deformed at nodal forces.

Usage:
   >>> Truss file_name
   
Command line arguments:
  file_name: File name in which the FE model is stored in json format

Created on Sat May 9 18:34:00 2020

@author: thurcni@163.com, xzhang@tsinghua.edu.cn
"""
import argparse
import FEData as model
from TrussElem import TrussElem
from PrePost import create_model_json, print_stress, plot_deformation
from utitls import assembly, solvedr

def FERun(DataFile, method="reduction"):
    # create FE model from DataFile in json format
    create_model_json(DataFile)

    # Element matrix computations and assembly
    for e in range(model.nel):
        ke = TrussElem(e)
        assembly(e, ke)
    
    # Partition and solution
    solvedr(method=method)

    # Postprocessing
    print_stress()
    plot_deformation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truss FE analysis program.")
    parser.add_argument("file_name", help="File name in which the FE model is stored in json format")
    parser.add_argument("--method", choices=["reduction", "penalty"], default="reduction",
                        help="Select the method for applying displacement boundary conditions (default: reduction).")
    args = parser.parse_args()

    FERun(args.file_name, method=args.method)