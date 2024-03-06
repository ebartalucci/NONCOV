###################################################
# MAIN SOURCE CODE FOR THE NONCOV DFT-NMR PROJECT #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 26.02.2024                 #
#               Last:  26.02.2024                 #
#               -----------------                 #
#             Stable release version              #
#                   v.1.1.0                       #
#                                                 #
###################################################

# This work is an attempt to write a script running on minimal import packages

# Import modules 
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re # RegEx

class NONCOVToolbox:

        def __init__(self):

        # Print header and version
        print("\n\n          #################################################")
        print("          | --------------------------------------------- |")
        print("          |         (NC)^2I.py: NMR Calculations          |")
        print("          |         for Noncovalent Interactions          |")
        print("          | --------------------------------------------- |")
        print("          |           Introducing: NONCOVToolbox          |")
        print("          |                       -                       |")
        print("          |     A collection of functions for working     |")
        print("          |         with calculated NMR parameters        |")
        print("          |                                               |")
        print("          |               Ettore Bartalucci               |")
        print("          |     Max Planck Institute CEC & RWTH Aachen    |")
        print("          |            Worringerweg 2, Germany            |")
        print("          |                                               |")
        print("          #################################################\n")
        
        # Print versions
        version = '0.0.1'
        print("Stable version: {}\n\n".format(version))
        print("Working python version:")
        print(sys.version)
        print('\n')



        # ---------------------------------------------------------------------------- #
        # BIOMOLECULAR APPLICATIONS: AMINO ACID STATISTICS