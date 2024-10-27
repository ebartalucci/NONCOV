###################################################
#       SOURCE CODE FOR THE NONCOV PROJECT        #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 26.02.2024                 #
#               Last:  17.10.2024                 #
#               -----------------                 #
#             Stable release version              #
#                   v.0.0.1                       #
#                                                 #
###################################################

# Attempt to run on minimal import packages

# Import modules 
import os
import sys
import random
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import re  # RegEx
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


class NONCOVToolbox:
    """
    The NONCOV toolbox
    """
    def __init__(self):
        pass

class NONCOVHeader:
    """
    Print headers and acknowledgments
    """
    _printed = False

    @staticmethod
    def print_header():
        if not NONCOVHeader._printed:

            # Print header and version
            print("\n\n          #################################################")
            print("          | --------------------------------------------- |")
            print("          |                NMR Calculations               |")
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
            
            # Disclaimer
            print('Please keep in mind that the ORCAAnalysis module \n')
            print('only works with ORCA 6 due to parsing settings. \n')

            # Print acknowledgments
            try:
                # acknowledgments.txt file is in the same directory as noncov.py
                file_dir = os.path.dirname(os.path.abspath(__file__))
                ack_file_path = os.path.join(file_dir, 'acknowledgments.txt')

                with open(ack_file_path, 'r') as f:
                    acknowledgments = f.read()
                    print(acknowledgments)
            
            except FileNotFoundError:
                print(f"Acknowledgments file not found at {ack_file_path}. Please ensure 'acknowledgments.txt' is in the correct directory.")
                sys.exit(1)
            except Exception as e:
                print(f"An error occurred: {e}")
                sys.exit(1)

            NONCOVHeader._printed = True

# ------------------------------------------------------------------------------
#                        NMR FUNCTIONS AND APPLICATIONS 
# ------------------------------------------------------------------------------
class NMRFunctions(NONCOVToolbox):
    """
    Collection of useful functions for working with NMR parameters.
    """
    def __init__(self):
        super().__init__()

    # 3x3 Matrix diagonalization and PAS shielding tensor ordering in Mehring and Haberlen conventions
    def test_diagonalize_tensor(self, shielding_tensor):
        """
        Take NMR shielding tensor as input and perform various operations, 
        including diagonalization and ordering according to various formalisms.
        
        Input
        :param shielding_tensor: tensor components in 3x3 chemical shielding matrix
        
        Output 
        :param shielding_tensor: original shielding tensor in molecular frame
        :param s_iso: isotropic chemical shift from symmetrized tensor
        :param diagonal_mehring: PAS components with magnitude ordering (IUPAC)
        :param eigenvals: unsorted PAS components
        :param eigenvecs: rotation matrix
        :param symmetry: second-rank tensor symmetry
        :param span: maximum width of the powder pattern
        :param skew: amount and orientation of the asymmetry of the tensor
        """

        # Notify user which module has been called
        print("# -------------------------------------------------- #")
        print("# TENSOR DIAGONALIZATION FUNCTION HAS BEEN REQUESTED #")
        print(f'\n')

        shielding_tensor = np.array(shielding_tensor)                
        print(f'Shielding Tensor is: \n{shielding_tensor}')
        print('Proceeding to transposing...\n')

        # Transpose matrix
        transposed = shielding_tensor.T
        print(f'Transposed matrix is: \n{transposed}')
        print('Proceeding to symmetrization...\n')

        # Symmetrize tensor
        sym_shielding_tensor = NMRFunctions.symmetrize_tensor(shielding_tensor)
        antisym_shielding_tensor = NMRFunctions.antisymmetrize_tensor(shielding_tensor)
        print(f'Symmetric tensor is: \n{sym_shielding_tensor}\n')
        print(f'Antisymmetric tensor is. \n{antisym_shielding_tensor}\n')
        print('Since antisymmetric part does not contribute to observable but only to relaxation, skipping...\n')
        print('Proceeding to diagonalization...\n')

        # Calculate eigenvalues and vectors 
        eigenvals, eigenvecs = np.linalg.eig(sym_shielding_tensor)
        eigenvals = eigenvals.round(2) # round them up
        eigenvecs = eigenvecs.round(2) # round them up
        print(f'Eigenvalues are: {eigenvals}, Eigenvectors are: \n{eigenvecs}\n')
        print('Proceeding to ordering eigenvalues and eigenvectors...\n')

        # Sort eigenvalues and eigenvectors based on magnitude of eigenvalues
        idx = np.argsort(np.abs(eigenvals))
        eigenvals_ordered = eigenvals[idx]
        eigenvecs_ordered = eigenvecs[:, idx]
        print(f'Magnitude-based ordering of eigenvalues is: \n{eigenvals_ordered} \n and of eigenvectors is: \n{eigenvecs_ordered}.')
        print('Proceeding to diagonalization...\n')

        # Compute diagonal matrix, define eigenvector columns as variables and preforme matrix multiplication 
        diagonal = np.diag(eigenvals)
        print(f'Diagonalized tensor is: \n{diagonal}')
        print('Proceeding to compute isotropic shift...\n')

        # Compute isotropic shift
        s_iso = np.sum(np.diag(diagonal)) / 3
        s_iso = s_iso.round(2)
        print(f'Isotropic shift is: {s_iso} ppm')
        print('Proceeding to Mehring ordering...\n')

        # Reorder matrix according to Mehring convention
        diagonal_mehring = sorted(np.diag(diagonal))
        sigma_11 = diagonal_mehring[0]
        sigma_22 = diagonal_mehring[1]
        sigma_33 = diagonal_mehring[2]
        diagonal_mehring = np.diag(diagonal_mehring)
        print(f'Diagonal tensor in Mehring order is: \n{diagonal_mehring}\n')
        print(f'where:\n \u03C3_11:{sigma_11} \n \u03C3_22:{sigma_22} \n \u03C3_33:{sigma_33} \n')
        print('Proceeding to shielding tensor symmetry analysis...\n')

        # Extract symmetry of the tensor from its eigenvalues
        unique_eigenvals = np.unique(np.round(np.real(eigenvals),7))
        symmetry = len(np.real(eigenvals)) - len(unique_eigenvals) 
        print(f'Symmetry of the tensor based on eigenvals count is: {symmetry}\n')
        if symmetry == 0:
            print('which means that the tensor is completely anysotropic \n')
        elif symmetry == 1:
            print('which means that the tensor has axial symmetry \n')
        elif symmetry == 2:
            print('which means that the tensor is completely isotropic \n')

        # Additional symmetry checks for shielding tensor
        is_symmetric = np.allclose(shielding_tensor, shielding_tensor.T) 
        if is_symmetric:
            print("The tensor is symmetric (S = S^T).\n")
        else:
            print("The tensor is not symmetric (S != S^T).\n")

        print('Checking for rotational symmetry:\n')
        for i, vec in enumerate(eigenvecs.T):
            print(f'Eigenvector row {i + 1}: {vec}')
        # 180 degrees rotation
        rotated_shielding_tensor = np.rot90(shielding_tensor, 2)
        print(f'180 degrees rotation results in the tensor: \n{rotated_shielding_tensor}\n')
        is_rotationally_symmetric = np.array_equal(shielding_tensor, rotated_shielding_tensor)
        print(f'Rotational symmetry is: \n{is_rotationally_symmetric}\n')
        print('Proceeding to compute span and skew of the tensor...\n')
        
        span = sigma_33 - sigma_11
        span = span.round(2)
        skew = 3*(sigma_22-s_iso)/span
        skew = skew.round(2)
        print(f'Span is: {span}\n')
        print(f'Skew is: {skew}\n')
        
        print('Summary of NMR parameters:\n')
        print(f'Shielding tensor: \n{shielding_tensor} \n')
        print(f'Isotropic shielding: {s_iso} \n')
        print(f'Mehring: \n{diagonal_mehring} \n')
        print(f'Unsorted Eigenvals: \n{eigenvals} \n')
        print(f'Span is: {span}\n')
        print(f'Skew is: {skew}\n')

        print('Proceeding...\n')
        print('Call tensor_to_euler for Euler angles extraction from eigenvectors...\n')

        print("# -------------------------------------------------- #")

        return shielding_tensor, s_iso, diagonal_mehring, eigenvals, eigenvecs, symmetry, span, skew

    
    # Backcalculate Euler angles from eigenvector matrix
    def tensor_to_euler(self, shielding_tensor, eigenvals, eigenvecs, symmetry, rotation_mode, order): 
        """
        Take eigenvectors from diagonalization step and back-infere Euler angles
        Input
        :param symmetric_tensor: 3x3 original molecular frame tensor symmetrized
        :param eigenvals: eigenvalues
        :param eigenvecs: eigenvectors
        :param symmetry: tensor symmetry computed while diagonalization
        :param rotation_mode: AZYZ, PZYZ, AZXZ, PZXZ
        :param order: Ascending, Descending, Absascending, None
        Output
        :param alpha
        :param beta
        :param gamma
        """

        # Notify user which module has been called
        print("# -------------------------------------------------- #")
        print("# EULER ANGLES CALCULATION MODULE HAS BEEN REQUESTED #")
        print(f'\n')
                
        # Eigenvectors are the rotation matrix
        R = eigenvecs
        print(f'Rotation matrix is: \n{R}\n')

        print(f'Eigenvalues are: {eigenvals}')
        
        print(f'Shielding tensor is: \n{shielding_tensor}\n')

        symmetric_tensor = NMRFunctions.symmetrize_tensor(shielding_tensor)

        print(f'Shielding tensor after symmetrization is: \n{symmetric_tensor}\n')

        eigenvals, eigenvecs = NMRFunctions.sort_eigenvalues(eigenvals, eigenvecs, order)
        print(f'Sorted eigenvalues are: {eigenvals}\n')
        print(f'According to the chosen orde: {order}\n')

        tensor_pas = np.diag(eigenvals)
        print(f'Shielding tensor in PAS is: \n{tensor_pas}\n')

        # ---------------------------------------------------------------------------------------------------
        # Get Euler angles based on rotation mode
        if rotation_mode == 'AZYZ':
            print('Active ZYZ right hand rotation requested')

            beta = np.arccos(R[2,2]) # cos(beta) element

            if R[2,2] == 1:
                alpha = np.arccos(R[0,0])
                gamma = 0
                
            else:
                alpha = np.arctan2(R[1,2]/np.sin(beta), R[0,2]/np.sin(beta))
                gamma = np.arctan2(R[2,1]/np.sin(beta), -R[2,0]/np.sin(beta))

            # Check if Euler angle extraction worked by comparing the rotated tensor with the original one
            molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(tensor_pas, alpha, beta, gamma, rotation_mode)   
            original_molecular_tensor = R @ tensor_pas @ np.linalg.inv(R) 

            # Check if any eigenvector value is negative by comparing the two matrices
            if np.array_equal(np.round(molecular_tensor,3), np.round(original_molecular_tensor,3)):
                print('Original molecular tensor and backrotated tensor are equal')
                pass
            
            # fix the eigenvector matrix when different
            else:
                print('Eigenvectors values are negative, proceeding to fix rotations.')
                R = - R

                beta = np.arccos(R[2,2]) # cos(beta) element

                if R[2,2] == 1: # cos(beta) != 0
                    alpha = np.arccos(R[0,0])
                    gamma = 0
                    
                else: # gimbal lock
                    alpha = np.arctan2(R[1,2]/np.sin(beta), R[0,2]/np.sin(beta))
                    gamma = np.arctan2(R[2,1]/np.sin(beta), -R[2,0]/np.sin(beta))

            if symmetry == 2: # spherical symmetry
                print('The tensor has spherical symmetry, angles are all zero..')
                alpha = 0
                beta = 0
                gamma = 0

            if symmetry == 0: 
                print('The tensor doesnt have spherical symmetry, calculating angles..')
                
                if np.round(eigenvals[2], 3) == np.round(eigenvals[1], 3):

                    print('The tensor is axially symmetric with yy = zz')

                    if np.round(symmetric_tensor[1,2], 3) ==0 and np.round(symmetric_tensor[0,1], 3) == 0:
                        
                        if np.abs(np.arcsin(np.sqrt((symmetric_tensor[1,1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))) == np.pi/2:
                            
                            gamma = np.arcsin(np.sqrt((symmetric_tensor[1,1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                            beta = 0
                            alpha = 0
                            
                            molecular_tensor =  NMRFunctions.backrotate_diagonal_tensor_with_euler(tensor_pas, alpha, beta, gamma, rotation_mode) 
                        
                            # Check if any eigenvector value is negative by comparing the two matrices
                            if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                pass
                            
                            else:
                                gamma = np.arcsin(-np.sqrt((symmetric_tensor[1,1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                beta = 0
                                gamma = 0

                        else:
                            gamma = np.arcsin(np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                            beta = np.arcsin(np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                            alpha = 0
                            
                            molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(tensor_pas, alpha, beta, gamma, rotation_mode)
                            
                            if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                pass
                            
                            else:
                                gamma = np.arcsin(-np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                beta = np.arcsin(np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                alpha = 0
                            
                            molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(tensor_pas, alpha, beta, gamma, rotation_mode)
                            
                            if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                pass
                            
                            else:
                                gamma = np.arcsin(np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                beta = np.arcsin(-np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                alpha = 0
                            
                            molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(tensor_pas, alpha, beta, gamma, rotation_mode)
                            
                            if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                pass
                            
                            else:
                                gamma = np.arcsin(-np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                beta = np.arcsin(-np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                alpha = 0
                    
                    else:
                        gamma = np.arcsin(np.sqrt((symmetric_tensor[1,1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                        beta = np.arctan2(-symmetric_tensor[1,2] / (np.sin(gamma) * np.cos(gamma) * (eigenvals[0] - eigenvals[1])), symmetric_tensor[0,1] / (np.sin(gamma) * np.cos(gamma) * (eigenvals[0] - eigenvals[1])))
                        alpha = 0
                
                else:
                    print('The tensor is not axially symmetric')

            else:
                gamma = 0
        
            alpha = np.mod(alpha, 2*np.pi)
            beta = np.mod(beta, 2*np.pi)
            gamma = np.mod(gamma, 2*np.pi)

            if beta > np.pi:
                alpha = alpha - np.pi
                alpha = np.mod(alpha, 2*np.pi)
                beta = 2*np.pi - beta
            
            if beta >= np.pi/2:
                alpha = np.pi + alpha
                alpha = np.mod(alpha, 2*np.pi)
                beta = np.pi - beta
                beta = np.mod(beta, 2*np.pi)
                gamma = np.pi - gamma
                gamma = np.mod(gamma, 2*np.pi)

            if gamma >= np.pi:
                gamma = gamma - np.pi
            
            print(f'Backrotated molecular tensor from calculated angles is: \n{molecular_tensor}\n')

            print(f'Original molecular tensor is: \n{original_molecular_tensor}\n')

            print(f'Angles have been generated successfully with Active ZYZ roataion mode and are in units of radians \u03B1 = {alpha}, \u03B2 = {beta}, \u03B3 = {gamma}')
        
        # ---------------------------------------------------------------------------------------------------

        # Get Euler angles based on rotation mode
        if rotation_mode == 'AZXZ':
            print('Active ZXZ right hand rotation requested')
            beta = np.arccos(R[2,2]) # cos(beta) element

            if R[2,2] == 1:
                alpha = np.arccos(R[0,0])
                gamma = 0
                
            else:
                alpha = np.arctan2(R[1,2]/np.sin(beta), R[0,2]/np.sin(beta))
                gamma = np.arctan2(R[2,1]/np.sin(beta), -R[2,0]/np.sin(beta))

            # Check if Euler angle extraction worked by comparing the rotated tensor with the original one
            molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(symmetric_tensor, alpha, beta, gamma, 'PZYZ')   
            original_molecular_tensor = eigenvecs * symmetric_tensor * np.linalg.inv(eigenvecs) 

            # Check if any eigenvector value is negative by comparing the two matrices
            if np.array_equal(np.round(molecular_tensor,3), np.round(original_molecular_tensor,3)):
                print('Original molecular tensor and backrotated tensor are equal')
                pass
            
            # fix the eigenvector matrix when different
            else:
                print('Eigenvectors values are negative, proceeding to fix rotations.')
                eigenvecs = - eigenvecs

                beta = np.arccos(R[2,2]) # cos(beta) element

                if R[2,2] == 1: # cos(beta) != 0
                    alpha = np.arccos(R[0,0])
                    gamma = 0
                    
                else: # gimbal lock
                    alpha = np.arctan2(R[1,2]/np.sin(beta), R[0,2]/np.sin(beta))
                    gamma = np.arctan2(R[2,1]/np.sin(beta), -R[2,0]/np.sin(beta))

            if symmetry == 1: # spherical symmetry
                print('The tensor has spherical symmetry, angles are all zero..')
                alpha = 0
                beta = 0
                gamma = 0

            if symmetry == 0: 
                print('The tensor doesnt have spherical symmetry, calculating angles..')
                
                if np.round(eigenvals[2], 3) == np.round(eigenvals[1], 3):

                    print('The tensor is axially symmetric with yy = zz')

                    if np.round(symmetric_tensor[1,2], 3) ==0 and np.round(symmetric_tensor[0,1], 3) == 0:
                        
                        if np.abs(np.arcsin(np.sqrt((symmetric_tensor[1,1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))) == np.pi/2:
                            
                            gamma = np.arcsin(np.sqrt((symmetric_tensor[1,1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                            beta = 0
                            alpha = 0
                            
                            molecular_tensor =  NMRFunctions.backrotate_diagonal_tensor_with_euler(eigenvals, alpha, beta, gamma, 'PZYZ') 
                        
                            # Check if any eigenvector value is negative by comparing the two matrices
                            if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                pass
                            
                            else:
                                gamma = np.arcsin(-np.sqrt((symmetric_tensor[1,1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                beta = 0
                                gamma = 0

                        else:
                            gamma = np.arcsin(np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                            beta = np.arcsin(np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                            alpha = 0
                            
                            molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(eigenvals, alpha, beta, gamma, 'PZYZ')
                            
                            if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                pass
                            
                            else:
                                gamma = np.arcsin(-np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                beta = np.arcsin(np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                alpha = 0
                            
                            molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(eigenvals, alpha, beta, gamma, 'PZYZ')
                            
                            if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                pass
                            
                            else:
                                gamma = np.arcsin(np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                beta = np.arcsin(-np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                alpha = 0
                            
                            molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(eigenvals, alpha, beta, gamma, 'PZYZ')
                            
                            if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                pass
                            
                            else:
                                gamma = np.arcsin(-np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                beta = np.arcsin(-np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                alpha = 0
                    
                    else:
                        gamma = np.arcsin(np.sqrt((symmetric_tensor[1,1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                        beta = np.arctan2(-symmetric_tensor[1,2] / (np.sin(gamma) * np.cos(gamma) * (eigenvals[0] - eigenvals[1])), symmetric_tensor[0,1] / (np.sin(gamma) * np.cos(gamma) * (eigenvals[0] - eigenvals[1])))
                        alpha = 0
            
            else:
                gamma = 0
        
            alpha = np.mod(- gamma, 2*np.pi)
            beta = np.mod(- beta, 2*np.pi)
            gamma = np.mod(- alpha, 2*np.pi)

            if beta > np.pi:
                beta = 2*np.pi - beta
                gamma = gamma - np.pi
                gamma = np.mod(gamma, 2*np.pi)
            
            if beta >= np.pi/2:
                alpha = - (alpha - np.pi)
                alpha = np.mod(alpha, 2*np.pi)
                beta = - (beta - np.pi)
                beta = np.mod(beta, 2*np.pi)
                gamma = gamma + np.pi
                gamma = np.mod(gamma, 2*np.pi)

            if alpha >= np.pi:
                alpha = alpha - np.pi

            print(f'Backrotated molecular tensor from calculated angles is: \n{molecular_tensor}\n')

            print(f'Original molecular tensor is: \n{original_molecular_tensor}\n')
            
            print('Angles have been generated successfully with Passive ZYZ roataion mode')           

        else:
            print('Rotation mode specified is not supported yet')

        # Get the angles in degrees and round up values
        alpha = np.degrees(alpha).round(2)
        beta = np.degrees(beta).round(2)
        gamma = np.degrees(gamma).round(2)

        print(f'Euler angles in degrees are are: \n{alpha}, {beta}, {gamma}\n')
        print("# -------------------------------------------------- #")

        return alpha, beta, gamma, tensor_pas
            
    # Define a rotation of a given tensor using Euler angles
    @staticmethod
    def backrotate_diagonal_tensor_with_euler(diagonal_pas, alpha, beta, gamma, rotation_mode):
        '''
        Take the diagonalized PAS tensor and rotate it back to its original form using Euler angles
        Input
        :param: diagonal_pas: 3x3 diagonal tensor in PAS
        :param: alpha, beta, gamma: Euler angles calculated from the tensor_to_euler function
        :param: rotation_mode: define either active or passive rotations and the axis
        Output
        :param: molecular_tensor: tensor rotated back to its molecular frame
        '''

        # Choose rotation modes ZYZ
        if rotation_mode == 'AZYZ' or rotation_mode == 'PZYZ':
            
            R_z1 =  np.array([
                            [np.cos(alpha), -np.sin(alpha), 0],
                            [np.sin(alpha), np.cos(alpha), 0],
                            [0, 0, 1]]
                            )
            
            R_y2 =  np.array([
                            [np.cos(beta), 0, np.sin(beta)],
                            [0, 1, 0],
                            [-np.sin(beta), 0, np.cos(beta)]]
                            )

            R_z3 =  np.array([
                            [np.cos(gamma), -np.sin(gamma), 0],
                            [np.sin(gamma), np.cos(gamma), 0],
                            [0, 0, 1]]
                            )
            
            R_tot = R_z1 @ R_y2 @ R_z3

            # invert if passive
            if rotation_mode == 'PZYZ':
                inv_R_z1 = np.linalg.inv(R_z1)
                inv_R_y2 = np.linalg.inv(R_y2)
                inv_R_z3 = np.linalg.inv(R_z3)

                R_tot = inv_R_z1 @ inv_R_y2 @ inv_R_z3

        # Choose rotation modes ZXZ
        elif rotation_mode == 'AZXZ' or rotation_mode == 'PZXZ':

            R_z1 =  np.array([
                            [np.cos(alpha), -np.sin(alpha), 0],
                            [np.sin(alpha), np.cos(alpha), 0],
                            [0, 0, 1]]
                            )
            
            R_x2 =  np.array([
                            [1, 0, 0],
                            [0, np.cos(beta), -np.sin(beta)],
                            [0, np.sin(beta), np.cos(beta)]]
                            )

            R_z3 =  np.array([
                            [np.cos(gamma), -np.sin(gamma), 0],
                            [np.sin(gamma), np.cos(gamma), 0],
                            [0, 0, 1]]
                            )
            
            R_tot = R_z1 @ R_x2 @ R_z3

            # invert if passive
            if rotation_mode == 'PZXZ':
                inv_R_z1 = np.linalg.inv(R_z1)
                inv_R_x2 = np.linalg.inv(R_x2)
                inv_R_z3 = np.linalg.inv(R_z3)
                
                R_tot = inv_R_z1 @ inv_R_x2 @ inv_R_z3
        
        else:
            print('The input rotation mode is wrong, please choose one of the following formats: AZYZ, PZYZ, AZXZ, PZXZ')

        backrotated_tensor = R_tot @ diagonal_pas @ R_tot.T

        return backrotated_tensor
    
    # Symmetrize given tensor
    @staticmethod
    def symmetrize_tensor(tensor):
        return (tensor + tensor.T) / 2

    # Antisymmetrize given tensor
    @staticmethod
    def antisymmetrize_tensor(tensor):
        return (tensor - tensor.T) / 2
    
    # Sort tensor eigenvalues based on given order
    @staticmethod
    def sort_eigenvalues(eigenvals, eigenvecs, order):
        if order == 'Ascending':
            indices = np.argsort(eigenvals)
        elif order == 'Descending':
            indices = np.argsort(eigenvals)[::-1]
        elif order == 'Absascending':
            indices = np.argsort(np.abs(eigenvals))
        else:
            indices = np.arange(len(eigenvals))
        
        return eigenvals[indices], eigenvecs[:, indices]
    
    # Generate sets of equivalent euler angles based on AZYZ, PZYZ, AZXZ, PZXZ conventions
    def EqEulerSet(self, alpha, beta, gamma):
        """
        Generate a set of equivalent Euler angle sets.
        Input:
        angles: a tuple containing three Euler angles (alpha, beta, gamma)
        Output:
        A list containing four equivalent Euler angle sets
        """
        # Make all possible permutations
        euler_equivalents = [[alpha, beta, gamma],
                            [alpha, beta, gamma + np.pi],
                            [alpha + np.pi, np.pi - beta, np.pi - gamma],
                            [alpha + np.pi, np.pi - beta, 2 * np.pi - gamma]]
        
        print(f'Equivalent Euler angles are: {euler_equivalents}')

        return euler_equivalents
    
# ------------------------------------------------------------------------------
#                         ANALYSE ORCA OUTPUTS AND PLOT                        #
# ------------------------------------------------------------------------------
class OrcaAnalysis(NONCOVToolbox):
    """
    Class for data analysis of ORCA outputs. Only works for the EPR/NMR module outputs.
    """
    def __init__(self):
        super().__init__()

    def convert_path(self, output_file):
        """
        Read the output (.mpi8.out) file from an ORCA calculation and convert to readable string
        :param output_file: output file from orca in the form .mpi8.out
        """
        converted_path = output_file.strip('\"').replace("\\", "/")
        print(f"Normalized path using os.path: {converted_path}")
    
        return converted_path
           
    def count_jobs_number(self, output_file):
        """
        Read the output (.mpi8.out) file from an ORCA calculation and count how many jobs have been run.
        :param output_file: output file from orca in the form .mpi8.out
        """
        with open(output_file, 'r', encoding='utf-8') as f:
            # Count number of jobs in output file
            lines = f.readlines()
            count = 0
            for line in lines:
                if line.strip().startswith("COMPOUND JOB"):
                    count += 1
            return count

    def count_convergence(self, output_file):
        """
        Read the output (.mpi8.out) file from an ORCA calculation and count how many times the geometry converged.
        :param output_file: output file from orca in the form .mpi8.out
        """
        with open(output_file, 'r', encoding='utf-8') as f:
            # Count number of converged geometries in output file
            lines = f.readlines()
            count = 0
            for line in lines:
                if line.strip().startswith("HURRAY"):
                    count += 1
            return count        
        
    def extract_level_of_theory(self, output_file):
        """
        Read the output (.mpi8.out) file from an ORCA calculation and extract level of theory.
        :param output_file: output file from orca in the form .mpi8.out
        """
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()

                start_search = False
                
                # Search for the line containing "# Level of theory"
                for i, line in enumerate(lines):

                    if 'INPUT FILE' in line:
                        start_search = True
                        continue

                    if "# Level of theory" in line or '!' in line:
                        # Extract the line immediately after it, this wont work if ppl dont use my syntax
                        level_of_theory_line = lines[i + 1].strip()

                        # Remove the line number from the line - do i want to keep this?
                        #level_of_theory = level_of_theory_line.replace("| 10>", "").strip()
                        level_of_theory = re.sub(r'\|\d+>', "", level_of_theory_line).strip()

                        return level_of_theory
                    
                    if '****END OF INPUT****' in line:
                        break

                return "Level of theory not found in the file."
        
        except FileNotFoundError:
            return f"File '{output_file}' not found."
        except Exception as e:
            return f"An error occurred: {str(e)}"
        
    def extract_molecule_names(self, filename):
        """
        Read the output (.mpi8.out) file from an ORCA calculation and extract list of molecules.
        :param output_file: output file from orca in the form .mpi8.out
        """
        molecule_names = []

        # Define the pattern to match lines starting with "* xyzfile" 
        pattern = re.compile(r'\|\s*\d+>\s*\* xyzfile -?\d+ \d+ (\S+\.xyz)')

        with open(filename, 'r') as file:

            for line in file:
                match = pattern.search(line)

                if match:
                    molecule_name = match.group(1)
                    molecule_names.append(molecule_name)

        return molecule_names

    def split_orca_output(self, scratch_dir, output_file):
        """
        This function splits the huge ORCA multi-job file into individual job files.
        Then sews it back with the initial output lines from ORCA so that the files can
        be opened in Avogadro, otherwise it doesnt work.
        :param output_file: output file from orca in the form .mpi8.out
        """
        if not os.path.isfile(output_file):
            print(f"Error in OrcaAnalysis: ORCA output file '{output_file}' not found, please define.")
            return 

        # Make use of RegEx for matching JOB lines
        job_matching = re.compile(r'COMPOUND\s+JOB\s+(\d+)')

        # Extract initial ORCA text before job files to append to splitted files
        initial_content = self.extract_initial_output_content(output_file, job_matching)

        with open(output_file, 'r') as f:
            current_job_n = None # current job number
            current_job_content = [] # job specific info

            for line in f:
                match = job_matching.search(line) # regex match search
                if match:
                    # if match is found, write to file
                    if current_job_n is not None:
                        # Ensure you have the folder to save the files
                        output_folder = f'OrcaAnalysis/split_orca_output/splitted_orca_job{current_job_n}.out'
                        output_file_path = os.path.join(scratch_dir, output_folder)
                        
                        # If doesnt exist, create
                        output_dir = os.path.dirname(output_file_path)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        print(f'Output file path is {output_file_path}')
                        
                        with open(output_file_path, 'w') as out:
                            out.writelines(initial_content + current_job_content) # initial orca info + job specific info
                        
                        print(f'Wrote job {current_job_n} to {output_file_path}')

                    current_job_n = match.group(1)
                    current_job_content = [] # reset

                current_job_content.append(line)

            # write last job to file
            if current_job_n is not None:
                # Ensure you have the folder to save the files
                output_folder = f'OrcaAnalysis/split_orca_output/splitted_orca_job{current_job_n}.out'
                output_file_path = os.path.join(scratch_dir, output_folder)
                
                with open(output_file_path, 'w') as out:
                    out.writelines(initial_content + current_job_content)
                
                print(f'Wrote job {current_job_n} to {output_file_path}')

        print(f'ORCA output has been split into {current_job_n} sub files for further analysis')

    def extract_initial_output_content(self, output_file, job_matching):
        """
        Add the initial ORCA file until JOB NUMBER to each file to be read by Avogadro.
        :param output_file: output file from orca in the form .mpi8.out
        :param job_matching: Regex expression for the "JOB NUMBER X" line
        """
        initial_content = []
        with open(output_file, 'r') as f:
            for line in f:
                match = job_matching.search(line) # if u match stop there
                if match:
                    break # break as soon as you see the first job line 
                initial_content.append(line) # and append to file
        return initial_content
    

    def extract_tensor_data(self, splitted_output_file):
        """
        Load the splitted orca output files and read total tensor and its components
        Input:
        splitted_output_file: orca output file splitted by number of jobs
        Output:
        :shielding_dia: diagonal diamagnetic shielding tensor components
        :shielding_para: diagonal paramagnetic shielding tensor components  
        :shielding_tot: diagonal total shielding tensor components
        :nuc_identity: nucleus associated with the tensor values
        """

        # Dictionaries to store diamagnetic tensor components for each nucleus type
        shielding_dia = {}  

        # Dictionaries to store paramagnetic tensor components for each nucleus type
        shielding_para = {}  

        # Dictionaries to store total tensor components for each nucleus type
        shielding_tot = {}  

        # Read shielding tensor from file - continue from here
        try:
            with open(splitted_output_file, 'r') as f:
                lines = f.readlines()

            # nucleus marker
            current_nucleus = None

            # marker for shielding search
            start_search = False

            for i, line in enumerate(lines):
                # Start searching after encountering the CHEMICAL SHIFTS flag
                if 'CHEMICAL SHIELDINGS (ppm)' in line:
                    start_search = True
                    continue

                if start_search:
                    line = line.strip()
                    if line.startswith('Nucleus'):
                        if current_nucleus is not None:
                            shielding_dia[current_nucleus] = current_dia_shielding
                            shielding_para[current_nucleus] = current_para_shielding
                            shielding_tot[current_nucleus] = current_tot_shielding
                        
                        # add the nucleus information to file
                        nucleus_info = line.split()[1:]
                        current_nucleus = f"Nucleus {' '.join(nucleus_info)}"
                        current_dia_shielding = []
                        current_para_shielding = []
                        current_tot_shielding = []

                    # Extract the various tensor components here
                    elif line.startswith('Diamagnetic contribution'):
                        try:
                            dia_tensor_matrix = []
                            for j in range(1, 4):
                                dia_tensor_matrix.append([float(x) for x in lines[i+j].split()])
                            current_dia_shielding = dia_tensor_matrix
                        except (ValueError, IndexError):
                            print('Error encountered when extracting Diamagnetic tensor components')
                            continue

                    elif line.startswith('Paramagnetic contribution'):
                        try:
                            para_tensor_matrix = []
                            for j in range(1, 4):
                                para_tensor_matrix.append([float(x) for x in lines[i+j].split()])
                            current_para_shielding = para_tensor_matrix
                        except (ValueError, IndexError):
                            print('Error encountered when extracting Paramagnetic tensor components')
                            continue

                    elif line.startswith('Total shielding'):
                        try:
                            tot_tensor_matrix = []
                            for j in range(1, 4):
                                tot_tensor_matrix.append([float(x) for x in lines[i+j].split()])
                            current_tot_shielding = tot_tensor_matrix
                        except (ValueError, IndexError):
                            print('Error encountered when extracting Total shielding tensor components')
                            continue
                
                # stop extraction at the end of the tensor nmr block of the output
                if 'CHEMICAL SHIELDING SUMMARY (ppm)' in line:
                    break

            # Store last nucleus data
            if current_nucleus is not None:
                shielding_dia[current_nucleus] = current_dia_shielding
                shielding_para[current_nucleus] = current_para_shielding
                shielding_tot[current_nucleus] = current_tot_shielding 

        except FileNotFoundError:
            print(f"File '{splitted_output_file}' not found.")
            return {}, {}, {}, []
        
        # Extract all the nuclear identities in xyz file
        nuc_identity = list(shielding_tot.keys())

        return shielding_dia, shielding_para, shielding_tot, nuc_identity
    
    def extract_spin_spin_couplings(self, splitted_output_file):
        # Dictionaries to store pairwise nuclear properties
        r_AB = {}
        ssDSO_matrices = {}
        ssPSO_matrices = {}
        ssFC_matrices = {}
        ssSD_matrices = {}

        try:
            with open(splitted_output_file, 'r') as f:
                lines = f.readlines()
            
            # Initialize variables to store data
            current_nuc_pair = None
            current_ssDSO = None
            current_ssPSO = None
            current_FC = None
            current_SD = None
            
            start_search = False
            
            for i, line in enumerate(lines):
                if '*** THE CP-SCF HAS CONVERGED ***' in line:
                    start_search = True
                    continue
                
                if start_search:
                    line = line.strip()
                    
                    # Extract nucleus information
                    if 'NUCLEUS A =' in line:
                        match = re.search(r'NUCLEUS A = (\w+)\s+(\d+) NUCLEUS B = (\w+)\s+(\d+)', line)
                        if match:
                            nucleus_a = f"{match.group(1)} {match.group(2)}"
                            nucleus_b = f"{match.group(3)} {match.group(4)}"
                            current_nuc_pair = (nucleus_a, nucleus_b)
                    
                    # Extract r(AB) value
                    elif 'r(AB)' in line:
                        match = re.search(r'r\(AB\) =\s+([\d\.]+)', line)
                        if match:
                            current_r_ab = float(match.group(1))
                            r_AB[current_nuc_pair] = current_r_ab
                    
                    # Extract Diamagnetic tensor matrix
                    elif line.startswith('Diamagnetic contribution (Hz)'):
                        try:
                            ssDSO_tensor_matrix = []
                            for j in range(1, 4):
                                ssDSO_tensor_matrix.append([float(x) for x in lines[i+j].split()])
                            current_ssDSO = ssDSO_tensor_matrix
                            ssDSO_matrices[current_nuc_pair] = current_ssDSO
                        except (ValueError, IndexError):
                            print('Error extracting Diamagnetic tensor components')
                    
                    # Extract Paramagnetic tensor matrix
                    elif line.startswith('Paramagnetic contribution (Hz)'):
                        try:
                            ssPSO_tensor_matrix = []
                            for j in range(1, 4):
                                ssPSO_tensor_matrix.append([float(x) for x in lines[i+j].split()])
                            current_ssPSO = ssPSO_tensor_matrix
                            ssPSO_matrices[current_nuc_pair] = current_ssPSO
                        except (ValueError, IndexError):
                            print('Error extracting Paramagnetic tensor components')
                    
                    # Extract Fermi-contact tensor matrix
                    elif line.startswith('Fermi-contact contribution (Hz)'):
                        try:
                            FC_tensor_matrix = []
                            for j in range(1, 4):
                                FC_tensor_matrix.append([float(x) for x in lines[i+j].split()])
                            current_FC = FC_tensor_matrix
                            ssFC_matrices[current_nuc_pair] = current_FC
                        except (ValueError, IndexError):
                            print('Error extracting Fermi-contact tensor components')
                    
                    # Extract Spin-dipolar tensor matrix
                    elif line.startswith('Spin-dipolar contribution (Hz)'):
                        try:
                            SD_tensor_matrix = []
                            for j in range(1, 4):
                                SD_tensor_matrix.append([float(x) for x in lines[i+j].split()])
                            current_SD = SD_tensor_matrix
                            ssSD_matrices[current_nuc_pair] = current_SD
                        except (ValueError, IndexError):
                            print('Error extracting Spin-dipolar tensor components')

        except FileNotFoundError:
            print(f"File '{splitted_output_file}' not found.")
            return {}, {}, {}, {}, {}, {}, {}
        
        return r_AB, ssDSO_matrices, ssPSO_matrices, ssFC_matrices, ssSD_matrices

    def extract_mayer_bond_order(self, splitted_output_file):
        """
        Read split orca file and extract tuples of Meyer bond orders between nuclei
        """
        bond_orders = {}
        start_reading = False

        with open(splitted_output_file, 'r') as f:
            for line in f:
                if 'Mayer bond orders larger than' in line:
                    start_reading = True
                    continue
                
                if 'TIMINGS' in line:
                    break

                if start_reading:
                    matches = re.findall(r'B\(\s*(\d+-[A-Z])\s*,\s*(\d+-[A-Z])\s*\)\s*:\s*([-\d.]+)', line)

                    for nucleus1, nucleus2, bond_order in matches:
                        bond_order_value = float(bond_order)

                        # For nucleus1, add (interacting nucleus2, bond order)
                        if nucleus1 not in bond_orders:
                            bond_orders[nucleus1] = []
                        bond_orders[nucleus1].append((nucleus2, bond_order_value))

                        # For nucleus2, add (interacting nucleus1, bond order)
                        if nucleus2 not in bond_orders:
                            bond_orders[nucleus2] = []
                        bond_orders[nucleus2].append((nucleus1, bond_order_value))

        return bond_orders
    
    # Here extract NCS analysis data
    
    # Here extract NPA data
    
    # Here extract NBO data
    
    
    def extract_xyz_coords(self, splitted_output_file):
        """
        From each property file extract x,y,z coordinates and nuclear identity to append
        to Machine Learning database
        :param: nuc_coords: list of nuclear Cartesian coordinates
        """
        nuc_coords = []

        coords_list = []

        start_reading = False

        with open(splitted_output_file, 'r') as f:
            for line in f:
                if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                    start_reading = True
                    continue
                if 'CARTESIAN COORDINATES (A.U.)' in line:
                    break

                if start_reading:
                    match = re.match(r'\s*([A-Z]+)\s*([-.\d]+)\s*([-.\d]+)\s*([-.\d]+)', line)

                    if match:
                        element = match.group(1)
                        x = float(match.group(2))
                        y = float(match.group(3))
                        z = float(match.group(4))

                        #atom_number = len(nuclear_identity)

                        coords = f'{element}: {x}, {y}, {z}'
                        coords_list.append(coords)
        
        for coord in coords_list:
            element, xyz = coord.split(':')
            x, y, z = xyz.split(', ')
            new_coord = f'{element} {x} {y} {z}'
            nuc_coords.append(new_coord)
        
        nuc_coords = [re.split(r'\s+', row.strip()) for row in nuc_coords]

        return nuc_coords

# ------------------------------------------------------------------------------
#                        MOLECULAR VISUALIZATION AND PLOTTING
# ------------------------------------------------------------------------------
class MolView(NONCOVToolbox):
    """
    Molecular visualization class
    """
    def __init__(self):
        super().__init__()

    def radiusovaloid(self, sxx, syy, szz, alpha, beta, gamma, theta, phi):
        '''
        to check
        '''
        r_ov = (sxx * (np.sin(gamma) * np.sin(alpha - phi) * np.sin(theta) + np.cos(gamma) * (np.cos(theta) * np.sin(beta) - np.cos(beta) * np.cos(alpha - phi) * np.sin(theta))) ** 2
            + syy * (np.cos(theta) * np.sin(beta) * np.sin(gamma) - (np.cos(beta) * np.cos(alpha - phi) * np.sin(gamma) + np.cos(gamma) * np.sin(alpha - phi) * np.sin(theta))) ** 2
            + szz * (np.cos(beta) * np.cos(theta) + np.cos(alpha - phi) * np.sin(beta) * np.sin(theta)) ** 2)
        
        print(f'Radius of ovaloid representation is: {r_ov}')
        
        return r_ov
    
    def plot_tensor_shielding(S_dia, S_para, S_tot, nuclear_identities, 
                            displacement_steps_distance, min_distance_value, max_distance_value, 
                            scratch_dir, iteration):
        """
        Plots tensor shielding values (diamagnetic, paramagnetic, and total) for each nucleus.

        :param S_dia: List of dictionaries with diamagnetic tensor components.
        :param S_para: List of dictionaries with paramagnetic tensor components.
        :param S_tot: List of dictionaries with total tensor components.
        :param nuclear_identities_2: List of nuclear identity strings.
        :param displacement_steps_distance: List of displacement steps.
        :param min_distance_value: Minimum distance value for shading the NONCOV effective region.
        :param max_distance_value: Maximum distance value for shading the NONCOV effective region.
        :param scratch_dir: Directory to save the plots.
        :param iteration: Current iteration number (used for filename).
        """
        # Create a folder to save the tensor plots as PDFs and JPEGs if it doesn't exist
        pas_tensors_figures_folder = os.path.join(scratch_dir, 'OrcaAnalysis/tensor_plots')
        os.makedirs(pas_tensors_figures_folder, exist_ok=True)

        # Loop over each nucleus identity and plot its tensor shielding values
        for nucleus_list in nuclear_identities:
            for nucleus_key in nucleus_list:
                if isinstance(nucleus_key, list):
                    nucleus_key = "_".join(map(str, nucleus_key))
                    
                # Extract individual contributions to shielding values for the current nucleus from each dictionary
                nucleus_values_S_dia = [d.get(nucleus_key, [])[0] for d in S_dia]
                nucleus_values_S_para = [d.get(nucleus_key, [])[0] for d in S_para]
                nucleus_values_S_tot = [d.get(nucleus_key, [])[0] for d in S_tot]

                # Split the nucleus_key into a tuple (nucleus number, element)
                nucleus = tuple(nucleus_key.split())

                # Plot the shielding values for the current nucleus
                plt.plot(nucleus_values_S_dia, marker='o', linestyle='-', color='darkblue', label=r'$\sigma$_dia_11')
                plt.plot(nucleus_values_S_para, marker='o', linestyle='-', color='orangered', label=r'$\sigma$_para_11')
                plt.plot(nucleus_values_S_tot, marker='o', linestyle='-', color='gold', label=r'$\sigma$_tot_11')

                # Highlight the NONCOV effective region (optional, can be commented out if not needed)
                if min_distance_value is not None and max_distance_value is not None:
                    plt.axvspan(min_distance_value, max_distance_value, alpha=0.2, color='grey', label='NONCOV \n effective region')
                
                # Set labels and title
                plt.xlabel('Displacement from initial geometry / Å')
                plt.ylabel('Shielding / ppm')
                plt.title(f'Nucleus {nucleus[1]} {nucleus[2]} - Iteration {iteration}')
                
                # Display legend
                plt.legend(loc='best')
                
        # Save the plot as a PDF in the output folder
        pdf_filename = os.path.join(pas_tensors_figures_folder, f'nucleus_{nucleus[1]}_iteration_{iteration}.pdf')
        plt.savefig(pdf_filename, bbox_inches='tight')

        # Save the plot as a JPEG in the output folder
        jpg_filename = os.path.join(pas_tensors_figures_folder, f'nucleus_{nucleus[1]}_iteration_{iteration}.jpg')
        plt.savefig(jpg_filename, bbox_inches='tight')
        
        # Show the plot (optional, can be commented out if you don't want to display the plots)
        #plt.show()

        # Clear the current figure for the next iteration
        plt.clf()

    def euler_angles_to_rotation_matrix(alpha, beta, gamma):
        """
        Convert Euler angles to rotation matrix.
        :param alpha: Rotation angle around the Z-axis
        :param beta: Rotation angle around the X-axis
        :param gamma: Rotation angle around the Z-axis
        :return: Rotation matrix
        """
        r = R.from_euler('zyz', [alpha, beta, gamma], degrees=True)
        return r.as_matrix()

    def plot_3D_tensor(tensor, label, color, ax):
        """
        Plot a 3D tensor (as ellipsoid).
        :param tensor: The tensor to plot
        :param label: Label for the tensor
        :param color: Color of the tensor
        :param ax: The axes to plot on
        """
        # Compute the ellipsoid coordinates - replace here with radius ovaloid
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Scale ellipsoid by the tensor eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(tensor)
        ellipsoid = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), np.array([x.flatten(), y.flatten(), z.flatten()])))
        x, y, z = ellipsoid.reshape((3, *x.shape))

        ax.plot_surface(x, y, z, color=color, alpha=0.5)
        ax.set_title(label)

    def plot_tensor_principal_axes(tensor_pas, tensor, color):
        """
        Plot the principal axes of a tensor.
        :param ax: The axes to plot on
        :param tensor: The tensor whose axes to plot
        :param color: Color of the axes
        """
        origin = np.array([0, 0, 0])
        eigenvalues, eigenvectors = np.linalg.eig(tensor)
        
        for i in range(3):
            tensor_pas.quiver(*origin, *eigenvectors[:, i], length=eigenvalues[i], color=color, label=f'Axis {i+1}')

        tensor_pas.set_xlim([-1, 1])
        tensor_pas.set_ylim([-1, 1])
        tensor_pas.set_zlim([-1, 1])
        tensor_pas.set_xlabel('X')
        tensor_pas.set_ylabel('Y')
        tensor_pas.set_zlabel('Z')

    def plot_3D_tensors_and_axes(tensor_pas, alpha, beta, gamma):
        """
        Plot the original tensor, rotated tensor, and their axes.
        :param tensor_pas: Diagonal tensor in PAS
        :param alpha: Rotation angle around the Z-axis (degrees)
        :param beta: Rotation angle around the X-axis (degrees)
        :param gamma: Rotation angle around the Z-axis (degrees)
        """
        # Compute the rotation matrix from Euler angles
        rotation_matrix = MolView.euler_angles_to_rotation_matrix(alpha, beta, gamma)
        
        # Compute the rotated tensor
        tensor_rotated = rotation_matrix @ tensor_pas @ np.linalg.inv(rotation_matrix)

        fig = plt.figure(figsize=(16, 12))

        # Original Tensor Plot
        ax1 = fig.add_subplot(221, projection='3d')
        MolView.plot_3D_tensor(tensor_pas, 'Original Tensor', 'blue', ax1)
        
        # Rotated Tensor Plot
        ax2 = fig.add_subplot(222, projection='3d')
        MolView.plot_3D_tensor(tensor_rotated, 'Rotated Tensor', 'red', ax2)

        # Original Tensor Axes
        ax3 = fig.add_subplot(223, projection='3d')
        MolView.plot_tensor_principal_axes(ax3, tensor_pas, 'blue')
        ax3.set_title('Original Tensor Axes')

        # Rotated Tensor Axes
        ax4 = fig.add_subplot(224, projection='3d')
        MolView.plot_tensor_principal_axes(ax4, tensor_rotated, 'red')
        ax4.set_title('Rotated Tensor Axes')

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------------------
#                        STRUCTURAL CHANGES AND CONFORMERS 
# ------------------------------------------------------------------------------
class StructureModifier(NONCOVToolbox):
    """
    Take an input structure with two fragments and displace one fragment away.
     General remarks:
     - only works for displacing two fragments
     - request NormaltoPlane when there is an aromatic moiety. This will displace
       $fragment2 away in the direction of the normal to the aromatic plane
     - request CentroidtoCentroid for the other cases. This will displace the
       two fragments away from each other in the direction connecting the two
       centroids. If the initial coords are negative it can be that the direction
       is also negative, please act accordingly.
    """
    def __init__(self):
        super().__init__()
    
    #-------------------------------------------------------------------#
    
    def UserChoice(self, mode):
        """User selects either NormaltoPlane or CentroidtoCentroid
           :param mode: 'CentroidtoCentroid' or 'NormaltoPlane'
        """
        print(f'You selected: {mode}\n')
        
        return mode
    
    #-------------------------------------------------------------------#

    def read_atomic_coord(self, file_path):
        """
        This function reads the geometry optimized atomic coordinates of the two fragments
        you want to displace, the input is the classical .xyz coordinate file that can
        be then feeded to ORCA after displacement.
        :param file_path: path of the atomic coordinate file you want to displace
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()[2:]  # skip the first two lines (atom count and comment)
            coordinates = []
            atom_identities = [] # this is the one storing atom identity info, need to append it at a later step
            for line in lines:
                atom_data = line.split()
                atom_identity = atom_data[0] 
                coord = [float(atom_data[1]), float(atom_data[2]), float(atom_data[3])]
                coordinates.append(coord)
                atom_identities.append(atom_identity)
            coordinates = np.array(coordinates)
            atom_identities = np.array(atom_identities)
            return coordinates, atom_identities

    #-------------------------------------------------------------------#

    def calculate_centroids(self, coordinates):
        """
        This function computes centroids for the defined fragments.
        :param coordinates: molecular model xyz coordinates for centroid calculations
        """
        num_atoms = len(coordinates)
        centroids = []
        for fragment in coordinates:
            centroid = np.sum(fragment, axis=0) / len(fragment)
            centroids.append(centroid)
        return np.array(centroids)

    def write_centroids(self, file_path, centroids):
        """
        This function writes centroids to file for furhter manipulation.
        :param file_path: path where the centroids are written
        :param centroids: centroid file with coordinates
        """
        with open(file_path, 'w') as f:
            f.write(f'Centroid coordinates:\n')
            for centroid in centroids:
                f.write(f'{centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f}\n')

    #-------------------------------------------------------------------#

    def plot_starting_molecular_fragments(self, coords1, coords2, centroids):
        """
        Plot the initial molecular fragments in 3D
        :param coords1: coordinates fragment 1, specified in input file
        :param coords2: coordinates fragment 2, specified in input file
        :param centroids: coordinates of the respective centroids, given in centroid file
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(coords1[:, 0], coords1[:, 1], coords1[:, 2], color='blue', label='Molecule 1')
        ax1.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], color='red', label='Molecule 2')
        ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='green', marker='x', s=100, label='Centroids')
        ax1.legend()

    #-------------------------------------------------------------------#

    def assign_molecule_fragments(self, coordinates, input_file):
        """
        Assign the respective fragments to atom in xyz list
        :param coordinates: atomic xyz coordinates
        :param input_file: specifies the numbering of the atoms of each fragment
        """
        with open(input_file, 'r') as f:
            lines = f.readlines()
            fragment1_indices = []
            fragment2_indices = []
            ring_frag1_indices = []
            current_fragment = None
            for line in lines:
                if line.strip() == "$fragment1":
                    current_fragment = fragment1_indices
                elif line.strip() == "$fragment2":
                    current_fragment = fragment2_indices
                elif line.strip() == "$ring_frag1":
                    current_fragment = ring_frag1_indices                   
                elif line.strip() == "$displacement":
                    break
                elif line.strip() == "$diss_lim":
                    break
                else:
                    index = int(line.strip()) - 1  # Convert 1-based index to 0-based index
                    current_fragment.append(index)    
        coords1 = coordinates[fragment1_indices]
        coords2 = coordinates[fragment2_indices]
        ring_coords = coordinates[ring_frag1_indices]
        return coords1, coords2, ring_coords

    #-------------------------------------------------------------------#

    def displace_fragment(self, coords1, displacement_direction, displacement_step, i):
        """
        This function displaces the fragment along the displacement_direction vector.
        :param coords1: coordinates of fragment 1 to be displaced
        :param displacement_direction: displace fragments along the direction connecting the two centroids
        :param displacement_step: how many angstroem to displace, specified in input file
        :param i: for the loop over displacements, specifies how many structures are generated. In future will be the dissociation limit value
        """
        displacement_direction /= np.linalg.norm(displacement_direction)  # Normalize the displacement direction vector
        displacement_vector = - displacement_direction * displacement_step * i # Displace along the normalized direction

        return coords1 + displacement_vector  # Apply displacement by adding the vector

    #-------------------------------------------------------------------#

    def write_displaced_xyz_file(self, file_path, coords_fixed, coords_displaced, atom_identities):
        """
        This function writes both the fixed and displaced coordinates to an XYZ file.
        :param file_path: where the structure files are written
        :param coords_fixed: coordinates of the fixed fragment, in this case fragment 2
        :param coords_displaced: coordinates of the displaced fragment, in this case fragment 1
        :param atom_identities: append to file the identities of each atom again, since they are lost in processing steps
        """
        file_path = Path(file_path)
        with open(file_path, 'w') as f:
            num_atoms = len(coords_fixed) + len(coords_displaced)
            f.write(f'{num_atoms}\n')
            f.write(f'Step {file_path.stem}\n')

            # Write fixed fragment coordinates
            for i, atom in enumerate(coords_fixed):
                f.write(f'{atom_identities[i]} {atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f}\n') 
            
            # Write displaced fragment coordinates
            for i, atom in enumerate(coords_displaced):
                f.write(f'{atom_identities[i+len(coords_fixed)]} {atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f}\n') # i+len(coord_fixed) to skip to the fixed fragment indices
            
    #-------------------------------------------------------------------#

    def count_fragments(self, input_file):
        with open(input_file, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                if line.strip().startswith("$fragment"):
                    count += 1
            return count

    #-------------------------------------------------------------------#
    #                 CENTROIDTOCENTROID FUNCTIONS
    #-------------------------------------------------------------------#

    def compute_distance_from_centroid(self, coord_displaced, centroids):
        """
        Calculate relative centroid distances for future analysis
        :param coords_displaced: coordinates of the 
        """

        # Convert the lists to NumPy arrays for element-wise operations
        coord_displaced = np.array(coord_displaced)

        # Assuming fragment_centroids contains two sets of centroids
        if len(centroids) == 2:
            fixed_centroid = centroids[1]
            fixed_centroid = np.array(fixed_centroid)

            # Compute distance 
            distance_to_centroid = np.linalg.norm(coord_displaced - fixed_centroid, axis=1)

            return distance_to_centroid
        else:
            print('Error: fragment_centroids should contain two sets of centroids, if more please modify Section 8')

    #-------------------------------------------------------------------#

    def write_distances_file(self, file_path, coords_displaced, distance_to_centroid, atom_identities, displacement_step):
        """
        This function writes the distances between fixed centroid and displaced coordinates to a file.
        :param file_path: where the distance files will be written
        :param coords_displaced: coordinates of the displaced fragment
        :param distance_to_centroid: distance of the displaced coordinates from centroid of the fixed fragment
        :param atom_identities: identities of each atom which are lost in the processing
        :param displacement_step: how many angstroem are we displacing this structures
        """
        file_path = Path(file_path)

        with open(file_path, 'w') as f:
            num_atoms = len(coords_displaced)
            f.write(f'Number of atoms: {num_atoms}\n')
            f.write(f'Step {file_path.stem}\n')
            f.write(f'Displacement step: {displacement_step} A\n')
            
            # Write displaced fragment coordinates
            for i, atom in enumerate(coords_displaced):
                f.write(f'{atom_identities[i]} {distance_to_centroid[i]}\n') 

    #-------------------------------------------------------------------#
    #                    NORMALTOPLANE FUNCTIONS
    #-------------------------------------------------------------------#
    @staticmethod
    def get_norm_arom_plane(ring_coords, moving_frag_centroid, tolerance=1e-1):
        """
        Compute the normal vector to a plane defined by three points. In this case
        the three aromatic carbons are chosen so that the resulting two vectors share
        the same origin.
        
        :param ring_coords: Coordinates of the six carbon atoms from the ring
        :param moving_frag_centroid: coordinates of the centroid of the fragment you want to move
        :param tolerance: specifies in Angstroem how much tolerance to complanarity the nuclei have
        """

        if len(ring_coords) < 3:
            raise ValueError("At least 3 points are required to define a plane.")

        # Calculate the centroid of the ring
        ring_centroid = StructureModifier.calculate_centroid(ring_coords)

        # Select any 3 non-collinear points from the ring to calculate normal vector
        vec1 = ring_coords[1] - ring_coords[0]
        vec2 = ring_coords[3] - ring_coords[0]

        # Compute the normal using the cross product
        normal_dir = np.cross(vec1, vec2)

        # Normalize the normal vector
        normal_dir /= np.linalg.norm(normal_dir)

        # Vector from ring centroid to moving fragment centroid
        vector_to_moving_frag = moving_frag_centroid - ring_centroid

        # Distance between the moving fragment centroid and the center of the aromatic ring
        distance_centroid_aromatics = np.linalg.norm(moving_frag_centroid - ring_centroid)
        distance_centroid_aromatics = distance_centroid_aromatics.round(2)

        print(f'Initial distance Centroid to Aromatic is: {distance_centroid_aromatics} Angstroms\n')

        # Check the direction of the normal vector, if they dont match, flip it
        if np.dot(normal_dir, vector_to_moving_frag) < 0:
            normal_dir = -normal_dir

        return normal_dir, ring_centroid

        #-------------------------------------------------------------------#
    @staticmethod
    def calculate_centroid(coords):
        """
        Get center of mass of any sets of coordinates.
        
        :param coords: coordinates you want to compute
        """
        return np.mean(coords, axis=0)

        #-------------------------------------------------------------------#
    @staticmethod
    def displace_fragment_along_normal(fragment_coords, normal_dir, displacement_step, diss_lim):
        """
        Displace the fragment 2 along the normal direction with respect to the plane for given set 
        of parameters
        
        :param fragment_coords: coordinates of the fragment to be displaced
        :param normal_dir: direction of displacement
        :param displacement_step: user defined value in Angstroem. You can find it in the input files
        :param: diss_lim: Dissociation limit user defined. Also in the input file
        """
        displaced_fragments = []

        for i in range(0, diss_lim):

            # Displace all fragment 2 atoms by the same step
            displaced_fragment = fragment_coords + normal_dir * displacement_step * (i + 1)
            displaced_fragments.append(displaced_fragment)

        return np.array(displaced_fragments)

        #-------------------------------------------------------------------#

    def NormaltoPlane(self, coords1, coords2, ring_coords, output_dir, molecule_name, diss_lim, displacement_step, atom_identities):
        """
        Take as input the two fragments and the ring coordinates and apply the 
        displacement transformations. Write to files at the end.
        
        :param coords1: Coordinates of the fixed fragment
        :param coords2: coordinates of the moving fragment
        :param ring_coords: subset of coords1 containing ring coordinates
        :param output_dir: where you want the structures to be saved
        :param molecule_name: name of the molecule you are displacing, needed it for saving files
        """

        # Calculate the centroid of fragment 2
        moving_frag_centroid = StructureModifier.calculate_centroid(coords2)

        try:
            # Calculate the normal vector and centroid of the plane for fragment 1 (ring)
            normal_dir, ring_centroid = StructureModifier.get_norm_arom_plane(ring_coords, moving_frag_centroid)

            # Displace the fragment 2 coordinates along the normal direction
            displaced_fragments = StructureModifier.displace_fragment_along_normal(coords2, normal_dir, displacement_step, diss_lim)

            # Save the coordinates for each displacement step
            for i, displaced_fragment in enumerate(displaced_fragments):

                # Write displaced structure to file
                output_file = os.path.join(output_dir, f'{molecule_name}_disp_struct_{i}.xyz')
                self.write_displaced_xyz_file(output_file, coords1, displaced_fragment, atom_identities)

            # Return the final displaced coordinates of fragment 2
            final_fragment_coords = displaced_fragments[-1]
            return final_fragment_coords

        except ValueError as e:
            print(e)

# ------------------------------------------------------------------------------
#               GENERATE A DATASET FOR MACHINE LEARNING APPLICATIONS
# ------------------------------------------------------------------------------ 
class MachineLearning(NONCOVToolbox):
    """
    Machine learning database
    """
    def __init__(self):
        super().__init__()
    
    def make_empty_nuc_prop_df(self, output_csv_path, db_name):
        # Headers of features = number of columns
        columns = ['Molecule', #flag
                    'Atom', 
                    'x_coord', #input to GNN
                    'y_coord', #input to GNN
                    'z_coord', #input to GNN
                    'sigma_iso',
                    'sigma_xx', #unsorted
                    'sigma_yy', #unsorted
                    'sigma_zz', #unsorted
                    'dia_sigma_xx', #unsorted
                    'dia_sigma_yy', #unsorted
                    'dia_sigma_zz', #unsorted
                    'para_sigma_xx', #unsorted
                    'para_sigma_yy', #unsorted
                    'para_sigma_zz', #unsorted
                    'sigma_11', #mehring - magnitude sort
                    'sigma_22', #mehring - magnitude sort
                    'sigma_33', #mehring - magnitude sort
                    's_tot_symmetry',
                    'span',
                    'skew'
                    ]
        
        # Create the dataframe
        df = pd.DataFrame(columns=columns)
        df_out = os.path.join(output_csv_path, db_name)
        
        df.to_csv(df_out, index=False)
        
        # Keep the user happy by saying that something happened
        print(f'The empty nuclear property dataset has been created and saved in: {df_out}')
        print('\n')

    def make_empty_pairwise_prop_df(self, output_csv_path, db_name):
        # Headers of features = number of columns
        columns = ['Molecule', 
                    'Atom_1', 
                    'Atom_2',
                    'r_12',
                    'x_coord_1', 
                    'y_coord_1', 
                    'z_coord_1', 
                    'x_coord_2', 
                    'y_coord_2', 
                    'z_coord_2',
                    'J_iso',
                    'J_FC_xx',
                    'J_FC_yy',
                    'J_FC_zz',
                    'J_DSO_xx',
                    'J_DSO_yy',
                    'J_DSO_zz',
                    'J_PSO_xx',
                    'J_PSO_yy',
                    'J_PSO_zz',
                    'J_SD_xx',
                    'J_SD_yy',
                    'J_SD_zz',
                    'Mayer_BO' 
                    ]
        
        # Create the dataframe
        df = pd.DataFrame(columns=columns)
        df_out = os.path.join(output_csv_path, db_name)
        
        df.to_csv(df_out, index=False)
        
        # Keep the user happy by saying that something happened
        print(f'The empty pairwise nuclear property dataset has been created and saved in: {df_out}')
        print('\n')

# ------------------------------------------------------------------------------
#                       DESCRIPTORS AND FEATURE SELECTION
# ------------------------------------------------------------------------------
class MolecularGraph(NONCOVToolbox):
    def __init__(self):
        super().__init__()
        self.graph = nx.Graph()

    def add_atom(self, atom_index, atom_type, coordinate):
        self.graph.add_node(atom_index, atom_type=atom_type, coordinate=coordinate[:3])
    
    def add_bond(self, atom1_index, atom2_index, bond_type="covalent"):
        self.graph.add_edge(atom1_index, atom2_index, bond_type=bond_type)

    def draw(self, mol_fragment=None):
        pos = nx.spring_layout(self.graph)  
        labels = nx.get_node_attributes(self.graph, 'atom_type')
        bond_types = nx.get_edge_attributes(self.graph, 'bond_type')
        
        colors = ['red', 'blue', 'orange', 'purple', 'darkseagreen', 'gold', ]
        
        node_colors = []
        
        edge_colors = ["gold" if bond == "noncovalent" else "black" for bond in bond_types.values()]

        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gold'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = [labels[node] for node in self.graph.nodes()]

        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color='skyblue',
                size=20,
                line_width=2)
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Molecular Graph',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[dict(
                                text="Molecule from XYZ file",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        fig.show()

    # clear
    def parse_xyz(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
               
        try:
            num_atoms = int(lines[0].strip()) # number of atoms in fragment
            atom_info = lines[2:2+num_atoms] # info on each atom with coordinates in 3D
        except ValueError:
            raise ValueError(f"Error parsing the number of atoms: '{lines[0]}' is not a valid integer.")
        
        atom_types = []
        coordinates = []

        for line in atom_info:
            parts = line.split()
            atom_type = parts[0] # nucleus
            x, y, z = map(float, parts[1:4]) # coordinates
            atom_types.append(atom_type)
            coordinates.append((x, y, z))

        coordinates = np.array(coordinates)
        return atom_types, coordinates

    # clear
    def calculate_distances(self, coordinates):
        num_atoms = len(coordinates)
        distances = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distances[i, j] = distances[j, i] = np.linalg.norm(coordinates[i] - coordinates[j])
        return distances
    
    # clear
    def plot_distance_matrix(self, distances, atom_labels):
        distance_matrix = squareform(pdist(distances, 'euclidean'))
        plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Distance')
        plt.title('Matrix of 2D distances')
        plt.xticks(ticks=np.arange(len(atom_labels)), labels=atom_labels)
        plt.yticks(ticks=np.arange(len(atom_labels)), labels=atom_labels)
        plt.show()

    # clear
    def plot_bond_matrix(self, bonds_matrix, atom_labels):
        plt.imshow(bonds_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Bool')
        plt.title('Matrix of 2D Bonds')
        plt.xticks(ticks=np.arange(len(atom_labels)), labels=atom_labels)
        plt.yticks(ticks=np.arange(len(atom_labels)), labels=atom_labels)
        plt.show()

    # clear
    def plot_bond_dist_matrix(self, bonds_matrix, distances, atom_labels):
        distance_matrix = squareform(pdist(distances, 'euclidean'))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im1 = axes[0].imshow(distance_matrix, cmap='viridis', interpolation='nearest')
        axes[0].set_title('Distance Matrix')
        fig.colorbar(im1, ax=axes[0])
        axes[0].set_xticks(np.arange(len(atom_labels)))
        axes[0].set_xticklabels(atom_labels, rotation=90)
        axes[0].set_yticks(np.arange(len(atom_labels)))
        axes[0].set_yticklabels(atom_labels)

        im2 = axes[1].imshow(bonds_matrix, cmap='gray_r', interpolation='nearest')
        axes[1].set_title('Bonds Matrix')
        fig.colorbar(im2, ax=axes[1])
        axes[1].set_xticks(np.arange(len(atom_labels)))
        axes[1].set_xticklabels(atom_labels, rotation=90)
        axes[1].set_yticks(np.arange(len(atom_labels)))
        axes[1].set_yticklabels(atom_labels)

        axes[2].imshow(distance_matrix, cmap='viridis', interpolation='nearest')
        im3 = axes[2].imshow(bonds_matrix, cmap='gray_r', interpolation='nearest', alpha=0.5)
        axes[2].set_title('Distance vs. Bonds')
        fig.colorbar(im3, ax=axes[2])
        axes[2].set_xticks(np.arange(len(atom_labels)))
        axes[2].set_xticklabels(atom_labels, rotation=90)
        axes[2].set_yticks(np.arange(len(atom_labels)))
        axes[2].set_yticklabels(atom_labels)

        plt.tight_layout()
        plt.show()

    # clear / need new logic
    def detect_bonds(self, atom_types, distances):
        covalent_radii = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66}  # Extend as needed
        bond_matrix = np.zeros(distances.shape, dtype=bool)
        for i, atom1 in enumerate(atom_types):
            for j, atom2 in enumerate(atom_types):
                if i < j:
                    max_bond_dist = covalent_radii[atom1] + covalent_radii[atom2] + 0.4  # tolerance
                    if distances[i, j] < max_bond_dist:
                        bond_matrix[i, j] = bond_matrix[j, i] = True
        return bond_matrix

    # clear / need new logic
    def detect_noncovalent_interactions(self, atom_types, distances):
        noncovalent_interactions = []
        for i, atom1 in enumerate(atom_types):
            for j, atom2 in enumerate(atom_types):
                if i < j:
                    if distances[i, j] > 2.5 and distances[i, j] < 4.0:  # Rough range for non-covalent interaction
                        interaction_type = "hydrogen_bond" if "H" in [atom1, atom2] else "vdW"
                        noncovalent_interactions.append((i, j, interaction_type))
        return noncovalent_interactions
    
    # clear / need new logic
    def plot_noncov_distance_map(self, noncovalent_interactions, atom_labels):
        # Determine the matrix size
        n = max(max(i[0], i[1]) for i in noncovalent_interactions) + 1

        interaction_matrix = np.zeros((n, n), dtype=int)
        vdw_matrix = np.zeros((n, n), dtype=bool)
        hb_matrix = np.zeros((n, n), dtype=bool)

        for i, j, interaction in noncovalent_interactions:
            interaction_matrix[i, j] = 1  # 1 for any interaction
            if interaction == 'vdW':
                vdw_matrix[i, j] = True
            elif interaction == 'hydrogen_bond':
                hb_matrix[i, j] = True

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        im1 = axes[0].imshow(vdw_matrix, cmap='Blues', interpolation='nearest')
        axes[0].set_title('vdW Interactions')
        fig.colorbar(im1, ax=axes[0])
        axes[0].set_xticks(np.arange(len(atom_labels)))
        axes[0].set_xticklabels(atom_labels, rotation=90)
        axes[0].set_yticks(np.arange(len(atom_labels)))
        axes[0].set_yticklabels(atom_labels)

        im2 = axes[1].imshow(hb_matrix, cmap='Reds', interpolation='nearest')
        axes[1].set_title('Hydrogen Bond Interactions')
        fig.colorbar(im2, ax=axes[1])
        axes[1].set_xticks(np.arange(len(atom_labels)))
        axes[1].set_xticklabels(atom_labels, rotation=90)
        axes[1].set_yticks(np.arange(len(atom_labels)))
        axes[1].set_yticklabels(atom_labels)

        axes[2].imshow(vdw_matrix, cmap='Blues', interpolation='nearest')
        im3 = axes[2].imshow(hb_matrix, cmap='Reds', interpolation='nearest', alpha=0.5)
        axes[2].set_title('Full NONCOV interactions')
        fig.colorbar(im3, ax=axes[2])
        axes[2].set_xticks(np.arange(len(atom_labels)))
        axes[2].set_xticklabels(atom_labels, rotation=90)
        axes[2].set_yticks(np.arange(len(atom_labels)))
        axes[2].set_yticklabels(atom_labels)

        plt.tight_layout()
        plt.show()

    def build_molecular_graph(self, atom_types, coordinates, covalent_bonds, noncovalent_interactions):
        mol_graph = MolecularGraph()

        # Add atoms to the graph
        for i, (atom_type, position) in enumerate(zip(atom_types, coordinates)):
            mol_graph.add_atom(i, atom_type, position)

        # Add covalent bonds
        for i in range(len(atom_types)):
            for j in range(i + 1, len(atom_types)):
                if covalent_bonds[i, j]:
                    mol_graph.add_bond(i, j, bond_type="covalent")

        # Add non-covalent interactions
        for i, j, interaction_type in noncovalent_interactions:
            mol_graph.add_bond(i, j, bond_type=interaction_type)

        return mol_graph
    
    def draw_subplots(self, covalent_bonds_graph, intramolecular_graph, intermolecular_graph, coordinates):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Draw Covalent Bonds Graph
        self._draw_graph(covalent_bonds_graph, coordinates, ax=axes[0], title='Covalent Bonds')

        # Draw Intramolecular Contacts Graph
        self._draw_graph(intramolecular_graph, coordinates, ax=axes[1], title='Intramolecular Contacts')

        # Draw Intermolecular Contacts Graph
        self._draw_graph(intermolecular_graph, coordinates, ax=axes[2], title='Intermolecular Contacts')
        
        plt.tight_layout()
        plt.show()

    def _draw_graph(self, graph, coordinates, ax, title):
        # Use the original coordinates as the positions for the nodes
        pos = {i: (coordinates[i][0], coordinates[i][1]) for i in graph.nodes()}  # X, Y coordinates

        labels = nx.get_node_attributes(graph, 'atom_type')
        bond_types = nx.get_edge_attributes(graph, 'bond_type')
        edge_colors = ["blue" if bond == "noncovalent" else "black" for bond in bond_types.values()]

        nx.draw(graph, pos, ax=ax, labels=labels, with_labels=True, node_size=500, node_color="skyblue", edge_color=edge_colors, font_size=8)
        ax.set_title(title)

    def build_covalent_bonds_graph(self, atom_types, coordinates, covalent_bonds):
        graph = nx.Graph()
        for i, (atom_type, position) in enumerate(zip(atom_types, coordinates)):
            graph.add_node(i, atom_type=atom_type, coordinate=position)
        for i in range(len(atom_types)):
            for j in range(i + 1, len(atom_types)):
                if covalent_bonds[i, j]:
                    graph.add_edge(i, j, bond_type="covalent")
        return graph

    def build_intramolecular_graph(self, atom_types, coordinates, covalent_bonds, noncovalent_interactions):
        graph = nx.Graph()
        for i, (atom_type, position) in enumerate(zip(atom_types, coordinates)):
            graph.add_node(i, atom_type=atom_type, coordinate=position)
        for i, j, interaction_type in noncovalent_interactions:
            if covalent_bonds[i, j]:  # Intramolecular if there's a covalent bond
                graph.add_edge(i, j, bond_type="intramolecular")
        return graph

    def build_intermolecular_graph(self, atom_types, coordinates, noncovalent_interactions):
        graph = nx.Graph()
        for i, (atom_type, position) in enumerate(zip(atom_types, coordinates)):
            graph.add_node(i, atom_type=atom_type, coordinate=position)
        for i, j, interaction_type in noncovalent_interactions:
            if interaction_type != "intramolecular":  # Intermolecular if not intramolecular
                graph.add_edge(i, j, bond_type="intermolecular")
        return graph
