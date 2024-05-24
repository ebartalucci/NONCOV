###################################################
#       SOURCE CODE FOR THE NONCOV PROJECT        #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 26.02.2024                 #
#               Last:  16.05.2024                 #
#               -----------------                 #
#             Stable release version              #
#                   v.0.0.1                       #
#                                                 #
###################################################

# This work is an attempt to write a script running on minimal import packages

# Import modules 
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re  # RegEx
import nmrglue as ng # NMRglue module for handling NMR spectra 

# Mother class - be careful with class inheritance here
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

        # Print acknowledgments
        try:
            with open('acknowledgments.txt', 'r') as f:
                acknowledgments = f.read()
                print(acknowledgments)
        except FileNotFoundError:
            print("Acknowledgments file not found. Please ensure 'acknowledgments.txt' is in the correct directory.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sys.exit(1)

    # ------------------------------------------------------------------------------
    #                           BIOMOLECULAR APPLICATIONS                          #
    # ------------------------------------------------------------------------------
    class AminoStat(NONCOVToolbox):
        """
        Collection of useful functions for working with proteins and sequences.
        """
        def __init__(self):
            # Print header and version
            print("\n\n          #################################################")
            print("          | --------------------------------------------- |")
            print("          |  Plot statistics of amino acids distribution  |")
            print("          |           in given protein sequence           |")
            print("          | --------------------------------------------- |")
            print("          |                       -                       |")
            print("          |           NMR FUNCTIONS COLLECTIONS           |")
            print("          |                                               |")
            print("          |               Ettore Bartalucci               |")
            print("          |     Max Planck Institute CEC & RWTH Aachen    |")
            print("          |            Worringerweg 2, Germany            |")
            print("          |                                               |")
            print("          #################################################\n")

        def space_prot_seq(self, prot_seq, spaced_prot_seq):
            """
            Take a sequence and add a space between each letter
            """
            try:
                with open(prot_seq, 'r') as f:
                    sequence = f.read()

                spaced_sequence = ' '.join(sequence)

                with open(spaced_prot_seq, 'w') as f:
                    f.write(spaced_sequence)

                print(f'Protein sequence from Uniprot now contains spaces between each amino acid letter and has been written to: {spaced_prot_seq}.')

            except FileNotFoundError:
                print('Input file not found, please specify')
            except Exception as e:
                print(f'An error occurred: {e}')

        def count_amino_acids(self, prot_seq, count_file):
            """
            Take spaced sequence and count how many of each amino acids you have
            """
            try:
                with open(prot_seq, 'r') as f:
                    sequence = f.read()

                amino_acid_count = {}
                for amino_acid in sequence:
                    if amino_acid in amino_acid_count:
                        amino_acid_count[amino_acid] += 1
                    else:
                        amino_acid_count[amino_acid] = 1

                with open(count_file, 'w') as f:
                    for amino_acid, count in amino_acid_count.items():
                        f.write(f"{amino_acid}: {count}\n")

                print('Amino acid counts written to amino_acid_count.txt')

            except FileNotFoundError:
                print('Input file not found. please specify')
            except Exception as e:
                print(f'An error occurred: {e}')

        def plot_amino_acid_statistics(self, count_file, plot_file):
            """
            Take the count of the amino acids and plot the histogram of the distribution
            """
            try:
                amino_acid_counts = {}
                total_count = 0

                with open(count_file, 'r') as f:
                    for line in f:
                        pairs = line.strip().split(': ')
                        if len(pairs) == 2:
                            amino_acid, count = pairs
                            amino_acid_counts[amino_acid] = int(count)
                            total_count += int(count)

                amino_acids = list(amino_acid_counts.keys())
                counts = list(amino_acid_counts.values())

                percentages = [count / total_count * 100 for count in counts]

                plt.bar(amino_acids, percentages)
                plt.xlabel('Amino Acid')
                plt.ylabel('Percentage (%)')
                plt.title('Amino Acids Distribution')

                for i in range(len(amino_acids)):
                    plt.text(amino_acids[i], percentages[i], f"{counts[i]} ({percentages[i]:.2f}%)", ha='center', va='bottom', rotation=90)

                plt.savefig(plot_file)
                plt.show()
                
            except FileNotFoundError:
                print('Count file not found. please specify')
            except Exception as e:
                print(f'An error occurred: {e}')

        def calculate_amino_acid_percentage(self, prot_seq):
            amino_acids_list = "ARNDCEQGHILKMFPSTWYV"  # List of 20 standard amino acids
            aa_percentage = {aa: prot_seq.count(aa) for aa in amino_acids_list}
            return aa_percentage

        def define_protein_domains(self):
            """
            Users defines the limit and names of the domains of the protein
            """
            
            print('Please define your domains according to Uniprot information.')
            try:
                n_domains = int(input('How many domains does your protein have? '))

                prot_domain_names = []
                prot_domain_boundaries = []

                for i in range(n_domains):
                    prot_domain_name = input(f'Enter name of domain {i+1}: ')
                    prot_domain_boundary = input(f'Enter boundaries for domain {i+1} position: ')

                    prot_domain_names.append(prot_domain_name)
                    prot_domain_boundaries.append(prot_domain_boundary)

                print(f'The domains of your protein are: {prot_domain_names} in regions {prot_domain_boundaries}')
            except Exception as e:
                print(f'An error occurred: {e}')
                
                # continue here with sequence walk

    # ------------------------------------------------------------------------------
    # NMR FUNCTIONS AND APPLICATIONS 
    # ------------------------------------------------------------------------------
    class NMRFunctions(NONCOVToolbox):
        """
        Collection of useful functions for working with NMR parameters.
        """
        def __init__(self):
            # Print header and version
            print("\n\n          #################################################")
            print("          | --------------------------------------------- |")
            print("          |         (NC)^2I.py: NMR Calculations          |")
            print("          |         for Noncovalent Interactions          |")
            print("          | --------------------------------------------- |")
            print("          |      CALLING EXTERNAL MODULE: NMRFunctions    |")
            print("          |                       -                       |")
            print("          |           NMR FUNCTIONS COLLECTIONS           |")
            print("          |                                               |")
            print("          |               Ettore Bartalucci               |")
            print("          |     Max Planck Institute CEC & RWTH Aachen    |")
            print("          |            Worringerweg 2, Germany            |")
            print("          |                                               |")
            print("          #################################################\n")
            pass

        # 3x3 Matrix diagonalization and PAS shielding tensor ordering in Mehring and Haberlen conventions
        def diagonalize_tensor(self, sxx, sxy, sxz, syx, syy, syz, szx, szy, szz):
            """
            Take NMR tensor elements as input and perform various operations, including diagonalization and ordering according to Mehring and Haberlen formalisms.
            Input
            :param sxx, sxy, sxz, syx, syy, syz, szx, szy, szz: individual tensor components for a 3x3 chemical shielding matrix
            Output
            :param diagonal_mehring: full diagonalized matrix in principal axis system according to sigma_11 < sigma_22 < sigma_33
            :param sigma_11: individual component
            :param sigma_22: individual component
            :param sigma_33: individual component

            :param diagonal_haberlen: full diagonalized matrix in principal axis system according to |sigma_yy - sigma_iso| < |sigma_xx - sigma_iso| < |sigma_zz - sigma_iso|
            :param sigma_XX: individual component
            :param sigma_YY: individual component
            :param sigma_ZZ: individual component
            """

            # Notify user which module has been called
            print("# -------------------------------------------------- #")
            print("# MATRIX DIAGONALIZATION FUNCTION HAS BEEN REQUESTED #")
            print(f'\n')
                    
            # Initialize shielding tensor matrix
            shielding_tensor = np.matrix([[sxx, sxy, sxz],
                                        [syx, syy, syz],
                                        [szx, szy, szz]]
                                        )
            
            print(f'Shielding Tensor is: \n{shielding_tensor}')
            print('Proceeding to transposing...\n')

            # Transpose matrix
            transposed = np.transpose(shielding_tensor)
            print(f'Transposed matrix is: \n{transposed}')
            print('Proceeding to symmetrization...\n')

            # Symmetrize tensor
            shielding_tensor = (shielding_tensor + transposed) / 2
            print(f'Symmetric tensor is: \n{shielding_tensor}')
            print('Proceeding to diagonalization...\n')

            # Calculate eigenvalues and vectors 
            eigenvals, eigenvecs = np.linalg.eig(shielding_tensor)
            eigenvals = eigenvals.round(5) # round them up
            eigenvecs = eigenvecs.round(5) # round them up
            print(f'Eigenvalues are: {eigenvals}, Eigenvectors are: \n{eigenvecs}')
            print('Proceeding to ordering eigenvalues and eigenvectors...\n')

            # Sort eigenvalues and eigenvectors based on magnitude of eigenvalues
            idx = np.argsort(np.abs(eigenvals))
            eigenvals_ordered = eigenvals[idx]
            eigenvecs_ordered = eigenvecs[:, idx]
            print(f'Magnitude-based ordering of eigenvalues is: {eigenvals_ordered}, and of eigenvectors is: \n{eigenvecs_ordered}.')
            print('Proceeding to diagonalization...\n')

            # Compute diagonal matrix, define eigenvector columns as variables and preforme matrix multiplication 
            diagonal = np.diag(eigenvals)
            print(f'Diagonalized tensor is: \n{diagonal}')
            print('Proceeding to compute isotropic shift...\n')

            # Compute isotropic shift
            s_iso = np.sum(np.diag(diagonal)) / 3
            s_iso = s_iso.round(5)
            print(f'Isotropic shift is: {s_iso} ppm')
            print('Proceeding to Haberlen ordering...\n')

            # Reorder matrix according to Haberlen convention
            diagonal_haberlen = np.argsort(np.diag(diagonal) - s_iso)
            sigma_XX = diagonal_haberlen[0]
            sigma_YY = diagonal_haberlen[1]
            sigma_ZZ = diagonal_haberlen[2]
            diagonal_haberlen = diagonal[diagonal_haberlen]
            # diagonal_haberlen = np.diag(np.diag(diagonal)[diagonal_haberlen])
            print(f'Diagonal tensor in Haberlen order is: \n{diagonal_haberlen}\n')
            print(f'''where:\n \u03C3_XX:{sigma_XX} \n \u03C3_YY:{sigma_YY} \n \u03C3_ZZ:{sigma_ZZ}''')
            print('Proceeding to Mehring ordering...\n')

            # Reorder matrix according to Mehring convention
            diagonal_mehring = sorted(np.diag(diagonal))
            sigma_11 = diagonal_mehring[0]
            sigma_22 = diagonal_mehring[1]
            sigma_33 = diagonal_mehring[2]
            diagonal_mehring = np.diag(diagonal_mehring)
            print(f'Diagonal tensor in Mehring order is: \n{diagonal_mehring}\n')
            print(f'''where:\n \u03C3_11:{sigma_11} \n \u03C3_22:{sigma_22} \n \u03C3_33:{sigma_33} \n''')
            print('Proceeding to extract tensor symmetry from eigenvalues...\n')

            # Extract symmetry of the tensor from its eigenvalues
            symmetry = len(eigenvals) - len(np.unique(np.round(eigenvals,7))) #check if here maybe i dont need the asymmetry parameter
            print(f'Symmetry of the tensor is: {symmetry}\n')
            print('Proceeding to Euler angles extraction from eigenvectors...\n')

            print("# -------------------------------------------------- #")

            return shielding_tensor, diagonal_mehring, diagonal_haberlen, eigenvals, eigenvecs, symmetry
        
        # Backcalculate Euler angles from eigenvector matrix
        def tensor_to_euler(self, symmetric_tensor, eigenvals, eigenvecs, symmetry, rotation_mode): 
            """
            Take eigenvectors from diagonalization step and back-infere Euler angles
            Input
            :param symmetric_tensor: 3x3 original molecular frame tensor symmetrized
            :param eigenvals: eigenvalues from diagonalized molecular frame tensor
            :param eigenvecs: 3x3 (rotation) matrix of eigenvectors
            :param symmetry: symmetry parameter of the tensor
            :param rotation_mode: AZYZ, PZYZ, AZXZ, PZXZ
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

            print(f'Symmetric tensor is: \n{symmetric_tensor}\n')

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
                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'AZYZ')   
                original_molecular_tensor = eigenvecs * diagonal_mehring * np.linalg.inv(eigenvecs) 

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
                                
                                molecular_tensor =  NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'AZYZ') 
                            
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
                                
                                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'AZYZ')
                                
                                if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                    pass
                                
                                else:
                                    gamma = np.arcsin(-np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                    beta = np.arcsin(np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                    alpha = 0
                                
                                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'AZYZ')
                                
                                if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                    pass
                                
                                else:
                                    gamma = np.arcsin(np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                    beta = np.arcsin(-np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                    alpha = 0
                                
                                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'AZYZ')
                                
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

                print('Angles have been generated successfully with Active ZYZ roataion mode')
            
            # ---------------------------------------------------------------------------------------------------
            # Get Euler angles based on rotation mode
            if rotation_mode == 'PZYZ':
                print('Passive ZYZ right hand rotation requested')
                beta = np.arccos(R[2,2]) # cos(beta) element

                if R[2,2] == 1:
                    alpha = np.arccos(R[0,0])
                    gamma = 0
                    
                else:
                    alpha = np.arctan2(R[0,2]/np.sin(beta), -R[1,2]/np.sin(beta))
                    gamma = np.arctan2(R[2,0]/np.sin(beta), R[2,1]/np.sin(beta))

                # Check if Euler angle extraction worked by comparing the rotated tensor with the original one
                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'AZXZ')   
                original_molecular_tensor = eigenvecs * diagonal_mehring * np.linalg.inv(eigenvecs) 

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
                        alpha = np.arctan2(R[0,2]/np.sin(beta), -R[1,2]/np.sin(beta))
                        gamma = np.arctan2(R[2,0]/np.sin(beta), R[2,1]/np.sin(beta))

                if symmetry == 1: # spherical symmetry
                    print('The tensor has spherical symmetry, angles are all zero..')
                    alpha = 0
                    beta = 0
                    gamma = 0

                if symmetry == 0: 
                    print('The tensor doesnt have spherical symmetry, calculating angles..')
                    
                    if np.round(eigenvals[2], 3) == np.round(eigenvals[1], 3):

                        print('The tensor is axially symmetric with yy = zz')

                        if np.round(symmetric_tensor[2,0], 3) ==0 and np.round(symmetric_tensor[1,0], 3) == 0:
                            
                            if np.abs(np.arcsin(np.sqrt((symmetric_tensor[0,0] - eigenvals[0]) / (eigenvals[1] - eigenvals[0])))) == 0 or np.abs(np.arcsin(np.sqrt((symmetric_tensor[0,0] - eigenvals[0]) / (eigenvals[1] - eigenvals[0])))) == np.pi:
                                
                                gamma = np.arcsin(np.sqrt((symmetric_tensor[0,0] - eigenvals[0]) / (eigenvals[1] - eigenvals[0])))
                                beta = 0
                                alpha = 0
                                
                                molecular_tensor =  NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'AZXZ') 
                            
                                # Check if any eigenvector value is negative by comparing the two matrices
                                if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                    pass
                                
                                else:
                                    gamma = np.arcsin(-np.sqrt((symmetric_tensor[0,1] - eigenvals[0]) / (eigenvals[1] - eigenvals[0])))
                                    beta = 0
                                    gamma = 0

                            else:
                                gamma = np.arcsin(np.sqrt((symmetric_tensor[0, 0] - eigenvals[0]) / (eigenvals[1] - eigenvals[0])))
                                #beta = np.arcsin(np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) - (eigenvals[0] - eigenvals[1]) * (np.sin(gamma) ** 2)) / ((eigenvals[2] - eigenvals[1]) *)))
                                alpha = 0
                                
                                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'PZYZ')
                                
                                if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                    pass
                                
                                else:
                                    gamma = np.arcsin(-np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                    beta = np.arcsin(np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                    alpha = 0
                                
                                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'PZYZ')
                                
                                if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                    pass
                                
                                else:
                                    gamma = np.arcsin(np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                    beta = np.arcsin(-np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                    alpha = 0
                                
                                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'PZYZ')
                                
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
                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'PZYZ')   
                original_molecular_tensor = eigenvecs * diagonal_mehring * np.linalg.inv(eigenvecs) 

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
                                
                                molecular_tensor =  NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'PZYZ') 
                            
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
                                
                                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'PZYZ')
                                
                                if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                    pass
                                
                                else:
                                    gamma = np.arcsin(-np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                    beta = np.arcsin(np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                    alpha = 0
                                
                                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'PZYZ')
                                
                                if np.array_equal(np.round(molecular_tensor,3), np.round(symmetric_tensor,3)):
                                    pass
                                
                                else:
                                    gamma = np.arcsin(np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) / (eigenvals[0] - eigenvals[1])))
                                    beta = np.arcsin(-np.sqrt((symmetric_tensor[2, 2] - eigenvals[2]) / (eigenvals[0] - eigenvals[2] + (eigenvals[1] - eigenvals[0]) * (np.sin(gamma) ** 2))))
                                    alpha = 0
                                
                                molecular_tensor = NMRFunctions.backrotate_diagonal_tensor_with_euler(diagonal_mehring, alpha, beta, gamma, 'PZYZ')
                                
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


            # Get the angles in degrees and round up values
            alpha = np.degrees(alpha).round(2)
            beta = np.degrees(beta).round(2)
            gamma = np.degrees(gamma).round(2)

            print(f'Euler angles are: \n{alpha}, {beta}, {gamma}\n')
            print("# -------------------------------------------------- #")

            return alpha, beta, gamma
                
        
        # Define a rotation of a given tensor using Euler angles
        def backrotate_diagonal_tensor_with_euler(self, diagonal_pas, alpha, beta, gamma, rotation_mode):
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
                
                R_tot = R_z1 * R_y2 * R_z3

                # invert if passive
                if rotation_mode == 'PZYZ':
                    inv_R_z1 = np.linalg.inv(R_z1)
                    inv_R_y2 = np.linalg.inv(R_y2)
                    inv_R_z3 = np.linalg.inv(R_z3)

                    R_tot = inv_R_z1 *inv_R_y2 * inv_R_z3

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
                
                R_tot = R_z1 * R_x2 * R_z3

                # invert if passive
                if rotation_mode == 'PZXZ':
                    inv_R_z1 = np.linalg.inv(R_z1)
                    inv_R_x2 = np.linalg.inv(R_x2)
                    inv_R_z3 = np.linalg.inv(R_z3)
                    
                    R_tot = inv_R_z1 *inv_R_x2 * inv_R_z3
            
            else:
                print('The input rotation mode is wrong, please choose one of the following formats: AZYZ, PZYZ, AZXZ, PZXZ')

            backrotated_tensor = R_tot * diagonal_pas * np.linalg.inv(R_tot)

            return backrotated_tensor

        
        # Define radius of Ovaloid for parametric plots
        def radiusovaloid(self, sxx, syy, szz, alpha, beta, gamma, theta, phi):
            '''
            to check
            '''
            r_ov = (sxx * (np.sin(gamma) * np.sin(alpha - phi) * np.sin(theta) + np.cos(gamma) * (np.cos(theta) * np.sin(beta) - np.cos(beta) * np.cos(alpha - phi) * np.sin(theta))) ** 2
                + syy * (np.cos(theta) * np.sin(beta) * np.sin(gamma) - (np.cos(beta) * np.cos(alpha - phi) * np.sin(gamma) + np.cos(gamma) * np.sin(alpha - phi) * np.sin(theta))) ** 2
                + szz * (np.cos(beta) * np.cos(theta) + np.cos(alpha - phi) * np.sin(beta) * np.sin(theta)) ** 2)
            
            print(f'Radius of ovaloid is: {r_ov}')
            
            return r_ov
        
        # Generate sets of equivalent euler angles based on AZYZ, PZYZ, AZXZ, PZXZ conventions
        def EqEulerSet(alpha, beta, gamma):
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
            # Print header and version
            print("\n\n          #################################################")
            print("          | --------------------------------------------- |")
            print("          |         (NC)^2I.py: NMR Calculations          |")
            print("          |         for Noncovalent Interactions          |")
            print("          | --------------------------------------------- |")
            print("          |      CALLING EXTERNAL MODULE: OrcaAnalysis    |")
            print("          |                       -                       |")
            print("          |           ORCA ANALYSIS COLLECTIONS           |")
            print("          |                                               |")
            print("          |               Ettore Bartalucci               |")
            print("          |     Max Planck Institute CEC & RWTH Aachen    |")
            print("          |            Worringerweg 2, Germany            |")
            print("          |                                               |")
            print("          #################################################\n")
            pass
        
        # SECTION 1: READ ORCA OUTPUT FILES AND COUNT NUMBER OF JOBS
        def count_jobs_number(output_file):
            """
            Read the output (.mpi8.out) file from an ORCA calculation and count how many jobs have been run.
            :param output_file: output file from orca in the form .mpi8.out
            """
            with open(output_file, 'r') as f:
                # Count number of jobs in output file
                lines = f.readlines()
                count = 0
                for line in lines:
                    if line.strip().startswith("$$$$$$$$$$$$$$$$  JOB NUMBER"):
                        count += 1
                return count

        # SECTION 2: READ ORCA OUTPUT FILES AND EXTRACT LEVEL OF THEORY
        def extract_level_of_theory(output_file):
            """
            Read the output (.mpi8.out) file from an ORCA calculation and extract level of theory.
            :param output_file: output file from orca in the form .mpi8.out
            """
            try:
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                    
                    # Search for the line containing "# Level of theory"
                    for i, line in enumerate(lines):
                        if "# Level of theory" in line:
                            # Extract the line immediately after it, this wont work if ppl dont use my syntax
                            level_of_theory_line = lines[i + 1].strip()

                            # Remove the line number from the line - do i want to keep this?
                            level_of_theory = level_of_theory_line.replace("| 10>", "").strip()

                            return level_of_theory

                    return "Level of theory not found in the file."
            
            except FileNotFoundError:
                return f"File '{output_file}' not found."
            except Exception as e:
                return f"An error occurred: {str(e)}"

        # SECTION 3: READ ORCA OUTPUT FILES AND SPLIT FOR NUMBER OF JOBS
        def split_orca_output(output_file):
            """
            This function splits the huge ORCA multi-job file into individual job files.
            Then sews it back with the initial output lines from ORCA so that the files can
            be opened in Avogadro, otherwise it doesnt work.
            :param output_file: output file from orca in the form .mpi8.out
            """
            if not os.path.isfile(output_file):
                print(f"Error in SECTION 3: ORCA output file '{output_file}' not found, please define.")
                return

            # Make use of RegEx for matching JOB lines
            job_matching = re.compile(r'\$+\s*JOB\s+NUMBER\s+(\d+) \$+')

            # Extract initial ORCA text before job files to append to splitted files
            initial_content = extract_initial_output_content(output_file, job_matching)

            with open(output_file, 'r') as f:
                current_job_n = None # current job number
                current_job_content = [] # job specific info

                for line in f:
                    match = job_matching.search(line) # regex match search
                    if match:
                        # if match is found, write to file
                        if current_job_n is not None:
                            output_file_path = f'split_output/splitted_orca_job{current_job_n}.out'
                            with open(output_file_path, 'w') as out:
                                out.writelines(initial_content + current_job_content) # initial orca info + job specific info
                            
                            print(f'Wrote job {current_job_n} to {output_file_path}')

                        current_job_n = match.group(1)
                        current_job_content = [] # reset

                    current_job_content.append(line)

                # write last job to file
                if current_job_n is not None:
                    output_file_path = f'split_output/splitted_orca_job{current_job_n}.out'
                    with open(output_file_path, 'w') as out:
                        out.writelines(initial_content + current_job_content)
                    
                    print(f'Wrote job {current_job_n} to {output_file_path}')

            print(f'ORCA output has been split into {current_job_n} sub files for further analysis')

        # This adds the initial content necessary for avogadro visualization to each splitted file
        def extract_initial_output_content(output_file, job_matching):
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

        # SECTION 4: READ ORCA PROPERTY FILES AND EXTRACT SHIELDINGS
        def read_property_file(property_file, job_number):
            """
            Read the property file from an ORCA calculation. Extract CSA shieldings (shielding tensor (ppm))
            and shifts (P(iso)) for each nucleus.

            Input:
            property_file:
                is the file that comes as outcome from ORCA calculations containing the most important informations
                on the simulations. We want the NMR parameter, but in principle it contains a summary of the ORCA
                .mpi8.out file.
            job_number:
                number of ORCA jobs ran
            Output:
            shielding_{job_number}.txt:
                file with condensed NMR data ready for plotting.
                It contains csa_tensor_data for the shielding tensor and shifts_data for the isotropic shifts
            """

            # Dictionary to store NMR data for each nucleus
            csa_tensor_data = {}
            shifts_data = {}

            nucleus_info = None
            reading_shielding = False

            with open(property_file, 'r') as f:
                for line in f:
                    if "Nucleus:" in line:
                        nucleus_info = line.strip().split()
                        nucleus_index = int(nucleus_info[1])
                        nucleus_name = nucleus_info[2]
                        csa_tensor_data[(nucleus_index, nucleus_name)] = []
                        reading_shielding = True
                    elif reading_shielding and "Shielding tensor (ppm):" in line:
                        # Skip the header line
                        next(f)
                        shielding_values = []
                        for _ in range(3):
                            tensor_line = next(f).split()
                            shielding_values.append([float(val) for val in tensor_line])
                        csa_tensor_data[(nucleus_index, nucleus_name)] = shielding_values
                    elif "P(iso)" in line:
                        shifts = float(line.split()[-1])
                        shifts_data[(nucleus_index, nucleus_name)] = shifts
                        reading_shielding = False

            # Write the extracted data to shieldings.txt
            output_shielding_path = f'nmr_data/shieldings_{job_number}.txt'
            for job_n in range(1, job_number +1):
                with open(output_shielding_path, "w") as output_f:
                    for (nucleus_index, nucleus_name), shielding_values in csa_tensor_data.items():
                        output_f.write(f"Nucleus {nucleus_index} {nucleus_name}\nShielding tensor (ppm):\n")
                        for values in shielding_values:
                            values = values[1:]
                            output_f.write("\t".join(map(str, values)) + "\n")
                        shifts = shifts_data[(nucleus_index, nucleus_name)]
                        output_f.write(f"Isotropic Shift Nucleus {nucleus_index} {nucleus_name}: {shifts}\n")
                        output_f.write("\n")
                
                print(f"Shieldings extracted and saved to 'nmr_data/shieldings_job{job_number}.txt'.")

        # SECTION 5: READ ORCA OUTPUT FILE FOR EXTRACTING SCALAR COUPLINGS
        def read_couplings(output_file): # need run only if in input ssall
            """
            Read the output file from an ORCA calculation. Extract scalar couplings for each nucleus

            Input:
            output_file:
                is the file that comes as outcome from ORCA calculations containing the most important informations
                on the simulations. We want the NMR parameter, which are towards the end of the ORCA .mpi8.out file.
            Output:
            j_couplings.txt:
                file with condensed NMR data ready for plotting.
                It contains scalar J couplings in pseudo-table format
            """

            # Dictionary to store NMR data for each nucleus
            j_coupling_data = {}

            # List to store the order of nuclei
            nuclei_order = []

            reading_couplings = False

            # Define regular expressions for start and end markers
            start_marker = re.compile(r'^\s*SUMMARY\s+OF\s+ISOTROPIC\s+COUPLING\s+CONSTANTS\s+\(Hz\)')
            end_marker = re.compile(r'Maximum memory used throughout the entire EPRNMR-calculation:')

            with open(output_file, 'r') as f:
                for line in f:
                    line = line.strip()

                    # Check for start marker
                    if start_marker.search(line):
                        reading_couplings = True
                        print("Entering the reading couplings block.")
                        continue  # Start reading couplings
                    
                    # Check for end marker
                    if end_marker.search(line):
                        reading_couplings = False
                        print("Exiting the reading couplings block.")
                        break  # Stop reading couplings when this line is encountered

                    # If we're reading couplings, process the line
                    if reading_couplings and line:
                        data_values = re.split(r'\s+', line)
                        nucleus = data_values[0]
                        j_couplings = data_values[1:]
                        nuclei_order.append(nucleus)  # Add the nucleus to the order list
                        j_coupling_data[nucleus] = j_couplings
                        
            # Write the formatted J coupling data to j_couplings.txt
            with open('nmr_data/j_couplings.txt', 'w') as output_file:
                # Write the header row with nuclei information
                output_file.write('\t'.join(nuclei_order) + '\n')
                
                # Write the data rows
                for nucleus, j_couplings in j_coupling_data.items():
                    output_file.write(f"{nucleus}\t{' '.join(j_couplings)}\n")

            print("J couplings extracted and saved to 'nmr_data/j_couplings.txt'.")

        # SECTION 6: PLOTTING NMR DATA (I) SHIELDING TENSOR COMPONENTS
        def extract_shielding_tensor(shielding_tensor):
            """
            Load the previously extracted shielding values and plot them as a function of distance.
            Input:
            shielding_tensor: filex extracted from the property data containing shielding tensor and isotropic shielding
            Output:
            sigma_xx, sigma_yy, sigma_zz: shielding tensor components for plotting with type dict   
            """

            # Dictionaries to store tensor components for each nucleus type
            sigma_xx = {}  
            sigma_yy = {}
            sigma_zz = {}
            
            # Read shielding tensor from file - continue from here
            with open(shielding_tensor, 'r') as f:
                lines = f.readlines()

            current_nucleus = None

            current_shielding_xx = []
            current_shielding_yy = []
            current_shielding_zz = []

            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('Nucleus'):
                    if current_nucleus is not None:
                        sigma_xx[current_nucleus] = current_shielding_xx
                        sigma_yy[current_nucleus] = current_shielding_yy
                        sigma_zz[current_nucleus] = current_shielding_zz
                    nucleus_info = line.split()[1:]  # Extracting both number and type
                    current_nucleus = f"Nucleus {' '.join(nucleus_info)}"
                    current_shielding_xx = []
                    current_shielding_yy = []
                    current_shielding_zz = []

                elif line.startswith('Shielding tensor (ppm):'):
                    tensor_component_xx = float(lines[i + 1].split()[0])
                    tensor_component_yy = float(lines[i + 2].split()[1])
                    tensor_component_zz = float(lines[i + 3].split()[2])

                    current_shielding_xx.append(tensor_component_xx)
                    current_shielding_yy.append(tensor_component_yy)
                    current_shielding_zz.append(tensor_component_zz)

            if current_nucleus is not None:
                sigma_xx[current_nucleus] = current_shielding_xx
                sigma_yy[current_nucleus] = current_shielding_yy
                sigma_zz[current_nucleus] = current_shielding_zz 

            # Print the dictionary keys, which are then nuclear types
            #print(f'Nuclear data keys for CSA tensors are: {sigma_xx.keys()}')
            
            # Get number of nuclei per simulation
            n_nuclei = len(sigma_xx.keys())
            #print(f'Number of nuclei per simulation is: {n_nuclei}')

            # Extract all the nuclear identities in xyz file
            nuc_identity = sigma_xx.keys()
            
            return sigma_xx, sigma_yy, sigma_zz, nuc_identity

        # SECTION 7: PLOTTING NMR DATA (II) ISOTROPIC CHEMICAL SHIFT
        def extract_isotropic_shifts(shielding_tensor):
            """
            Load the previously extracted isotropic shift values and plot them as a function of distance
            """

            nucleus_data = {}  # Dictionary to store data for each nucleus type
            
            with open(shielding_tensor, 'r') as f:
                lines = f.readlines()

            current_nucleus = None
            current_data = []

            for line in lines:
                line = line.strip()
                if line.startswith('Nucleus'):
                    if current_nucleus is not None:
                        nucleus_data[current_nucleus] = current_data
                    nucleus_info = line.split()[1:]  # Extracting both number and type
                    current_nucleus = f"Nucleus {' '.join(nucleus_info)}"
                    current_data = []
                elif line.startswith('Isotropic Shift Nucleus'):
                    isotropic_shift = float(line.split()[-1])
                    current_data.append(isotropic_shift)

            if current_nucleus is not None:
                nucleus_data[current_nucleus] = current_data
            
            # Print the dictionary keys, which are then nuclear types
            #print(f'Nuclear data keys are: {nucleus_data.keys()}')
            
            # Get number of nuclei per simulation
            n_nuclei = len(nucleus_data.keys())
            #print(f'Number of nuclei per simulation is: {n_nuclei}')
            
            return nucleus_data

        # SECTION 8: NONCOVALENT INTERACTION DISTANCE CLASSIFIER
        def set_noncov_interactions():
            # Display possible options to the user
            # Adjust based on the number of noncovalent interactions you want
            print("NONCOV effective distances options:")
            print("1. Cation-pi")
            print("2. Anion-pi")
            print("3. pi-pi")
            print("4. H-bond")
            print("5. Polar-pi")
            print("6. n-pi*")

            # Get user input and validate
            while True:
                try:
                    user_input = int(input("Enter your choice of NONCOV type please (1-6): "))
                    if 1 <= user_input <= 6:
                        return user_input
                    else:
                        print("Invalid choice. Please enter a number between 1 and 6.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

        def set_boundary_distance_values(user_choice):
            # Set min and max effective distance values in Angstroem based on user's choice
            if user_choice == 1: # Cation-pi interaction from https://doi.org/10.1016%2Fj.jmb.2021.167035
                return 2, 6
            elif user_choice == 2: # Anion-pi interaction from https://doi.org/10.1039%2Fc5sc01386k
                return 2, 5
            elif user_choice == 3: # pi-pi interaction
                return 1, 5
            elif user_choice == 4: # H-bond interaction from https://doi.org/10.1016/B978-012486052-0/50005-1
                return 2.7, 3.3
            elif user_choice == 5: # Polar-pi interaction
                return 1, 5
            elif user_choice == 6: # n-pi* interaction
                return 1, 5

        # ----------------------------------------------------------------#


        # -------------------------- start --------------------------------#
        # ----------------------------------------------------------------#
        # SECTION 9: EXTRACT INITIAL DISTANCE BETWEEN NUCLEAR PAIRS FOR DISTANCE PLOTS
        # compute the displacement as a difference between the coordinates of the first point and the
        # coordinates of the second point.


        # SECTION 10: PLOTTING NMR DATA (III) IN PAS: DIAMAGNETIC, PARAMAGNETIC, TOTAL CSA TENSOR
        def extract_csa_tensor_in_pas(splitted_output_file):
            """
            Load the splitted orca output files and read diagonal components of diamagnetic, paramagnetic and total CSA tensor
            Input:
            splitted_output_file: orca output file splitted by number of jobs
            Output:
            :sigma_dia: diagonal diamagnetic shielding tensor components in the PAS  
            :sigma_para: diagonal paramagnetic shielding tensor components in the PAS  
            :sigma_tot: diagonal total shielding tensor components in the PAS - used in the tensor ellipsoid
            Convention:
            sigma_11 < sigma_22 < sigma_33 in [ppm]  
            Diagonalization:
            given by orca automatically as sT*s
            """

            # Dictionaries to store diamagnetic tensor components for each nucleus type
            sigma_dia = {}  

            # Dictionaries to store paramagnetic tensor components for each nucleus type
            sigma_para = {}  

            # Dictionaries to store total tensor components for each nucleus type
            sigma_tot = {}  

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
                    if 'CHEMICAL SHIFTS' in line:
                        start_search = True
                        continue

                    if start_search:
                        line = line.strip()
                        if line.startswith('Nucleus'):
                            if current_nucleus is not None:
                                sigma_dia[current_nucleus] = current_dia_shielding
                                sigma_para[current_nucleus] = current_para_shielding
                                sigma_tot[current_nucleus] = current_tot_shielding
                            # add the nucleus information to file
                            nucleus_info = line.split()[1:]
                            current_nucleus = f"Nucleus {' '.join(nucleus_info)}"
                            current_dia_shielding = []
                            current_para_shielding = []
                            current_tot_shielding = []

                        # Extract the various tensor components here
                        elif line.startswith('sDSO'):
                            try:
                                dia_tensor_components = [float(x) for x in line.split()[1:4]]
                                current_dia_shielding.append(dia_tensor_components)
                            except (ValueError, IndexError):
                                #print('Error encountered when extracting sDSO diamagnetic tensor components')
                                continue

                        elif line.startswith('sPSO'):
                            try:
                                para_tensor_components = [float(x) for x in line.split()[1:4]]
                                current_para_shielding.append(para_tensor_components)
                            except (ValueError, IndexError):
                                #print('Error encountered when extracting sPSO paramagnetic tensor components')
                                continue

                        elif line.startswith('Total'):
                            try:
                                tot_tensor_components = [float(x) for x in line.split()[1:4]]
                                current_tot_shielding.append(tot_tensor_components)
                            except (ValueError, IndexError):
                                #print('Error encountered when extracting total shielding tensor components')
                                continue
                    
                    # stop extraction at the end of the tensor nmr block of the output
                    if 'CHEMICAL SHIELDING SUMMARY (ppm)' in line:
                        break

                # Store last nucleus data
                if current_nucleus is not None:
                    sigma_dia[current_nucleus] = current_dia_shielding
                    sigma_para[current_nucleus] = current_para_shielding
                    sigma_tot[current_nucleus] = current_tot_shielding 
                
            except FileNotFoundError:
                print(f"File '{splitted_output_file}' not found.")
                return {}, {}, {}, []

            # Extract all the nuclear identities in xyz file
            nuc_identity = list(sigma_tot.keys())

            return sigma_dia, sigma_para, sigma_tot, nuc_identity

        # SECTION 11: PLOT MOLECULAR FRAME
        def plot_3d_molecule(molecule_path, sizes=None):
            """
            Load and plot the molecular coordinates in 3D and display them according to standard CPK convention.
            Basically a very time consuming way to do what Avogadro does but worse.. Need fixing 
            Input:
            molecule_path: path to the xyz file to load and display
            sizes: <optional> define atomic radii for pure graphical display, if not defined when function is called, they are taken from
            default (atomic radii [pm] * 5)
            """
            # --------------------------------------------------------------------------------------------#
            # 3D Molecular representation plot
            # Define CPK coloring representation for atom types: https://en.wikipedia.org/wiki/CPK_coloring
            cpk_colors = {
                'H': 'grey',
                'C': 'black',
                'O': 'red',
                'N': 'blue',
                'S': 'yellow',
                'P': 'purple',
                'F': 'cyan',
                'Cl': 'green',
                'Br': 'darkred',
                'I': 'pink',
                'Co': 'silver',
                'Fe': 'silver',
                'Ni': 'silver',
                'Cu': 'silver'
            }

            # Default empirical sizes (picometers*5) for atomic radii: https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
            if sizes is None:
                sizes = {
                    'H': 25*5,
                    'C': 70*5,
                    'O': 60*5,
                    'N': 65*5,
                    'S': 100*5,
                    'P': 100*5,
                    'F': 50*5,
                    'Cl': 100*5,
                    'Br': 115*5,
                    'I': 140*5,
                    'Co': 135*5,
                    'Fe': 140*5,
                    'Ni': 135*5,
                    'Cu': 135*5
                }

            # Read XYZ molecular coordinates from the file
            with open(molecule_path, 'r') as file:
                lines = file.readlines()

            num_atoms = int(lines[0])
            coordinates = []
            atom_types = []

            for line in lines[2:]:
                parts = line.split()
                atom_type = parts[0]
                x, y, z = map(float, parts[1:])
                coordinates.append([x, y, z])
                atom_types.append(atom_type)

            coordinates = np.array(coordinates)

            # Plot atoms in 3D space with CPK colors and atomic radial sizes as defined earlier
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for i in range(num_atoms):
                ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], c=cpk_colors[atom_types[i]], s=sizes[atom_types[i]])

            # Connect atoms with bonds if the distance between atom pairs is less than 1.7A - need to adjust a bit especially for short distances
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    if np.linalg.norm(coordinates[i] - coordinates[j]) < 1.7:  # Bond length threshold
                        ax.plot([coordinates[i, 0], coordinates[j, 0]], [coordinates[i, 1], coordinates[j, 1]], [coordinates[i, 2], coordinates[j, 2]], color='grey')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.grid(False)  # Remove the grid background

            plt.show()


        # SECTION 12: PLOT TENSORS ELLIPSOIDS ON MOLECULAR FRAME



    # ------------------------------------------------------------------------------
    # NMR FUNCTIONS AND APPLICATIONS 
    # ------------------------------------------------------------------------------
    class DistanceScanner:
        """
        Take an input structure with two fragments and displace along centroid vector
        """
        
    # ------------------------------------------------------------------------------
    # GENERATE A DATASET FOR MACHINE LEARNING APPLICATIONS
    # ------------------------------------------------------------------------------ 
    class GenerateMLDataset:
        
        def __init__(self, root_directory, output_csv_path):

            # Print header and version
            print("\n\n          #################################################")
            print("          | --------------------------------------------- |")
            print("          |         (NC)^2I.py: NMR Calculations          |")
            print("          |         for Noncovalent Interactions          |")
            print("          | --------------------------------------------- |")
            print("          |        CALLING CLASS: GenerateMLDataset       |")
            print("          |                       -                       |")
            print("          |              MAKE DATASET TABLES              |")
            print("          |                                               |")
            print("          |               Ettore Bartalucci               |")
            print("          |     Max Planck Institute CEC & RWTH Aachen    |")
            print("          |            Worringerweg 2, Germany            |")
            print("          |                                               |")
            print("          #################################################\n")
                        
            self.root_directory = root_directory
        
            self.output_csv_path = output_csv_path

            # Headers of features, total features = 19
            self.columns = ['molecule', # Categorical
                            'atom', # Categorical
                            'noncov', # Categorical
                            'x_coord', # Integer
                            'y_coord', # Integer
                            'z_coord', # Integer
                            'tot_shielding_11', # Integer
                            'tot_shielding_22', # Integer
                            'tot_shielding_33', # Integer
                            'dia_shielding_11', # Integer
                            'dia_shielding_22', # Integer
                            'dia_shielding_33', # Integer
                            'para_shielding_11', # Integer
                            'para_shielding_22', # Integer
                            'para_shielding_33', # Integer
                            'iso_shift', # Integer 
                            'nmr_functional', # Categorical
                            'nmr_basis_set', # Categorical
                            'aromatic' # Binary             
                            ]
            
            # Create the dataframe
            self.df = pd.DataFrame(columns=self.columns)

            # Keep the user happy by saying that something happened
            print(f'The empty dataset has been created and saved in: {self.output_csv_path}')
            print('\n')

        # Extract features from splitted orca output files
        def extract_data_for_ml_database(self, file_path):
            """
            Read the splitted output file from an ORCA calculation and features for machine learning.
            Input
            :param file_path: output file from orca in the form .out

            Output: data dictionary containing
            :molecule
            :atom
            :noncov
            :x_coord
            :y_coord
            :z_coord
            :tot_shielding
            :dia_shielding
            :para_shielding
            :iso_shift
            :nmr_functional
            :nmr_basis_set
            :aromatic
            """
            # Define empty feature variables to extract and append to dataset
            # All shielding tensors are diagonalized in PAS and ordered in the Mehring way before being appended to database
            data = {
                'molecule': [],
                'atom': [],
                'noncov': [],
                'x_coord': [],
                'y_coord': [],
                'z_coord': [],
                'tot_shielding_11': [],
                'tot_shielding_22': [],
                'tot_shielding_33': [],
                'dia_shielding_11': [],
                'dia_shielding_22': [],
                'dia_shielding_33': [],
                'para_shielding_11': [],
                'para_shielding_22': [],
                'para_shielding_33': [],
                'iso_shift': [],
                'nmr_functional': [],
                'nmr_basis_set': [],
                'aromatic': []
            }


            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                    # Search markers
                    coordinates_found = False
                    shielding_found = False
                    molecule_found = False
                    noncov_found = False
                    aromatic_found = False

                    current_dia_shielding = []
                    current_para_shielding = []
                    current_tot_shielding = []

                    for line in lines:
                        # @UserStaticInputs
                        # Get molecular information from the user-defined Molecule flag in the input file
                        if '# Molecule:' in line:
                            molecule_found = True
                            data['molecule'].append(line.split(':')[-1].strip())
                        
                        # Get Noncov type information from the user-defined Noncov flag in the input file
                        elif '# Noncov:' in line:
                            noncov_found = True
                            data['noncov'].append(line.split(':')[-1].strip())
                        
                        # Get aromatic information from the user-defined Aromatic flag in the input file
                        elif '# Aromatic:' in line:
                            aromatic_found = True
                            data['aromatic'].append(line.split(':')[-1].strip()) # as binary 1 or 0

                        # To check, maybe get these info from shielding files?
                        # Get atom and relative coordinates info from file
                        elif 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                            coordinates_found = True
                        
                        # add a linethat add numbers to nuclei to match
                        elif coordinates_found and line.strip():
                            atomic_info = line.split()
                            data['atom'].append(atomic_info[0])
                            data['x_coord'].append(float(atomic_info[1]))
                            data['y_coord'].append(float(atomic_info[2]))
                            data['z_coord'].append(float(atomic_info[3]))
                        
                        # Get the total, diamagnetic and paramagnetic tensors and diagonalize them
                        elif 'CHEMICAL SHIFTS' in line:
                            shielding_found = True
                            continue
                        
                        if shielding_found:
                            line = line.strip()

                            if line.startswith('Diamagnetic contribution to the shielding tensor (ppm) :'):
                                try:
                                    dia_tensor_components = [float(x) for x in line.split()[1:4]]
                                    current_dia_shielding.append(dia_tensor_components)
                                except (ValueError, IndexError):
                                    continue

                            elif line.startswith('Paramagnetic contribution to the shielding tensor (ppm):'):
                                try:
                                    para_tensor_components = [float(x) for x in line.split()[1:4]]
                                    current_para_shielding.append(para_tensor_components)
                                except (ValueError, IndexError):
                                    continue

                            elif line.startswith('Total shielding tensor (ppm):'):
                                try:
                                    tot_tensor_components = [float(x) for x in line.split()[1:4]]
                                    current_tot_shielding.append(tot_tensor_components)
                                except (ValueError, IndexError):
                                    continue

                    # Diagonalize the tensors
                    if current_dia_shielding:
                        data['dia_shielding_11'], data['dia_shielding_22'], data['dia_shielding_33'] = NMRFunctions.diagonalize_tensor(
                            *current_dia_shielding)
                    if current_para_shielding:
                        data['para_shielding_11'], data['para_shielding_22'], data['para_shielding_33'] = NMRFunctions.diagonalize_tensor(
                            *current_para_shielding)
                    if current_tot_shielding:
                        data['tot_shielding_11'], data['tot_shielding_22'], data['tot_shielding_33'] = NMRFunctions.diagonalize_tensor(
                            *current_tot_shielding)
                                
                
                    # Get functional and basis set for NMR calculations from the line containing the Level of theory flag
                    for i, line in enumerate(lines):
                        if "# Level of theory" in line:
                            # Extract the line immediately after it, this wont work if ppl dont use my syntax
                            level_of_theory_line = lines[i + 1].strip()

                            # Remove the line number from the line - do i want to keep this?
                            level_of_theory = level_of_theory_line.replace("| 10> !", "").strip()
                            level_of_theory = level_of_theory.split()

                            # Extract functional and basis set
                            data['nmr_functional'] = level_of_theory[0]
                            data['nmr_basis_set'] = level_of_theory[1]
                    
                    if not molecule_found and not noncov_found and not aromatic_found:
                        data['molecule'] = [0]
                        data['noncov'] = [0]  
                        data['aromatic'] = [0] 
                        
            except FileNotFoundError:
                return f"File '{file_path}' not found."
            except Exception as e:
                return f"An error occurred: {str(e)}"
            
            return data


        # Search for all the splitted output files from an ORCA calculation in the Machine learning project root directory
        def search_files(self):
            # Iterate through all directories and subdirectories in root for the orca output files
            for root, dirs, files in os.walk(self.root_directory):
                for file in files:
                    if file.startswith('splitted_') and file.endswith('.out'): # working with output files is much easier than with full mpi8.out
                        # get the path to those files
                        file_path = os.path.join(root, file)

                        instance_data = self.extract_data_for_ml_database(file_path)
                        
                        # Check if the instance data is not empty (indicating an error)
                        #if instance_data:
                            # Construct DataFrame from the instance data dictionary
                            #instance_df = pd.DataFrame(instance_data)
                            
                            # Append the DataFrame to the main DataFrame
                            #self.df = self.df.append(instance_df, ignore_index=True)
                            
                            # Write to CSV file
                            #self.df.to_csv(self.output_csv_path, index=False)
                        #else:
                            #print(f"No data extracted from file: {file_path}")

            if self.df.empty:
                print("No raw data has been found matching the specified criteria.")

