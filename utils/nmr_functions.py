###################################################
#   COLLECTION OF FUNCTIONS FOR NMR APPLICATIONS  #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 16.02.2024                 #
#               Last:  29.04.2024                 #
#               -----------------                 #
#             Stable release version              #
#                   v.0.1.1                       #
#                                                 #
###################################################

import numpy as np

class NMRFunctions:
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
    @staticmethod
    def diagonalize_tensor(sxx, sxy, sxz, syx, syy, syz, szx, szy, szz):
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
    def tensor_to_euler(symmetric_tensor, eigenvals, eigenvecs, symmetry, rotation_mode): 
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
                            beta = np.arcsin(np.sqrt((symmetric_tensor[1, 1] - eigenvals[1]) - (eigenvals[0] - eigenvals[1]) * (np.sin(gamma) ** 2)) / ((eigenvals[2] - eigenvals[1]) *)))
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
    @staticmethod
    def radiusovaloid(sxx, syy, szz, alpha, beta, gamma, theta, phi):
        '''
        to check
        '''
        r_ov = (sxx * (np.sin(gamma) * np.sin(alpha - phi) * np.sin(theta) + np.cos(gamma) * (np.cos(theta) * np.sin(beta) - np.cos(beta) * np.cos(alpha - phi) * np.sin(theta))) ** 2
              + syy * (np.cos(theta) * np.sin(beta) * np.sin(gamma) - (np.cos(beta) * np.cos(alpha - phi) * np.sin(gamma) + np.cos(gamma) * np.sin(alpha - phi) * np.sin(theta))) ** 2
              + szz * (np.cos(beta) * np.cos(theta) + np.cos(alpha - phi) * np.sin(beta) * np.sin(theta)) ** 2)
        
        print(f'Radius of ovaloid is: {r_ov}')
        
        return r_ov
    
    # Generate sets of equivalent euler angles based on AZYZ, PZYZ, AZXZ, PZXZ conventions
    @staticmethod
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