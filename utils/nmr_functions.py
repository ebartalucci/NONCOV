###################################################
# COLLECTION OF FUNCTIONS FOR NMR TRANSFORMATIONS #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 16.02.2024                 #
#               Last:  26.02.2024                 #
#               -----------------                 #
#             Stable release version              #
#                   v.0.0.2                       #
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
        print("          |                WORKFLOW STEP x                |")
        print("          |                       -                       |")
        print("          |           NMR FUNCTIONS COLLECTIONS           |")
        print("          |                                               |")
        print("          |               Ettore Bartalucci               |")
        print("          |     Max Planck Institute CEC & RWTH Aachen    |")
        print("          |            Worringerweg 2, Germany            |")
        print("          |                                               |")
        print("          #################################################\n")
        pass

    @staticmethod
    def diagonalize_tensor(sxx, sxy, sxz, syx, syy, syz, szx, szy, szz):
        """
        Take NMR tensor as input and perform various operations, including diagonalization and ordering in Mehring and Haberlen formalisms.
        Input
        :param sxx, sxy, sxz, syx, syy, syz, szx, szy, szz: individual tensor components for a 3x3 chemical shielding matrix
        Output
        :param diagonal_mehring: full diagonalized matrix in principal axis system according to sigma_11 < sigma_22 < sigma_33
        :param sigma_11: individual component
        :param sigma_22: individual component
        :param sigma_33: individual component
        """

        # Notify user which module has been called
        print("# -------------------------------------------------- #")
        print("# MATRIX DIAGONALIZATION FUNCTION HAS BEEN REQUESTED #")
        print(f'\n')
                
        # Initialize shielding tensor matrix
        matrix = np.matrix([[sxx, sxy, sxz],
                            [syx, syy, syz],
                            [szx, szy, szz]]
                            )
        
        print(f'Shielding Tensor is: \n{matrix}')
        print('Proceeding to transposing...\n')

        # Transpose matrix
        transposed = np.transpose(matrix)
        print(f'Transposed matrix is: \n{transposed}')
        print('Proceeding to diagonalization...\n')

        # Calculate eigenvalues and vectors, take absolute value of eigenvalues 
        eigenvals, eigenvecs = np.linalg.eig(matrix)
        eigenvals = eigenvals.round(5) # round them up
        eigenvecs = eigenvecs.round(5) # round them up
        print(f'Eigenvalues are: {eigenvals}, Eigenvectors are: \n{eigenvecs}')
        print('Proceeding to ordering eigenvalues and eigenvectors...\n')

        # Sort eigenvalues and eigenvectors based on magnitude of eigenvalues
        idx = np.argsort(np.abs(eigenvals))
        eigenvals_ordered = eigenvals[idx]
        eigenvecs_ordered = eigenvecs[:, idx]
        print(f'Ordered eigenvalues are: {eigenvals_ordered}, ordered eigenvectors are: \n{eigenvecs_ordered}.')
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
        diagonal_haberlen = np.argsort(np.abs(np.diag(diagonal) - s_iso))
        diagonal_haberlen = np.diag(np.diag(diagonal)[diagonal_haberlen])
        print(f'Diagonal tensor in Habelen order is: \n{diagonal_haberlen}')

        # Reorder matrix according to Mehring convention
        diagonal_mehring = sorted(np.diag(diagonal))
        sigma_11 = diagonal_mehring[0]
        sigma_22 = diagonal_mehring[1]
        sigma_33 = diagonal_mehring[2]
        diagonal_mehring = np.diag(diagonal_mehring)
        print(f'Diagonal tensor in Mehring order is: \n{diagonal_mehring}\n')
        print(f'''where:\n \u03C3_11:{sigma_11} \n \u03C3_22:{sigma_22} \n \u03C3_33:{sigma_33}''')

        print("# -------------------------------------------------- #")

        return sigma_11, sigma_22, sigma_33
    
    @staticmethod
    def active_rotations():
        
        rot_mat_A = np.matrix([[np.cos(a)*np.cos(b)*np.cos(g), sxy, sxz],
                            [syx, syy, syz],
                            [szx, szy, szz]]
                            )
        
        rot_mat_A_inverted = np.matrix([[sxx, sxy, sxz],
                            [syx, syy, syz],
                            [szx, szy, szz]]
                            )


    @staticmethod
    def ovaloid_tensorplot(sigma_11, sigma_22, sigma_33, x, y, z):

