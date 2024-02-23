###################################################
# COLLECTION OF FUNCTIONS FOR NMR TRANSFORMATIONS #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 16.02.2024                 #
#               Last:  16.02.2024                 #
#               -----------------                 #
#             Stable release version              #
#                   v.0.0.1                       #
#                                                 #
###################################################

import numpy as np

class NMRFunc:
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
    def diagonalize_matrix(sxx, sxy, sxz, syx, syy, syz, szx, szy, szz):
        
        # Initialize shielding tensor matrix
        matrix = np.matrix([[sxx, sxy, sxz],
                            [syx, syy, syz],
                            [szx, szy, szz]]
                            )
        
        print(f'Shielding Tensor is: \n{matrix}')
        print('Proceeding to transposing..\n')

        # Transpose matrix
        transposed = np.transpose(matrix)
        print(f'Transposed matrix is: \n{transposed}')
        print('Proceeding to symmetrization..\n')

        # Symmetrize the matrix
        symmetrized = np.array((matrix+transposed)/2)
        print(f'Symmetrized matrix is: \n{symmetrized}')
        print('Proceeding to diagonalization..\n')

        # Calculate eigenvalues and vectors, take absolute value of eigenvalues 
        eigenvals, eigenvecs = np.linalg.eig(symmetrized)
        eigenvals = eigenvals.round(5) # round them up
        eigenvecs = eigenvecs.round(5) # round them up
        print(f'Eigenvalues are: {eigenvals}, Eigenvectors are: \n{eigenvecs}')
        print('Proceeding to ordering eigenvalues and eigenvectors...\n')

        # Sort eigenvalues and eigenvectors based on magnitude of eigenvalues
        idx = np.argsort(np.abs(eigenvals))
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        print(f'Ordered eigenvalues are: {eigenvals}, ordered eigenvectors are: \n{eigenvecs}.')
        print('Proceeding to diagonalization...\n')

        #Compute diagonal matrix, define eigenvector columns as variables and preforme matrix multiplication 
        diagonal = np.diag(eigenvals)
        print(f'Diagonalized matrix is: \n{diagonal}')
        print('Proceeding to compute isotropic shift...\n')

        # Compute isotropic shift
        s_iso = np.sum(np.diag(diagonal)) / 3
        print(f'Isotropic shift is: {s_iso}')
        print('Proceeding to Haberlen ordering...\n')

        # Reorder matrix according to Haberlen convention
        diagonal_haberlen = np.argsort(np.abs(np.diag(diagonal) - s_iso))
        diagonal_haberlen = np.diag(np.diag(diagonal)[diagonal_haberlen])
        print(f'Diagonal matrix in Habelen order is: \n{diagonal_haberlen}')

        # Reorder matrix according to Mehring convention



        return diagonal, diagonal_haberlen






# Example usage:
if __name__ == "__main__":
    nmr_utils = NMRFunc()

    a = 1
    b =2
    c = 3
    d = 4
    e = 5
    f = 6
    g = 7
    h = 8
    i = 9
    sym, diag = nmr_utils.diagonalize_matrix(a,b,c,d,e,f,g,h,i)

