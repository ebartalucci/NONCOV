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
        
        print(f'Shielding Tensor is: {matrix}')
        print('Proceeding to diagonalization..')

        # Transpose matrix
        transposed = np.transpose(matrix)
        print(f'Transposed matrix is: {transposed}')
        print('Proceeding to symmetrization..')

        # Symmetrize the matrix
        symmetrized = np.array((matrix+transposed)/2)
        print(f'Symmetrized matrix is: {symmetrized}')
        print('Proceeding to diagonalization..')

        # Calculate eigenvalues and vectors, take absolute value of eigenvalues 
        eigenvals, eigenvecs = np.linalg.eig(symmetrized)
        absolute_eigenvals = np.absolute(eigenvals)
        absolute_eigenvals_S = np.power(absolute_eigenvals,2)
        print(f'Eigenvalues are: {eigenvals}, Eigenvectors are: {eigenvecs}')
        print('Proceeding to extract diagonal components, writing to diagonalized matrix..')

        #Compute diagonal matrix, define eigenvector columns as variables and preforme matrix multiplication 
        diagonal = (np.diag(absolute_eigenvals_S)).round(5)
        a = eigenvecs[:, 0].round(5)
        b = eigenvecs[:, 1].round(5)
        c = eigenvecs[:, 2].round(5)
        m_vecs = np.vstack((a,b,c)).round(5)
        m_vecsT=np.transpose(m_vecs).round(5)
        diagonalized_matrix = m_vecsT.dot(diagonal).dot(m_vecs).round(5)


        return symmetrized, diagonalized_matrix






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
    print(diag)

