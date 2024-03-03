###################################################
# COLLECTION OF FUNCTIONS FOR NMR TRANSFORMATIONS #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 16.02.2024                 #
#               Last:  03.03.2024                 #
#               -----------------                 #
#             Stable release version              #
#                   v.0.1.0                       #
#                                                 #
###################################################

import numpy as np
from scipy.optimize import minimize

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
        shielding_tensor = np.matrix([[sxx, sxy, sxz],
                                    [syx, syy, syz],
                                    [szx, szy, szz]]
                                    )
        
        print(f'Shielding Tensor is: \n{shielding_tensor}')
        print('Proceeding to transposing...\n')

        # Transpose matrix
        transposed = np.transpose(shielding_tensor)
        print(f'Transposed matrix is: \n{transposed}')
        print('Proceeding to diagonalization...\n')

        # Calculate eigenvalues and vectors, take absolute value of eigenvalues 
        eigenvals, eigenvecs = np.linalg.eig(shielding_tensor)
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
        sigma_XX = diagonal_haberlen[0]
        sigma_YY = diagonal_haberlen[1]
        sigma_ZZ = diagonal_haberlen[2]
        diagonal_haberlen = np.diag(np.diag(diagonal)[diagonal_haberlen])
        print(f'Diagonal tensor in Haberlen order is: \n{diagonal_haberlen}\n')
        print(f'''where:\n \u03C3_XX:{sigma_XX} \n \u03C3_YY:{sigma_YY} \n \u03C3_ZZ:{sigma_ZZ}''')

        # Reorder matrix according to Mehring convention
        diagonal_mehring = sorted(np.diag(diagonal))
        sigma_11 = diagonal_mehring[0]
        sigma_22 = diagonal_mehring[1]
        sigma_33 = diagonal_mehring[2]
        diagonal_mehring = np.diag(diagonal_mehring)
        print(f'Diagonal tensor in Mehring order is: \n{diagonal_mehring}\n')
        print(f'''where:\n \u03C3_11:{sigma_11} \n \u03C3_22:{sigma_22} \n \u03C3_33:{sigma_33}''')

        print("# -------------------------------------------------- #")

        return shielding_tensor, diagonal_mehring, sigma_11, sigma_22, sigma_33
    
    # Right handed active rotation matrices for tensors alignment
    @staticmethod
    def active_rh_rotation(diagonal_mehring, alpha, beta, gamma):
        '''
        Perform a right handed active rotation using Euler angles as input values
        Input:
        Diagonal matrix elements in PAS
        Euler angles a, b, g
        Output:
        rotated_sigma_11, rotated_sigma_22, rotated_sigma_33
        '''
        # Notify user which module has been called
        print("# --------------------------------------------------- #")
        print("# THE ACTIVE RIGHT HAND ROTATIONS HAVE BEEN REQUESTED #")
        print(f'\n')

        # Rotation matrices
        def R_x(angle):
            R_x =  np.array([
                            [1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]]
                            )
            return R_x
    
        def R_y(angle):
            R_y =  np.array([
                            [np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)]]
                            )
            return R_y
    
        def R_z(angle):
            R_z =  np.array([
                            [np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]]
                            )
            return R_z
        
        # Rotate shielding tensor from initial PAS
        rotated_diagonal_mehring = np.linalg.multi_dot([R_z(alpha), R_y(beta), R_z(gamma), 
                                                        diagonal_mehring, 
                                                        R_z(-gamma), R_y(-beta), R_z(-alpha)]
                                                        )
        
        print(f'Perform active right hand rotation of the shielding tensor: \n{diagonal_mehring}\n')
        print(f'''as:\n {R_z} {R_y} {R_z} {diagonal_mehring} -{R_z} -{R_y} -{R_z}''')
        
        # Get individual components
        rotated_sigma_11 = rotated_diagonal_mehring[0]
        rotated_sigma_22 = rotated_diagonal_mehring[1]
        rotated_sigma_33 = rotated_diagonal_mehring[2]
        print(f'Rotated diagonal tensor in Mehring order is: \n{rotated_diagonal_mehring}\n')
        print(f'''where:\n rot_\u03C3_11:{rotated_sigma_11} \n rot_\u03C3_22:{rotated_sigma_22} \n rot_\u03C3_33:{rotated_sigma_33}''')

        print("# --------------------------------------------------- #")

        return rotated_diagonal_mehring, rotated_sigma_11, rotated_sigma_22, rotated_sigma_33
    
    # Define minimization method to find Euler angles as chisquare < 10e-10
    @staticmethod
    def dev_matrices(mat1, mat2):
        '''
        Calculate the chi-squared matrix between two matrices.
        Input:
        mat1: numpy array representing the expected matrix (original shielding tensor)
        mat2: numpy array representing the observed matrix (PAS shielding tensor)
        Output:
        deviation between the matrices
        '''
        # Ensure matrices have the same shape
        if mat1.shape != mat2.shape:
            raise ValueError("Matrices must have the same shape.")
                   
        # Compute chi-square statistic
        dev_matrices = np.sum((mat2 - mat1)**2)
        print(f'Deviation between matirces is: {dev_matrices}')
        
        return dev_matrices
    
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
    
    # Generate sets of equivalent euler angles
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
       
    # Minimization through chisquare (<10e-10)
    @staticmethod
    def minimize_chisq(dev_matrices, active_rh_rotation, shielding_tensor):
        """
        Minimize the chi-squared function with Powell's method.
        Input:
        active_rh_rotation: function returning the rotated shielding tensor
        shielding_tensor: the original initial shielding tensor
        dev_matrices: function to compute the deviation between two matrices
        alpha_start, alpha_end: bounds for Euler angle alpha
        beta_start, beta_end: bounds for Euler angle beta
        gamma_start, gamma_end: bounds for Euler angle gamma
        Output:
        chisq: the minimized chi-squared value
        alpha, beta, gamma: the optimized values of parameters alpha, beta, and gamma
        """

        # Boundaries for Euler angles to avoid redundant orientations. 
        # Gives postive projections of the PAS Y and Z axes on the molecular frame z axis
        alpha_start = 0
        alpha_end = 2 * np.pi
        beta_start = 0
        beta_end = np.pi / 2
        gamma_start = 0
        gamma_end = np.pi

        # Define the objective function for minimization
        def objective_function(angles):
            # Rotate the shielding tensor using the given angles
            rotated_diagonal_mehring, _, _, _ = active_rh_rotation(diagonal_mehring, *angles)
            # Compute the deviation between the rotated tensor and the original tensor
            deviation = dev_matrices(rotated_diagonal_mehring, shielding_tensor)
            return deviation

        # Minimize the objective function using Powell's method
        result = minimize(objective_function, x0=[alpha_start, beta_start, gamma_start], bounds=[(alpha_start, alpha_end), (beta_start, beta_end), (gamma_start, gamma_end)], method="Powell")
        chisq = result.fun
        alpha, beta, gamma = result.x
        return chisq, alpha, beta, gamma
        


# Test zone
xx =-5.9766
xy =-65.5206
xz =-9.5073
yx =-60.3020
yy =-23.0881
yz =-28.2399
zx =-10.8928
zy =-25.2372
zz =56.277

shielding_tensor, diagonal_mehring, sigma_11, sigma_22, sigma_33 = NMRFunctions.diagonalize_tensor(xx, xy, xz, yx, yy, yz, zx, zy, zz)
dev = NMRFunctions.dev_matrices(diagonal_mehring, shielding_tensor)

chisq, alpha, beta, gamma = NMRFunctions.minimize_chisq(NMRFunctions.dev_matrices, NMRFunctions.active_rh_rotation, shielding_tensor)
print(chisq)
print(alpha)



