###################################################
#       SOURCE CODE FOR THE NONCOV PROJECT        #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 26.02.2024                 #
#               Last:  12.03.2021                 #
#               -----------------                 #
#             Stable release version              #
#                   v.0.1.0                       #
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
            version = '0.1.0'
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
    def diagonalize_tensor(self, shielding_tensor):
        """
        Take NMR shielding tensor as input and perform various operations, 
        including diagonalization and ordering according to various formalisms.
        
        Input
        :param shielding_tensor: tensor components in 3x3 chemical shielding matrix
        
        Output 
        :param shielding_tensor: original shielding tensor in molecular frame
        :param s_iso: isotropic chemical shift from symmetrized tensor
        :param diagonal_mehring: PAS components with magnitude ordering (IUPAC)
        :param diagonal_haberlen: PAS sorted according to haberlen convention
        :param anisotropy: largest separation from center of gravity (iso shift)
        :param asymmetry: deviation from axially symmetric tensor
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
        print('Proceeding to compute isotropic shielding...\n')

        # Compute isotropic shift
        s_iso = np.sum(np.diag(diagonal)) / 3
        s_iso = s_iso.round(2)
        print(f'Isotropic shielding is: {s_iso} ppm')
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
        
        # Span and Skew calculations
        span = sigma_33 - sigma_11
        span = span.round(2)
        skew = 3*(sigma_22-s_iso)/span
        skew = skew.round(2)
        print(f'Span is: {span}\n')
        print(f'Skew is: {skew}\n')

        # Haberlen convention 
        diagonal_haberlen = [eigenvals[0]-s_iso, eigenvals[1]-s_iso, eigenvals[2]-s_iso]        
        red_anisotropy = diagonal_haberlen[0]
        asymmetry = (diagonal_haberlen[2]-diagonal_haberlen[1]) / diagonal_haberlen[0]
        anisotropy = (3*diagonal_haberlen[0])/2
        print(f'Haberlen is: {diagonal_haberlen}\n')
        print(f'Reduced anisotropy is: {red_anisotropy}\n')
        print(f'Asymmetry is: {asymmetry}\n')
        print(f'Anisotropy is: {anisotropy}\n')
        
        print('Summary of NMR parameters:\n')
        print(f'Shielding tensor: \n{shielding_tensor} \n')
        print(f'Isotropic shielding: {s_iso} \n')
        print(f'Mehring: \n{diagonal_mehring} \n')
        print(f'Unsorted Eigenvals: \n{eigenvals} \n')
        print(f'Span is: {span}\n')
        print(f'Skew is: {skew}\n')
        print(f'Haberlen is: {diagonal_haberlen}\n')
        print(f'Reduced anisotropy is: {red_anisotropy}\n')
        print(f'Asymmetry is: {asymmetry}\n')
        print(f'Anisotropy is: {anisotropy}\n')

        print('Proceeding...\n')
        print('Call tensor_to_euler for Euler angles extraction from eigenvectors...\n')

        print("# -------------------------------------------------- #")

        return shielding_tensor, s_iso, diagonal_mehring, eigenvals, eigenvecs, symmetry, span, skew, diagonal_haberlen, red_anisotropy, asymmetry, anisotropy
    
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
                #if line.strip().startswith("COMPOUND JOB"):
                if line.strip().startswith("$$$$$$$$$$$$$$$$  JOB NUMBER"):
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
        #job_matching = re.compile(r'COMPOUND\s+JOB\s+(\d+)')
        job_matching = re.compile(r'\$+\s*JOB\s+NUMBER\s+(\d+) \$+')

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
                if 'CHEMICAL SHIFTS' in line:
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
#                        MOLECULAR VISUALIZATION AND PLOTTING
# ------------------------------------------------------------------------------




# ------------------------------------------------------------------------------
#                       DESCRIPTORS AND FEATURE SELECTION
# ------------------------------------------------------------------------------

