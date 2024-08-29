###################################################
#                                                 #
#               Ettore Bartalucci                 #
#               First: 27.05.2024                 #
#               Last:  27.05.2024                 #
#               -----------------                 #
#             Stable release version              #
#                   v.0.0.1                       #
#                                                 #
###################################################

import noncov as NONCOV
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Get path
current_dir = os.getcwd()
print(f'Current working directory is: {current_dir}')

# Call all modules
def main():
    # need to call here an if cycle for either do post processing or structure generation

    # --- Working with the ORCA output file --- #
    #orca_output = os.path.join(current_dir, 'run_all_displaced_distances.mpi8.out')
    orca_output = input("Enter the path to the ORCA file you want to work with: ")

    # Count number of simulation jobs that were ran
    n_jobs = NONCOV.NONCOVToolbox.OrcaAnalysis.count_jobs_number(orca_output)
    print(f'Number of ORCA jobs in file: {n_jobs}')
    if n_jobs > 20:
        print(f'Careful, you are working with a possibly large output file of several GB\n')
        print(f'If using version controls consider setting up a .gitignore \n')

    # Extract level of theory
    lot_out = NONCOV.NONCOVToolbox.OrcaAnalysis.extract_level_of_theory(orca_output)
    print(f'Level of theory for the NMR calculations is: {lot_out}\n')

    # Split orca output in several subfiles
    if n_jobs > 2:
        print('Your output file will be now spilt into subfiles. \n')
        NONCOV.NONCOVToolbox.OrcaAnalysis.split_orca_output(orca_output)
    # ---------------------------------------------- #


    # --- Define boundary distance ranges in NONCOV --- #
    # Define the boundaries ([A]) for various noncovalent interactions
    NONCOV.NONCOVToolbox.OrcaAnalysis.run_boundary_checks()
    # ------------------------------------------------ #

    # --- Extract the various CSA tensor components --- #
    # Initialize displacement steps in Angstrom - need to find a clever way to do this
    displacement_steps_distance = [job * 0.25 for job in range(1,n_jobs+1)]

    # Initialize variables for shielding tensor components
    S_dia = []
    S_para = []
    S_tot = []
    nuclear_identities = []
    

    # Extract NMR data from each splitted file
    for job_number in range (1, n_jobs+1): # split files = number of jobs
        
        # All splitted outputs from main big .out MPI8 file
        orca_splitted_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scratch/OrcaAnalysis/split_orca_output', f'splitted_orca_job{job_number}.out'))

        # Extract CSA data
        S_dia, S_para, S_tot, nuclear_identities = NONCOV.NONCOVToolbox.OrcaAnalysis.extract_csa_data(orca_splitted_output, job_number)
        


#         # Get all the shieldings_i.txt files 
#         shielding_data = os.path.join(current_dir, f'nmr_data/shieldings_{job_number}.txt')
        
#         # Extract isotropic shifts
#         nucleus_data = extract_isotropic_shifts(shielding_data)

#         # Extract shielding tensor components for each nucleus
#         SXX, SYY, SZZ, nuclear_identities = extract_shielding_tensor(shielding_data)

#         # Append isotropic shifts
#         Siso.append(nucleus_data)
    
#         # Append xx,yy,zz shielding tensor components (non-diagonalized)
#         Sxx.append(SXX)
#         Syy.append(SYY)
#         Szz.append(SZZ)

#         # Extract various components of diagonal shielding tensor in PAS
#         sigma_dia, sigma_para, sigma_tot, nuclear_identities_2 = extract_csa_tensor_in_pas(orca_output_splitted)

#         # Append PAS diagonal tensors components to respective variables
#         S_dia.append(sigma_dia)
#         S_para.append(sigma_para)
#         S_tot.append(sigma_tot)



#     # ----------------- PLOTS SECTION ---------------- #

#     # Create a folder to save the shifts plots as PDFs and JPEG if it doesn't exist
#     shifts_figures_folder = 'shifts_plots'
#     os.makedirs(shifts_figures_folder, exist_ok=True)

#     # Plot the shielding parameters for each nucleus
#     for nucleus_key in nuclear_identities:
#         # Extract shielding values for the current nucleus from each dictionary
#         nucleus_values_Sxx = [d.get(nucleus_key, [])[0] for d in Sxx]
#         nucleus_values_Syy = [d.get(nucleus_key, [])[0] for d in Syy]
#         nucleus_values_Szz = [d.get(nucleus_key, [])[0] for d in Szz]
#         nucleus_values_Siso = [d.get(nucleus_key, [])[0] for d in Siso]

#         # Split the nucleus_key into a tuple (nucleus number, element)
#         nucleus = tuple(nucleus_key.split())

#         # Plot the shielding values for the current nucleus
#         plt.plot(displacement_steps_distance, nucleus_values_Sxx, marker='o', linestyle='-', color='darkblue', label=r'$\sigma$_xx')
#         plt.plot(displacement_steps_distance, nucleus_values_Syy, marker='o', linestyle='-', color='orangered', label=r'$\sigma$_yy')
#         plt.plot(displacement_steps_distance, nucleus_values_Szz, marker='o', linestyle='-', color='gold', label=r'$\sigma$_zz')
#         plt.plot(displacement_steps_distance, nucleus_values_Siso, marker='*', linestyle='-', color='magenta', label=r'$\sigma$_iso')

#         # Highlight the NONCOV effective region
#         plt.axvspan(min_distance_value, max_distance_value, alpha=0.2, color='grey', label='NONCOV \n effective region')
        
#         # Set labels and title
#         plt.xlabel('Displacement from initial geometry / Å')
#         plt.ylabel('Shielding / ppm')
#         plt.title(f'Nucleus {nucleus[1]} {nucleus[2]}')
        
#         # Display legend
#         plt.legend(loc='best')
        
#         # Save the plot as a PDF in the output folder
#         pdf_filename = os.path.join(shifts_figures_folder, f'nucleus_{nucleus[1]}_{nucleus[2]}.pdf')
#         plt.savefig(pdf_filename, bbox_inches='tight')

#         # Save the plot as a jpeg in the output folder
#         jpg_filename = os.path.join(shifts_figures_folder, f'nucleus_{nucleus[1]}_{nucleus[2]}.jpg')
#         plt.savefig(jpg_filename, bbox_inches='tight')
        
#         # Show the plot (optional, can be commented out if you don't want to display the plots)
#         #plt.show()

#         # Clear the current figure for the next iteration
#         plt.clf()


#     # Create a folder to save the shifts plots as PDFs and JPEG if it doesn't exist
#     pas_tensors_figures_folder = 'tensor_plots'
#     os.makedirs(pas_tensors_figures_folder, exist_ok=True)

#     # Plot the shielding parameters for each nucleus
#     for nucleus_key_2 in nuclear_identities_2:
#         # Extract individual contributions to shielding values for the current nucleus from each dictionary
#         nucleus_values_S_dia = [d.get(nucleus_key_2, [])[0] for d in S_dia]
#         nucleus_values_S_para = [d.get(nucleus_key_2, [])[0] for d in S_para]
#         nucleus_values_S_tot = [d.get(nucleus_key_2, [])[0] for d in S_tot]

#         # Split the nucleus_key into a tuple (nucleus number, element)
#         nucleus_2 = tuple(nucleus_key_2.split())

#         # Extract 11, 22 and 33 components of the tensor for each contribution
#         S_dia_11 = []
#         S_dia_22 = []
#         S_dia_33 = []

#         S_para_11 = []
#         S_para_22 = []
#         S_para_33 = []

#         S_tot_11 = []
#         S_tot_22 = []
#         S_tot_33 = []

#         # Loop through dict
#         # for nucleus in directory, for tensor component in nucleus append

#         # Plot the shielding values for the current nucleus
#         plt.plot(displacement_steps_distance, nucleus_values_S_dia, marker='o', linestyle='-', color='darkblue', label=r'$\sigma$_dia_11')
#         plt.plot(displacement_steps_distance, nucleus_values_S_para, marker='o', linestyle='-', color='orangered', label=r'$\sigma$_para_11')
#         plt.plot(displacement_steps_distance, nucleus_values_S_tot, marker='o', linestyle='-', color='gold', label=r'$\sigma$_tot_11')
#         plt.plot(displacement_steps_distance, nucleus_values_Siso, marker='*', linestyle='-', color='magenta', label=r'$\sigma$_iso')

#         # Highlight the NONCOV effective region
#         #plt.axvspan(min_distance_value, max_distance_value, alpha=0.2, color='grey', label='NONCOV \n effective region')
        
#         # Set labels and title
#         plt.xlabel('Displacement from initial geometry / Å')
#         plt.ylabel('Shielding / ppm')
#         plt.title(f'Nucleus {nucleus_2[1]} {nucleus_2[2]}')
        
#         # Display legend
#         plt.legend(loc='best')
        
#         # Save the plot as a PDF in the output folder
#         pdf_filename = os.path.join(pas_tensors_figures_folder, f'nucleus_{nucleus_2[1]}.pdf')
#         plt.savefig(pdf_filename, bbox_inches='tight')

#         # Save the plot as a jpeg in the output folder
#         jpg_filename = os.path.join(pas_tensors_figures_folder, f'nucleus_{nucleus_2[1]}.jpg')
#         plt.savefig(jpg_filename, bbox_inches='tight')
        
#         # Show the plot (optional, can be commented out if you don't want to display the plots)
#         #plt.show()

#         # Clear the current figure for the next iteration
#         plt.clf()


#     # ------------------------------------------------ #




# def main():

# current_dir = os.getcwd()
# print(f'Current working directory is: {current_dir}')

# root_directory = os.path.join(current_dir, 'Machine_learning/raw')
# print(f'Dataset root directory is: {root_directory}')

# output_csv_path = os.path.join(current_dir, 'Machine_learning/datasets/model_structures/test.csv')
# print(f'Dataset directory is: {output_csv_path}')

# generate_dataset = GenerateMLDataset(root_directory, output_csv_path)
# generate_dataset.search_files()

# if __name__ == "__main__":
# main()


# # Example usage
# toolbox = NONCOVToolbox()
# amino_stats = toolbox.AminoStat()

# # Example usage
# current_dir = os.getcwd()

# protein_sequence = os.path.join(current_dir, 'scratch/amino_acid_stats/spidersilks.txt')
# spaced_sequence = os.path.join(current_dir, 'scratch/amino_acid_stats/spaced_spidersilks.txt')
# count_file = os.path.join(current_dir, 'scratch/amino_acid_stats/silks_amino_acid_count.txt')
# plot_file = os.path.join(current_dir, 'scratch/amino_acid_stats/silks_amino_acid_statistics.pdf')

# #amino_stats = AminoStat()

# amino_stats.space_prot_seq(protein_sequence, spaced_sequence)
# amino_stats.count_amino_acids(spaced_sequence, count_file)
# amino_stats.plot_amino_acid_statistics(count_file, plot_file)

# amino_stats.define_protein_domains()

if __name__ == '__main__':
    main()


def main(molecule_path):

    mol_graph = MolecularGraph()

    # Parse the XYZ file
    atom_types, coordinates = mol_graph.parse_xyz(molecule_path)
    
    # Calculate pairwise distances
    distances = mol_graph.calculate_distances(coordinates)
    
    # Detect covalent bonds
    covalent_bonds = mol_graph.detect_bonds(atom_types, distances)
    
    # Detect non-covalent interactions
    noncovalent_interactions = mol_graph.detect_noncovalent_interactions(atom_types, distances)
    
    # Build the molecular graph
    #mol_graph = mol_graph.build_molecular_graph(atom_types, coordinates, covalent_bonds, noncovalent_interactions)
    
    # Visualize the molecular graph
    #mol_graph.draw()

    # Plots 
    mol_graph.plot_bond_dist_matrix(covalent_bonds, distances, atom_types)
    mol_graph.plot_noncov_distance_map(noncovalent_interactions, atom_types)

    # Build different graphs
    covalent_bonds_graph = mol_graph.build_covalent_bonds_graph(atom_types, coordinates, covalent_bonds)
    intramolecular_graph = mol_graph.build_intramolecular_graph(atom_types, coordinates, covalent_bonds, noncovalent_interactions)
    intermolecular_graph = mol_graph.build_intermolecular_graph(atom_types, coordinates, noncovalent_interactions)

    # Draw subplots while preserving atom positions
    mol_graph.draw_subplots(covalent_bonds_graph, intramolecular_graph, intermolecular_graph, coordinates)




threshold = 1.6

current_dir = os.getcwd()
print(f'Current working directory is: {current_dir}')
#molecule = os.path.join(current_dir, 'scratch/test_structs/caffeine.xyz')

molecule = 'D:/PhD/Data/DFT/NONCOV/DFT_simulations/codes/scratch/test_structs/benzene_H2O.xyz'

main(molecule)