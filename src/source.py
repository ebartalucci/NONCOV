        # ----------------------------------------------------------------#
        # ----------------------- MAIN BLOCK -----------------------------#
        # ----------------------------------------------------------------#
        # RELATIVE PATH
        current_dir = os.getcwd()
        print(f'Current working directory is: {current_dir}')

        # SECTION MAIN: Call all modules
        def main():

            # --- Wroking with the orca MPI8 output file --- #
            # Assign MPI8 output file
            orca_output = os.path.join(current_dir, 'run_all_displaced_distances.mpi8.out')

            # Count number of simulation jobs that were ran
            n_jobs = count_jobs_number(orca_output)
            print(f'Number of ORCA DFT calculations in file: {n_jobs}')

            # Extract level of theory
            lot_out = extract_level_of_theory(orca_output)
            print(f'Level of theory for the NMR calculations is: {lot_out}')

            # Read J couplings: to be finished, only call within if statement if 'ssall' in input file
            # read_couplings(orca_output)

            # Split orca output in several subfiles
            #split_orca_output(orca_output)
            # ---------------------------------------------- #


            # --- Define boundary distance ranges in NONCOV --- #
            # Define the boundaries ([A]) for the regions for various types of noncovalent interactions
            # Get user's NONCOV choice
            noncov_type = set_noncov_interactions()

            # Set min and max values based on user's choice
            min_distance_value, max_distance_value = set_boundary_distance_values(noncov_type)

            # Display the selected values
            print(f"Selected boundary distance values / Å: min={min_distance_value}, max={max_distance_value}")

            # ------------------------------------------------ #


            # --- Working with property files from ORCA --- #
            # Initialize displacement steps in Angstrom
            displacement_steps_distance = [job * 0.25 for job in range(1,n_jobs+1)]

            # Initialize isotropic shift variable
            Siso = []

            # Initialize variables for shielding tensor components
            Sxx = []
            Syy = []
            Szz = []

            # Initialize variables for diagonal shielding tensor components in PAS
            S_dia = []
            S_para = []
            S_tot = []

            # Initialize nuclear identities
            nuclear_identities = []
            nuclear_identities_2 = []
        
            # Extract NMR data from parameter files
            for job_number in range (1, n_jobs+1): #Property files = number of jobs

                # Property files from all jobs (except first)
                orca_properties = os.path.join(current_dir, f'properties/run_all_displaced_distances_job{job_number}_property.txt')

                # All splitted outputs from main big .out MPI8 file
                orca_output_splitted = os.path.join(current_dir, f'split_output/splitted_orca_job{job_number}.out')
                
                # Read each of them
                read_property_file(orca_properties, job_number)

                # Get all the shieldings_i.txt files 
                shielding_data = os.path.join(current_dir, f'nmr_data/shieldings_{job_number}.txt')
                
                # Extract isotropic shifts
                nucleus_data = extract_isotropic_shifts(shielding_data)

                # Extract shielding tensor components for each nucleus
                SXX, SYY, SZZ, nuclear_identities = extract_shielding_tensor(shielding_data)

                # Append isotropic shifts
                Siso.append(nucleus_data)
            
                # Append xx,yy,zz shielding tensor components (non-diagonalized)
                Sxx.append(SXX)
                Syy.append(SYY)
                Szz.append(SZZ)

                # Extract various components of diagonal shielding tensor in PAS
                sigma_dia, sigma_para, sigma_tot, nuclear_identities_2 = extract_csa_tensor_in_pas(orca_output_splitted)

                # Append PAS diagonal tensors components to respective variables
                S_dia.append(sigma_dia)
                S_para.append(sigma_para)
                S_tot.append(sigma_tot)
        


            # ----------------- PLOTS SECTION ---------------- #

            # Create a folder to save the shifts plots as PDFs and JPEG if it doesn't exist
            shifts_figures_folder = 'shifts_plots'
            os.makedirs(shifts_figures_folder, exist_ok=True)

            # Plot the shielding parameters for each nucleus
            for nucleus_key in nuclear_identities:
                # Extract shielding values for the current nucleus from each dictionary
                nucleus_values_Sxx = [d.get(nucleus_key, [])[0] for d in Sxx]
                nucleus_values_Syy = [d.get(nucleus_key, [])[0] for d in Syy]
                nucleus_values_Szz = [d.get(nucleus_key, [])[0] for d in Szz]
                nucleus_values_Siso = [d.get(nucleus_key, [])[0] for d in Siso]

                # Split the nucleus_key into a tuple (nucleus number, element)
                nucleus = tuple(nucleus_key.split())

                # Plot the shielding values for the current nucleus
                plt.plot(displacement_steps_distance, nucleus_values_Sxx, marker='o', linestyle='-', color='darkblue', label=r'$\sigma$_xx')
                plt.plot(displacement_steps_distance, nucleus_values_Syy, marker='o', linestyle='-', color='orangered', label=r'$\sigma$_yy')
                plt.plot(displacement_steps_distance, nucleus_values_Szz, marker='o', linestyle='-', color='gold', label=r'$\sigma$_zz')
                plt.plot(displacement_steps_distance, nucleus_values_Siso, marker='*', linestyle='-', color='magenta', label=r'$\sigma$_iso')

                # Highlight the NONCOV effective region
                plt.axvspan(min_distance_value, max_distance_value, alpha=0.2, color='grey', label='NONCOV \n effective region')
                
                # Set labels and title
                plt.xlabel('Displacement from initial geometry / Å')
                plt.ylabel('Shielding / ppm')
                plt.title(f'Nucleus {nucleus[1]} {nucleus[2]}')
                
                # Display legend
                plt.legend(loc='best')
                
                # Save the plot as a PDF in the output folder
                pdf_filename = os.path.join(shifts_figures_folder, f'nucleus_{nucleus[1]}_{nucleus[2]}.pdf')
                plt.savefig(pdf_filename, bbox_inches='tight')

                # Save the plot as a jpeg in the output folder
                jpg_filename = os.path.join(shifts_figures_folder, f'nucleus_{nucleus[1]}_{nucleus[2]}.jpg')
                plt.savefig(jpg_filename, bbox_inches='tight')
                
                # Show the plot (optional, can be commented out if you don't want to display the plots)
                #plt.show()

                # Clear the current figure for the next iteration
                plt.clf()


            # Create a folder to save the shifts plots as PDFs and JPEG if it doesn't exist
            pas_tensors_figures_folder = 'tensor_plots'
            os.makedirs(pas_tensors_figures_folder, exist_ok=True)

            # Plot the shielding parameters for each nucleus
            for nucleus_key_2 in nuclear_identities_2:
                # Extract individual contributions to shielding values for the current nucleus from each dictionary
                nucleus_values_S_dia = [d.get(nucleus_key_2, [])[0] for d in S_dia]
                nucleus_values_S_para = [d.get(nucleus_key_2, [])[0] for d in S_para]
                nucleus_values_S_tot = [d.get(nucleus_key_2, [])[0] for d in S_tot]

                # Split the nucleus_key into a tuple (nucleus number, element)
                nucleus_2 = tuple(nucleus_key_2.split())

                # Extract 11, 22 and 33 components of the tensor for each contribution
                S_dia_11 = []
                S_dia_22 = []
                S_dia_33 = []

                S_para_11 = []
                S_para_22 = []
                S_para_33 = []

                S_tot_11 = []
                S_tot_22 = []
                S_tot_33 = []

                # Loop through dict
                # for nucleus in directory, for tensor component in nucleus append

                # Plot the shielding values for the current nucleus
                plt.plot(displacement_steps_distance, nucleus_values_S_dia, marker='o', linestyle='-', color='darkblue', label=r'$\sigma$_dia_11')
                plt.plot(displacement_steps_distance, nucleus_values_S_para, marker='o', linestyle='-', color='orangered', label=r'$\sigma$_para_11')
                plt.plot(displacement_steps_distance, nucleus_values_S_tot, marker='o', linestyle='-', color='gold', label=r'$\sigma$_tot_11')
                plt.plot(displacement_steps_distance, nucleus_values_Siso, marker='*', linestyle='-', color='magenta', label=r'$\sigma$_iso')

                # Highlight the NONCOV effective region
                #plt.axvspan(min_distance_value, max_distance_value, alpha=0.2, color='grey', label='NONCOV \n effective region')
                
                # Set labels and title
                plt.xlabel('Displacement from initial geometry / Å')
                plt.ylabel('Shielding / ppm')
                plt.title(f'Nucleus {nucleus_2[1]} {nucleus_2[2]}')
                
                # Display legend
                plt.legend(loc='best')
                
                # Save the plot as a PDF in the output folder
                pdf_filename = os.path.join(pas_tensors_figures_folder, f'nucleus_{nucleus_2[1]}.pdf')
                plt.savefig(pdf_filename, bbox_inches='tight')

                # Save the plot as a jpeg in the output folder
                jpg_filename = os.path.join(pas_tensors_figures_folder, f'nucleus_{nucleus_2[1]}.jpg')
                plt.savefig(jpg_filename, bbox_inches='tight')
                
                # Show the plot (optional, can be commented out if you don't want to display the plots)
                #plt.show()

                # Clear the current figure for the next iteration
                plt.clf()


            # ------------------------------------------------ #
    
    
    

    def main():

        current_dir = os.getcwd()
        print(f'Current working directory is: {current_dir}')

        root_directory = os.path.join(current_dir, 'Machine_learning/raw')
        print(f'Dataset root directory is: {root_directory}')
        
        output_csv_path = os.path.join(current_dir, 'Machine_learning/datasets/model_structures/test.csv')
        print(f'Dataset directory is: {output_csv_path}')

        generate_dataset = GenerateMLDataset(root_directory, output_csv_path)
        generate_dataset.search_files()

    if __name__ == "__main__":
        main()
        
        
# Example usage
toolbox = NONCOVToolbox()
amino_stats = toolbox.AminoStat()

# Example usage
current_dir = os.getcwd()

protein_sequence = os.path.join(current_dir, 'scratch/amino_acid_stats/spidersilks.txt')
spaced_sequence = os.path.join(current_dir, 'scratch/amino_acid_stats/spaced_spidersilks.txt')
count_file = os.path.join(current_dir, 'scratch/amino_acid_stats/silks_amino_acid_count.txt')
plot_file = os.path.join(current_dir, 'scratch/amino_acid_stats/silks_amino_acid_statistics.pdf')

#amino_stats = AminoStat()

amino_stats.space_prot_seq(protein_sequence, spaced_sequence)
amino_stats.count_amino_acids(spaced_sequence, count_file)
amino_stats.plot_amino_acid_statistics(count_file, plot_file)

amino_stats.define_protein_domains()

if __name__ == '__main__':
    main()