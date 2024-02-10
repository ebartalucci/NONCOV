###################################################
# PLOT ORCA OUTPUT FOR ITERATIVE NMR CALCULATIONS #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 16.09.2023                 #
#               Last:  07.02.2024                 #
#               -----------------                 #
#             Stable release version              #
#                   v.1.1.0                       #
#                                                 #
###################################################

# If this work leads to publication, the following list shall be acknowledged:
# - Ms. Olivia Gampp for helpful discussions and suggestions in displacement of fragments
# - ChatGPT v3.5 used for debugging and support with the coding
# - 

# Import modules 
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import re # RegEx

# -------------------------- DONE --------------------------------#
# ----------------------------------------------------------------#
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

# ----------------------------------------------------------------#


# -------------------------- DONE --------------------------------#
# ----------------------------------------------------------------#
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

# ----------------------------------------------------------------#


# -------------------------- DONE --------------------------------#
# ----------------------------------------------------------------#
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

# ----------------------------------------------------------------#


# -------------------------- DONE --------------------------------#
# ----------------------------------------------------------------#
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

# ----------------------------------------------------------------#


# -------------------------- Start --------------------------------#
# ----------------------------------------------------------------#
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

# ----------------------------------------------------------------#


# -------------------------- DONE --------------------------------#
# ----------------------------------------------------------------#
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

# ----------------------------------------------------------------#


# -------------------------- DONE --------------------------------#
# ----------------------------------------------------------------#
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

# ----------------------------------------------------------------#


# -------------------------- halfway --------------------------------#
# ----------------------------------------------------------------#
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
    if user_choice == 1: #Cation-pi interaction from https://doi.org/10.1016%2Fj.jmb.2021.167035
        return 2, 6
    elif user_choice == 2: #Anion-pi interaction from https://doi.org/10.1039%2Fc5sc01386k
        return 2, 5
    elif user_choice == 3: #pi-pi interaction
        return 1, 5
    elif user_choice == 4: #H-bond interaction from https://doi.org/10.1016/B978-012486052-0/50005-1
        return 2.7, 3.3
    elif user_choice == 5: #Polar-pi interaction
        return 1, 5
    elif user_choice == 6: #n-pi* interaction
        return 1, 5

# ----------------------------------------------------------------#


# -------------------------- start --------------------------------#
# ----------------------------------------------------------------#
# SECTION 9: EXTRACT INITIAL DISTANCE BETWEEN NUCLEAR PAIRS FOR DISTANCE PLOTS

# ----------------------------------------------------------------#

# -------------------------- start --------------------------------#
# ----------------------------------------------------------------#
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

    return sigma_dia, sigma_para, sigma_tot

# ----------------------------------------------------------------#


# -------------------------- start --------------------------------#
# ----------------------------------------------------------------#
# SECTION 11: PLOT MOLECULAR FRAME AND TENSORS ELLIPSOIDS
def plot_3d_molecule_and_pas_tensor(molecule_path, sizes=None):
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

# ----------------------------------------------------------------#


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
        sigma_dia, sigma_para, sigma_tot = extract_csa_tensor_in_pas(orca_output_splitted)

        # Append PAS diagonal tensors components to respective variables
        S_dia.append(sigma_dia)
        S_para.append(sigma_para)
        S_tot.append(sigma_tot)


    # ----------------- PLOTS SECTION ---------------- #
    # Calculate MAX/MIN for the values of the isotropic shifts
    #sigma_iso_max = max(Siso)
    #sigma_iso_min = min(Siso)
    #print(sigma_iso_max)
    #print(sigma_iso_min)
    #nucleus_label = 'Nucleus 9 F'
    #delta_sigma = sigma_iso_max[0] - sigma_iso_min[0]

    # Plot the max min values of the isotropic shift
    #plt.figure(1)
    #plt.scatter([0, 0], [sigma_iso_min, sigma_iso_max], color='blue', label= r'$\Delta$$\delta$ is:' + str(delta_sigma))
    #plt.plot([0, 0], [sigma_iso_min, sigma_iso_max], color='blue', linestyle='-', linewidth=2)
    #plt.xlabel(nucleus_label)
    #plt.ylabel('Shift / ppm')
    #plt.legend(loc='best')
    #plt.show()

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
    for nucleus_key in nuclear_identities:
        # Extract shielding values for the current nucleus from each dictionary
        nucleus_values_S_dia = [d.get(nucleus_key, [])[0] for d in S_dia]
        nucleus_values_S_para = [d.get(nucleus_key, [])[0] for d in S_para]
        nucleus_values_S_tot = [d.get(nucleus_key, [])[0] for d in S_tot]

        # Split the nucleus_key into a tuple (nucleus number, element)
        nucleus = tuple(nucleus_key.split())

        # Plot the shielding values for the current nucleus
        plt.plot(displacement_steps_distance, nucleus_values_S_dia, marker='o', linestyle='-', color='darkblue', label=r'$\sigma$_xx')
        plt.plot(displacement_steps_distance, nucleus_values_S_para, marker='o', linestyle='-', color='orangered', label=r'$\sigma$_yy')
        plt.plot(displacement_steps_distance, nucleus_values_S_tot, marker='o', linestyle='-', color='gold', label=r'$\sigma$_zz')
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


    # ------------------------------------------------ #


if __name__ == '__main__':
    main()