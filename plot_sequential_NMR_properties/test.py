# SECTION 10: PLOTTING NMR DATA (III) IN PAS: DIAMAGNETIC, PARAMAGNETIC, TOTAL CSA TENSOR

import os

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
                        print('Error encountered when extracting sDSO diamagnetic tensor components')
                        continue

                elif line.startswith('sPSO'):
                    try:
                        para_tensor_components = [float(x) for x in line.split()[1:4]]
                        current_para_shielding.append(para_tensor_components)
                    except (ValueError, IndexError):
                        print('Error encountered when extracting sPSO paramagnetic tensor components')
                        continue

                elif line.startswith('Total'):
                    try:
                        tot_tensor_components = [float(x) for x in line.split()[1:4]]
                        current_tot_shielding.append(tot_tensor_components)
                    except (ValueError, IndexError):
                        print('Error encountered when extracting total shielding tensor components')
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

dir = os.getcwd()
orca = os.path.join(dir, 'split_output/splitted_orca_job1.out')
diamagnetic, paramagnetic, total, nuc_identity = extract_csa_tensor_in_pas(orca)
print("Diamagnetic:", diamagnetic)
print("Paramagnetic:", paramagnetic)
print("Total:", total)
#print("Nuclear Identities:", nuc_identity)
print(type(diamagnetic))