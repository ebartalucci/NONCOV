###################################################
# FROM ORCA OUTPUT TO CSV FILE FOR ML APPLICATION #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 15.02.2024                 #
#               Last:  26.02.2024                 #
#               -----------------                 #
###################################################

import pandas as pd
import os
import sys
#from utils.nmr_functions import NMRFunctions

class GenerateMLDataset:
    
    def __init__(self, root_directory, output_csv_path):

        # Print header and version
        print("\n\n          #################################################")
        print("          | --------------------------------------------- |")
        print("          |         (NC)^2I.py: NMR Calculations          |")
        print("          |         for Noncovalent Interactions          |")
        print("          | --------------------------------------------- |")
        print("          |                WORKFLOW STEP 6                |")
        print("          |                       -                       |")
        print("          |              MAKE DATASET TABLES              |")
        print("          |                                               |")
        print("          |               Ettore Bartalucci               |")
        print("          |     Max Planck Institute CEC & RWTH Aachen    |")
        print("          |            Worringerweg 2, Germany            |")
        print("          |                                               |")
        print("          #################################################\n")
        
        # Print versions
        version = '0.0.2'
        print("Stable version: {}\n\n".format(version))
        print("Working python version:")
        print(sys.version)
        print('\n')
        
        self.root_directory = root_directory
       
        self.output_csv_path = output_csv_path

        # Headers of features, total features = 19
        self.columns = ['molecule', # Categorical
                        'atom', # Categorical
                        'noncov', # Categorical
                        'x_coord', # Integer
                        'y_coord', # Integer
                        'z_coord', # Integer
                        'tot_shielding_11', # Integer
                        'tot_shielding_22', # Integer
                        'tot_shielding_33', # Integer
                        'dia_shielding_11', # Integer
                        'dia_shielding_22', # Integer
                        'dia_shielding_33', # Integer
                        'para_shielding_11', # Integer
                        'para_shielding_22', # Integer
                        'para_shielding_33', # Integer
                        'iso_shift', # Integer 
                        'nmr_functional', # Categorical
                        'nmr_basis_set', # Categorical
                        'aromatic' # Binary             
                        ]
        
        # Create the dataframe
        self.df = pd.DataFrame(columns=self.columns)

        # Keep the user happy by saying that something happened
        print(f'The dataset has been created and saved in: {self.output_csv_path}')
        print('\n')

    # Extract features from splitted orca output files
    def extract_data_for_ml_database(self, file_path):
        """
        Read the splitted output file from an ORCA calculation and features for machine learning.
        Input
        :param file_path: output file from orca in the form .out

        Output
        :nmr_functional
        :nmr_basis_set
        """
        # Define empty feature variables to extract and append to dataset
        molecule = []
        atom = []
        noncov = []
        x_coord = []
        y_coord = []
        z_coord = []
        tot_shielding_11 = []
        tot_shielding_22 = []
        tot_shielding_33 = []
        dia_shielding_11 = []
        dia_shielding_22 = []
        dia_shielding_33 = []
        para_shielding_11 = []
        para_shielding_22 = []
        para_shielding_33 = []
        iso_shift = []
        nmr_functional = []
        nmr_basis_set = []
        aromatic = []

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Get molecule info from input orca file, the flag is in the commented section as $Molecule:

                # Get atom and relative coordinates info from file
                coordinates_found = False
                for line in lines:
                    if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                        coordinates_found = True
                    elif coordinates_found and line.strip():
                        atomic_info = line.split()
                        atom = atomic_info[0]
                        x_coord = float(atomic_info[1])
                        y_coord = float(atomic_info[2])
                        z_coord = float(atomic_info[3])
                        print(f'{atom}, {x_coord}')
                    elif coordinates_found and not line.strip():
                        break
                


                # Get functional and basis set for NMR calculations
                # Search for the line containing "# Level of theory"
                for i, line in enumerate(lines):
                    if "# Level of theory" in line:
                        # Extract the line immediately after it, this wont work if ppl dont use my syntax
                        level_of_theory_line = lines[i + 1].strip()

                        # Remove the line number from the line - do i want to keep this?
                        level_of_theory = level_of_theory_line.replace("| 10> !", "").strip()
                        level_of_theory = level_of_theory.split()

                        # Extract functional and basis set
                        nmr_functional = level_of_theory[0]
                        print(f'Functional for NMR calculations is: {nmr_functional}\n')
                        nmr_basis_set = level_of_theory[1]
                        print(f'Basis set for NMR calculations is: {nmr_basis_set}\n')

                        return nmr_functional, nmr_basis_set
                    
                    else: # if no level of theory found avoid crash by feeding empty list to the database
                        print('No level of theory information has been found in file. Please check the input data.')
        
        except FileNotFoundError:
            return f"File '{file_path}' not found."
        except Exception as e:
            return f"An error occurred: {str(e)}"


    # Search for all the splitted output files from an ORCA calculation in the Machine learning project root directory
    def search_files(self):
        # Iterate through all directories and subdirectories in root for the orca output files
        for root, dirs, files in os.walk(self.root_directory):
            for file in files:
                if file.startswith('splitted_') and file.endswith('.out'): # working with output files is much easier than with full mpi8.out
                    # get the path to those files
                    file_path = os.path.join(root, file)

                    # extract the required data from each file, this will be your instance vector nu_instance
                    instance_data = self.extract_data_for_ml_database(file_path)
                    
                    # Add the extracted data to the DataFrame
                    self.df = self.df.append(instance_data, ignore_index=True)

                    # Write to CSV file (check if maybe another format is better, no excel since its propertary)
                    self.df.to_csv(self.output_csv_path, index=False)
                
                # Raise error if no data in folder
                else:
                    
                    print('No raw data has been found in root with the following characteristics: startswith: splitted_, endswith: .out. Please adjust your search options.')





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

