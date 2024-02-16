###################################################
# FROM ORCA OUTPUT TO CSV FILE FOR ML APPLICATION #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 15.02.2024                 #
#               Last:  15.02.2024                 #
#               -----------------                 #
###################################################

import pandas as pd
import os
import sys

class GenerateMLDataset:
    
    def __init__(self, root_directory, output_csv_path):
        
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
                        'functional', # Categorical
                        'basis_set', # Categorical
                        'aromatic' # Binary             
                        ]
        # Create the dataframe
        self.df = pd.DataFrame(columns=self.columns)

    def extract_data_from_file(self, file_path):
        # Your code to extract data from the file
        # Return the extracted data in a suitable format
        pass

    def search_files(self):
        # Iterate through all directories and subdirectories
        for root, dirs, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    data = self.extract_data_from_file(file_path)
                    
                    # Add the extracted data to the DataFrame
                    self.df = self.df.append(data, ignore_index=True)

    def save_to_csv(self):
        self.df.to_csv(self.output_csv_path, index=False)

def main():

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
    version = '0.0.1'
    print("Stable version: {}\n\n".format(version))
    print("Working python version:")
    print(sys.version)

    current_dir = os.getcwd()
    print(f'Current working directory is: {current_dir}')

    root_directory = os.path.join(current_dir, 'Machine_learning/datasets/MODELS')
    print(f'Current dataset root directory is: {root_directory}')
    
    output_csv_path = os.path.join(current_dir, 'Machine_learning/datasets/MODELS/test.csv')
    print(f'Current dataset directory is: {output_csv_path}')

    generate_dataset = GenerateMLDataset(root_directory, output_csv_path)
    generate_dataset.search_files()
    generate_dataset.save_to_csv()

if __name__ == "__main__":
    main()

