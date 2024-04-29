###################################################
# PLOT AMINO ACID STATISTICS IN PROTEIN SEQUENCE  #
# ----------------------------------------------- #
#               Ettore Bartalucci                 #
#               First: 23.02.2024                 #
#               Last:  23.02.2024                 #
#               -----------------                 #
#             Stable release version              #
#                   v.0.0.1                       #
#                                                 #
###################################################

import matplotlib.pyplot as plt
import numpy as np
import os

class AminoStat():
    
    # Print header when code is ran
    def __init__(self):
        # Print header and version
        print("\n\n          #################################################")
        print("          | --------------------------------------------- |")
        print("          |  Plot statistics of amino acids distribution  |")
        print("          |           in given protein sequence           |")
        print("          | --------------------------------------------- |")
        print("          |                       -                       |")
        print("          |           NMR FUNCTIONS COLLECTIONS           |")
        print("          |                                               |")
        print("          |               Ettore Bartalucci               |")
        print("          |     Max Planck Institute CEC & RWTH Aachen    |")
        print("          |            Worringerweg 2, Germany            |")
        print("          |                                               |")
        print("          #################################################\n")
        pass

    # Take input protein sequence and add a space between each one-letter code amino acid
    def space_prot_seq(self, prot_seq, spaced_prot_seq):
        try:
            with open(prot_seq, 'r') as f:
                sequence = f.read()

            spaced_sequence = ' '.join(sequence)

            with open(spaced_prot_seq, 'w') as f:
                f.write(spaced_sequence)

            print(f'Protein sequence from Uniprot now contains spaces between each amino acid letter and has been written to: {spaced_prot_seq}.')
            print('Continuing to next step...')

        except FileNotFoundError:
            print('Input file not found, please specify')
        except:
            print('An error occurred')

    # Count how many amino acids of each type you have in the sequence
    def count_amino_acids(self, prot_seq, count_file):
        try:
            with open(prot_seq, 'r') as f:
                sequence = f.read()

            amino_acid_count = {}
            for amino_acid in sequence:
                if amino_acid in amino_acid_count:
                    amino_acid_count[amino_acid] += 1
                else:
                    amino_acid_count[amino_acid] = 1

            with open(count_file, 'w') as f:
                for amino_acid, count in amino_acid_count.items():
                    f.write(f"{amino_acid}: {count}\n")

            print('Amino acid counts written to amino_acid_count.txt')
            print('Continuing to next step...')
        except FileNotFoundError:
            print('Input file not found. please specify')
        except:
            print('An error occurred')

    # Plot an histogram of the relative distribution of the amino acids in the sequence
    def plot_amino_acid_statistics(self, count_file, plot_file):
        amino_acid_counts = {}
        total_count = 0

        with open(count_file, 'r') as f:
            for line in f:
                pairs = line.strip().split(': ')
                if len(pairs) == 2:
                    amino_acid, count = pairs
                    amino_acid_counts[amino_acid] = int(count)
                    total_count += int(count)

        amino_acids = list(amino_acid_counts.keys())
        counts = list(amino_acid_counts.values())

        percentages = [count / total_count * 100 for count in counts]

        plt.bar(amino_acids, percentages)
        plt.xlabel('Amino Acid')
        plt.ylabel('Percentage (%)')
        #plt.ylim(0, 20)  # Set the y-axis limit to 0% to 18%
        plt.title('Amino Acids Distribution')
        
        for i in range(len(amino_acids)):
            plt.text(amino_acids[i], percentages[i], f"{counts[i]} ({percentages[i]:.2f}%)", ha='center', va='bottom', rotation=90)
        
        plt.savefig(plot_file)
        plt.show()

    # Calculate amino acid occurrence statistics
    def calculate_amino_acid_percentage(self, prot_seq):
        amino_acids_list = "ARNDCEQGHILKMFPSTWYV"  # List of 20 standard amino acids
        aa_percentage = {aa: prot_seq.count(aa) for aa in amino_acids_list}
        return aa_percentage
    
    # Ask the user information on the protein domains
    def define_protein_domains(self):
        print('Please define your domains according to Uniprot information.')
        n_domains = int(input('How many domains does your protein have? '))

        prot_domain_names = []
        prot_domain_boundaries = [] 

        try:
            for i in range(n_domains):
                prot_domain_name = input(f'Enter name of domain {i+1}: ')
                prot_domain_boundary = input(f'Enter boundaries for domain {i+1} position: ')

                prot_domain_names.append(prot_domain_name)
                prot_domain_boundaries.append(prot_domain_boundary)

            return prot_domain_names, prot_domain_boundaries
        except:
            print('An error occurred')
            
        print(f'The domains of your protein are: {prot_domain_names} in regions {prot_domain_boundaries}')





# Example usage
current_dir = os.getcwd()

protein_sequence = os.path.join(current_dir, 'scratch/amino_acid_stats/spidersilks.txt')
spaced_sequence = os.path.join(current_dir, 'scratch/amino_acid_stats/spaced_spidersilks.txt')
count_file = os.path.join(current_dir, 'scratch/amino_acid_stats/silks_amino_acid_count.txt')
plot_file = os.path.join(current_dir, 'scratch/amino_acid_stats/silks_amino_acid_statistics.pdf')

amino_stats = AminoStat()

amino_stats.space_prot_seq(protein_sequence, spaced_sequence)
amino_stats.count_amino_acids(spaced_sequence, count_file)
amino_stats.plot_amino_acid_statistics(count_file, plot_file)

amino_stats.define_protein_domains()
