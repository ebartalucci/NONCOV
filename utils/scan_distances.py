
# Importing necessary modules
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plot
from sklearn.cluster import KMeans
from timeit import default_timer as timer

#-------------------------------------------------------------------#
#------------------------------FUNCTIONS----------------------------#
#-------------------------------------------------------------------#

# SECTION 1: READING XYZ ATOMIC COORDINATES FROM FILE AND SPLIT THE FRAGMENTS
def read_atomic_coord(file_path):
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

# SECTION 2: CALCULATION OF CENTROIDS AND STORING VALUES IN FILE
def calculate_centroids(coordinates):
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

def write_centroids(file_path, centroids):
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

# SECTION 3: CHECKPOINT COMPUTE TOPOLOGY AND K-NEAREST CLUSTERING FOR MOLECULAR CENTROIDS
def plot_starting_molecular_fragments(coords1, coords2, centroids):
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

# SECTION 4: READ USER PROVIDED INPUT FILE, SPLIT AND ASSIGN MOLECULAR FRAGMENTS TO INDIVIDUAL MOLECULES
def assign_molecule_fragments(coordinates, input_file):
    """
    Assign the respective fragments to atom in xyz list
    :param coordinates: atomic xyz coordinates
    :param input_file: specifies the numbering of the atoms of each fragment
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
        fragment1_indices = []
        fragment2_indices = []
        current_fragment = None
        for line in lines:
            if line.strip() == "$fragment1":
                current_fragment = fragment1_indices
            elif line.strip() == "$fragment2":
                current_fragment = fragment2_indices
            elif line.strip() == "$displacement":
                break
            else:
                index = int(line.strip()) - 1  # Convert 1-based index to 0-based index
                current_fragment.append(index)    
    coords1 = coordinates[fragment1_indices]
    coords2 = coordinates[fragment2_indices]
    return coords1, coords2

#-------------------------------------------------------------------#

# SECTION 5: DISPLACE THE TWO FRAGMENT ALONG THE CENTROID LINE MOVING ONE AND FIXING THE OTHER
def displace_fragment(coords1, displacement_direction, displacement_step, i):
    """
    This function displaces the fragment along the displacement_direction vector.
    :param coords1: coordinates of fragment 1 to be displaced
    :param displacement_direction: displace fragments along the direction connecting the two centroids
    :param displacement_step: how many angstroem to displace, specified in input file
    :param i: for the loop over displacements, specifies how many structures are generated. In future will be the dissociation limit value
    """
    displacement_direction /= np.linalg.norm(displacement_direction)  # Normalize the displacement direction vector
    displacement_vector = - displacement_direction * displacement_step * i # Displace along the normalized direction
    print(f'Displacement vector: {displacement_vector}')
    print(type(displacement_vector))
    return coords1 - displacement_vector  # Apply displacement by adding the vector
    
    # From olivia
    #displacement_vector = coords1 - displacement_direction * i  # displace along one axis 
    #print(f'Displacement vector:{displacement_vector}') # write this to log file
    #print(type(displacement_vector)) # write this to log file
    #return displacement_vector

#-------------------------------------------------------------------#

# SECTION 6: WRITE NEW DISPLACED COORDINATES TO FILES
def write_displaced_xyz_file(file_path, coords_fixed, coords_displaced, atom_identities):
    """
    This function writes both the fixed and displaced coordinates to an XYZ file.
    :param file_path: where the structure files are written
    :param coords_fixed: coordinates of the fixed fragment, in this case fragment 2
    :param coords_displaced: coordinates of the displaced fragment, in this case fragment 1
    :param atom_identities: append to file the identities of each atom again, since they are lost in processing steps
    """
    with open(file_path, 'w') as f:
        num_atoms = len(coords_fixed) + len(coords_displaced)
        f.write(f'{num_atoms}\n')
        f.write(f'Step {file_path.stem}\n')

        # Write fixed fragment coordinates
        for i, atom in enumerate(coords_fixed):
            f.write(f'{atom_identities[i+len(coords_displaced)]} {atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f}\n') # i+len(coord_fixed) to skip to the fixed fragment indices
        
        # Write displaced fragment coordinates
        for i, atom in enumerate(coords_displaced):
            f.write(f'{atom_identities[i]} {atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f}\n') 
        
#-------------------------------------------------------------------#

# SECTION 7: count fragments for K-means clustering
def count_fragments(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if line.strip().startswith("$fragment"):
                count += 1
        return count

#-------------------------------------------------------------------#

# SECTION 8: COMPUTE DISTANCE OF DISPLACED ATOMS FROM FIXED CENTROID
def compute_distance_from_centroid(coord_displaced, centroids):
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

# SECTION 9: WRITE DISTANCES TO FILES
def write_distances_file(file_path, coords_displaced, distance_to_centroid, atom_identities, displacement_step):
    """
    This function writes the distances between fixed centroid and displaced coordinates to a file.
    :param file_path: where the distance files will be written
    :param coords_displaced: coordinates of the displaced fragment
    :param distance_to_centroid: distance of the displaced coordinates from centroid of the fixed fragment
    :param atom_identities: identities of each atom which are lost in the processing
    :param displacement_step: how many angstroem are we displacing this structures
    """
    with open(file_path, 'w') as f:
        num_atoms = len(coords_displaced)
        f.write(f'Number of atoms: {num_atoms}\n')
        f.write(f'Step {file_path.stem}\n')
        f.write(f'Displacement step: {displacement_step} A\n')
         
        # Write displaced fragment coordinates
        for i, atom in enumerate(coords_displaced):
            f.write(f'{atom_identities[i]} {distance_to_centroid[i]}\n') 


