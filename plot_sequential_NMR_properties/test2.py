import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


# Example usage
xyz_file_path = 'molecule.xyz'  # Replace 'molecule.xyz' with the path to your XYZ file
matrices = {
    'H': np.eye(3) * 0.1,  # Example matrix for hydrogen nucleus
    'C': np.eye(3) * 0.2,  # Example matrix for carbon nucleus
    # Add matrices for other atom types as needed
}
plot_3d_molecule_and_pas_tensor(xyz_file_path)
