import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Function to calculate the normal vector using the cross product of vectors with same origin formed by 3 points
def get_norm_arom_plane(arom_C_coords, moving_frag_centroid, tolerance=1e-1):
    if len(arom_C_coords) < 3:
        raise ValueError("At least 3 points are required to define a plane.")
    
    # Calculate the centroid
    C6_centroid = calculate_centroid(arom_C_coords)
    
    # Select any 3 non-collinear points from the ring
    vec1 = arom_C_coords[1] - arom_C_coords[0]
    vec2 = arom_C_coords[3] - arom_C_coords[0]
    
    # Compute the normal using the cross product
    normal_dir = np.cross(vec1, vec2)
    
    # Normalize the normal vector
    normal_dir /= np.linalg.norm(normal_dir)
    
    # Vector from ring centroid to moving fragment centroid
    vector_to_moving_frag = moving_frag_centroid - C6_centroid
    
    # Distance between the moving fragment centroid and the center of the aromatic ring
    distance_centroid_aromatics = np.linalg.norm(moving_frag_centroid - C6_centroid)
    distance_centroid_aromatics = distance_centroid_aromatics.round(2)
    print(f'Distance Centroid to Aromatic is: {distance_centroid_aromatics} Angstroms\n')
    
    # Check the direction of the normal vector
    if np.dot(normal_dir, vector_to_moving_frag) < 0:
        normal_dir = -normal_dir
    
    return normal_dir, C6_centroid

# Function to calculate the centroid (middle point) of the plane formed by points
def calculate_centroid(coords):
    return np.mean(coords, axis=0)

# Function to displace all fragment coordinates along a normal direction
def displace_fragment_along_normal(fragment_coords, normal_dir, distance_step, k):
    displaced_fragments = []
    for i in range(k):
        # Displace all fragment 2 atoms by the same step
        displaced_fragment = fragment_coords + normal_dir * distance_step * (i + 1)
        displaced_fragments.append(displaced_fragment)
        print(f"Iteration {i + 1}, Fragment 2 Updated Coordinates:\n {displaced_fragment}\n")
    return np.array(displaced_fragments)

# Plot the molecule, plane, and normal vectors
def plot_molecule_and_plane(arom_C_coord, normal_dir, C6_centroid, fragment_2_coords, displaced_fragments):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot fragment 1 aromatic carbons
    ax.scatter(arom_C_coord[:, 0], arom_C_coord[:, 1], arom_C_coord[:, 2], color='grey', s=75, label='Fragment 1 Aromatics')
    
    # Plot the plane
    point = C6_centroid
    d = -point.dot(normal_dir)
    
    # Create a grid to represent the plane
    xx, yy = np.meshgrid(np.linspace(min(arom_C_coord[:, 0]), max(arom_C_coord[:, 0]), 10),
                         np.linspace(min(arom_C_coord[:, 1]), max(arom_C_coord[:, 1]), 10))
    zz = (-normal_dir[0] * xx - normal_dir[1] * yy - d) * 1. / normal_dir[2]
    
    ax.plot_surface(xx, yy, zz, color='grey', alpha=0.3, rstride=100, cstride=100)

    # Plot the normal vector
    ax.quiver(C6_centroid[0], C6_centroid[1], C6_centroid[2], 
              normal_dir[0], normal_dir[1], normal_dir[2], length=1.0, color='grey', label='Normal Vector')
    
    # Plot initial fragment 2 coordinates
    ax.scatter(fragment_2_coords[:, 0], fragment_2_coords[:, 1], fragment_2_coords[:, 2], color='blue', s=75, label='Fragment 2 Initial')

    # Plot the displaced fragment coordinates
    for i, displaced_fragment in enumerate(displaced_fragments):
        ax.scatter(displaced_fragment[:, 0], displaced_fragment[:, 1], displaced_fragment[:, 2], color='orange', s=50, label=f'Displaced Fragment Iter {i+1}' if i == 0 else None)
    
    # Plot centroid of the aromatic ring
    ax.scatter(C6_centroid[0], C6_centroid[1], C6_centroid[2], color='k', s=50, label='Aromatic Centroid')
    
    # Set plot labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # Add legend and show plot
    ax.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Coordinates for fragment 1 (aromatic carbons)
    fragment_1_coords = np.array([
        [-0.51724, 0.54821, 0.35582],  # Aromatic carbon positions
        [-1.15680, 1.08010, -0.76605],
        [-2.54090, 1.20110, -0.78853],
        [-3.29528, 0.79560, 0.30810],
        [-2.66210, 0.26887, 1.42834],
        [-1.27710, 0.14437, 1.45429]
    ])
    
    # Coordinates for fragment 2 (nuclei to be moved)
    fragment_2_coords = np.array(
        [[-2.25242346748428,     -2.84526244677385,     -1.00005731324906],
        [-2.11949654452286,     -1.44057901776393,     -0.43482100788833],
        [-0.93370642353954,     -1.06116011344179,     -0.21623756572648],
        [-3.20987427392875,     -0.81901542862293,     -0.29331344723916],
        [-1.42803444607753,     -3.49211652457309,     -0.67974363142805],
        [-3.21304637368794,     -3.30147786994342,     -0.73640404204677],
        [-2.21060930992580,     -2.77985916125602,     -2.09531685663904]]
    )

    # Calculate the centroid of fragment 2
    moving_frag_centroid = calculate_centroid(fragment_2_coords)
    
    try:
        # Calculate the normal vectors and centroid of the plane for fragment 1 (aromatic carbons)
        normal_dir, C6_centroid = get_norm_arom_plane(fragment_1_coords, moving_frag_centroid)
        
        # Output the normal vector direction
        print(f"Normal Vector: {normal_dir}")
        print(f"Centroid of the Plane: {C6_centroid}")
        
        # Displace the fragment 2 coordinates along the normal direction
        distance_step = 0.25  # Define the step size for displacement
        k = 20  # Number of displacement steps

        displaced_fragments = displace_fragment_along_normal(fragment_2_coords, normal_dir, distance_step, k)
        
        # Plot the two fragments and displaced positions
        plot_molecule_and_plane(fragment_1_coords, normal_dir, C6_centroid, fragment_2_coords, displaced_fragments)
        
        # Print the final coordinates of fragment 2
        final_fragment_coords = displaced_fragments[-1]
        print(f"Final Fragment 2 Coordinates after {k} iterations:\n{final_fragment_coords}")
        
    except ValueError as e:
        print(e)
