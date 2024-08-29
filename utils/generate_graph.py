import os
import sys
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

class MolecularGraph:
    def __init__(self):
        super().__init__('MolFeats')
        self.graph = nx.Graph()

    def add_atom(self, atom_index, atom_type, coordinate):
        self.graph.add_node(atom_index, atom_type=atom_type, coordinate=coordinate[:3])
    
    def add_bond(self, atom1_index, atom2_index, bond_type="covalent"):
        self.graph.add_edge(atom1_index, atom2_index, bond_type=bond_type)

    def draw(self):
        pos = nx.spring_layout(self.graph)  
        labels = nx.get_node_attributes(self.graph, 'atom_type')
        bond_types = nx.get_edge_attributes(self.graph, 'bond_type')
        edge_colors = ["blue" if bond == "noncovalent" else "red" for bond in bond_types.values()]

        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = [labels[node] for node in self.graph.nodes()]

        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color='skyblue',
                size=20,
                line_width=2)
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Molecular Graph',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[dict(
                                text="Molecule from XYZ file",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        fig.show()

    # clear
    def parse_xyz(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
               
        try:
            num_atoms = int(lines[0].strip()) # number of atoms in fragment
            atom_info = lines[2:2+num_atoms] # info on each atom with coordinates in 3D
        except ValueError:
            raise ValueError(f"Error parsing the number of atoms: '{lines[0]}' is not a valid integer.")
        
        atom_types = []
        coordinates = []

        for line in atom_info:
            parts = line.split()
            atom_type = parts[0] # nucleus
            x, y, z = map(float, parts[1:4]) # coordinates
            atom_types.append(atom_type)
            coordinates.append((x, y, z))

        coordinates = np.array(coordinates)
        return atom_types, coordinates

    # clear
    def calculate_distances(self, coordinates):
        num_atoms = len(coordinates)
        distances = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distances[i, j] = distances[j, i] = np.linalg.norm(coordinates[i] - coordinates[j])
        return distances
    
    # clear
    def plot_distance_matrix(self, distances, atom_labels):
        distance_matrix = squareform(pdist(distances, 'euclidean'))
        plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Distance')
        plt.title('Matrix of 2D distances')
        plt.xticks(ticks=np.arange(len(atom_labels)), labels=atom_labels)
        plt.yticks(ticks=np.arange(len(atom_labels)), labels=atom_labels)
        plt.show()

    # clear
    def plot_bond_matrix(self, bonds_matrix, atom_labels):
        plt.imshow(bonds_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Bool')
        plt.title('Matrix of 2D Bonds')
        plt.xticks(ticks=np.arange(len(atom_labels)), labels=atom_labels)
        plt.yticks(ticks=np.arange(len(atom_labels)), labels=atom_labels)
        plt.show()

    # clear
    def plot_bond_dist_matrix(self, bonds_matrix, distances, atom_labels):
        distance_matrix = squareform(pdist(distances, 'euclidean'))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im1 = axes[0].imshow(distance_matrix, cmap='viridis', interpolation='nearest')
        axes[0].set_title('Distance Matrix')
        fig.colorbar(im1, ax=axes[0])
        axes[0].set_xticks(np.arange(len(atom_labels)))
        axes[0].set_xticklabels(atom_labels, rotation=90)
        axes[0].set_yticks(np.arange(len(atom_labels)))
        axes[0].set_yticklabels(atom_labels)

        im2 = axes[1].imshow(bonds_matrix, cmap='gray_r', interpolation='nearest')
        axes[1].set_title('Bonds Matrix')
        fig.colorbar(im2, ax=axes[1])
        axes[1].set_xticks(np.arange(len(atom_labels)))
        axes[1].set_xticklabels(atom_labels, rotation=90)
        axes[1].set_yticks(np.arange(len(atom_labels)))
        axes[1].set_yticklabels(atom_labels)

        axes[2].imshow(distance_matrix, cmap='viridis', interpolation='nearest')
        im3 = axes[2].imshow(bonds_matrix, cmap='gray_r', interpolation='nearest', alpha=0.5)
        axes[2].set_title('Distance vs. Bonds')
        fig.colorbar(im3, ax=axes[2])
        axes[2].set_xticks(np.arange(len(atom_labels)))
        axes[2].set_xticklabels(atom_labels, rotation=90)
        axes[2].set_yticks(np.arange(len(atom_labels)))
        axes[2].set_yticklabels(atom_labels)

        plt.tight_layout()
        plt.show()

    # clear / need new logic
    def detect_bonds(self, atom_types, distances):
        covalent_radii = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66}  # Extend as needed
        bond_matrix = np.zeros(distances.shape, dtype=bool)
        for i, atom1 in enumerate(atom_types):
            for j, atom2 in enumerate(atom_types):
                if i < j:
                    max_bond_dist = covalent_radii[atom1] + covalent_radii[atom2] + 0.4  # tolerance
                    if distances[i, j] < max_bond_dist:
                        bond_matrix[i, j] = bond_matrix[j, i] = True
        return bond_matrix

    # clear / need new logic
    def detect_noncovalent_interactions(self, atom_types, distances):
        noncovalent_interactions = []
        for i, atom1 in enumerate(atom_types):
            for j, atom2 in enumerate(atom_types):
                if i < j:
                    if distances[i, j] > 2.5 and distances[i, j] < 4.0:  # Rough range for non-covalent interaction
                        interaction_type = "hydrogen_bond" if "H" in [atom1, atom2] else "vdW"
                        noncovalent_interactions.append((i, j, interaction_type))
        return noncovalent_interactions
    
    # clear / need new logic
    def plot_noncov_distance_map(self, noncovalent_interactions, atom_labels):
        # Determine the matrix size
        n = max(max(i[0], i[1]) for i in noncovalent_interactions) + 1

        interaction_matrix = np.zeros((n, n), dtype=int)
        vdw_matrix = np.zeros((n, n), dtype=bool)
        hb_matrix = np.zeros((n, n), dtype=bool)

        for i, j, interaction in noncovalent_interactions:
            interaction_matrix[i, j] = 1  # 1 for any interaction
            if interaction == 'vdW':
                vdw_matrix[i, j] = True
            elif interaction == 'hydrogen_bond':
                hb_matrix[i, j] = True

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        im1 = axes[0].imshow(vdw_matrix, cmap='Blues', interpolation='nearest')
        axes[0].set_title('vdW Interactions')
        fig.colorbar(im1, ax=axes[0])
        axes[0].set_xticks(np.arange(len(atom_labels)))
        axes[0].set_xticklabels(atom_labels, rotation=90)
        axes[0].set_yticks(np.arange(len(atom_labels)))
        axes[0].set_yticklabels(atom_labels)

        im2 = axes[1].imshow(hb_matrix, cmap='Reds', interpolation='nearest')
        axes[1].set_title('Hydrogen Bond Interactions')
        fig.colorbar(im2, ax=axes[1])
        axes[1].set_xticks(np.arange(len(atom_labels)))
        axes[1].set_xticklabels(atom_labels, rotation=90)
        axes[1].set_yticks(np.arange(len(atom_labels)))
        axes[1].set_yticklabels(atom_labels)

        axes[2].imshow(vdw_matrix, cmap='Blues', interpolation='nearest')
        im3 = axes[2].imshow(hb_matrix, cmap='Reds', interpolation='nearest', alpha=0.5)
        axes[2].set_title('Full NONCOV interactions')
        fig.colorbar(im3, ax=axes[2])
        axes[2].set_xticks(np.arange(len(atom_labels)))
        axes[2].set_xticklabels(atom_labels, rotation=90)
        axes[2].set_yticks(np.arange(len(atom_labels)))
        axes[2].set_yticklabels(atom_labels)

        plt.tight_layout()
        plt.show()

    def build_molecular_graph(self, atom_types, coordinates, covalent_bonds, noncovalent_interactions):
        mol_graph = MolecularGraph()

        # Add atoms to the graph
        for i, (atom_type, position) in enumerate(zip(atom_types, coordinates)):
            mol_graph.add_atom(i, atom_type, position)

        # Add covalent bonds
        for i in range(len(atom_types)):
            for j in range(i + 1, len(atom_types)):
                if covalent_bonds[i, j]:
                    mol_graph.add_bond(i, j, bond_type="covalent")

        # Add non-covalent interactions
        for i, j, interaction_type in noncovalent_interactions:
            mol_graph.add_bond(i, j, bond_type=interaction_type)

        return mol_graph
    
    def draw_subplots(self, covalent_bonds_graph, intramolecular_graph, intermolecular_graph, coordinates):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Draw Covalent Bonds Graph
        self._draw_graph(covalent_bonds_graph, coordinates, ax=axes[0], title='Covalent Bonds')

        # Draw Intramolecular Contacts Graph
        self._draw_graph(intramolecular_graph, coordinates, ax=axes[1], title='Intramolecular Contacts')

        # Draw Intermolecular Contacts Graph
        self._draw_graph(intermolecular_graph, coordinates, ax=axes[2], title='Intermolecular Contacts')
        
        plt.tight_layout()
        plt.show()

    def _draw_graph(self, graph, coordinates, ax, title):
        # Use the original coordinates as the positions for the nodes
        pos = {i: (coordinates[i][0], coordinates[i][1]) for i in graph.nodes()}  # X, Y coordinates

        labels = nx.get_node_attributes(graph, 'atom_type')
        bond_types = nx.get_edge_attributes(graph, 'bond_type')
        edge_colors = ["blue" if bond == "noncovalent" else "red" for bond in bond_types.values()]

        nx.draw(graph, pos, ax=ax, labels=labels, with_labels=True, node_size=500, node_color="skyblue", edge_color=edge_colors, font_size=8)
        ax.set_title(title)

    def build_covalent_bonds_graph(self, atom_types, coordinates, covalent_bonds):
        graph = nx.Graph()
        for i, (atom_type, position) in enumerate(zip(atom_types, coordinates)):
            graph.add_node(i, atom_type=atom_type, coordinate=position)
        for i in range(len(atom_types)):
            for j in range(i + 1, len(atom_types)):
                if covalent_bonds[i, j]:
                    graph.add_edge(i, j, bond_type="covalent")
        return graph

    def build_intramolecular_graph(self, atom_types, coordinates, covalent_bonds, noncovalent_interactions):
        graph = nx.Graph()
        for i, (atom_type, position) in enumerate(zip(atom_types, coordinates)):
            graph.add_node(i, atom_type=atom_type, coordinate=position)
        for i, j, interaction_type in noncovalent_interactions:
            if covalent_bonds[i, j]:  # Intramolecular if there's a covalent bond
                graph.add_edge(i, j, bond_type="intramolecular")
        return graph

    def build_intermolecular_graph(self, atom_types, coordinates, noncovalent_interactions):
        graph = nx.Graph()
        for i, (atom_type, position) in enumerate(zip(atom_types, coordinates)):
            graph.add_node(i, atom_type=atom_type, coordinate=position)
        for i, j, interaction_type in noncovalent_interactions:
            if interaction_type != "intramolecular":  # Intermolecular if not intramolecular
                graph.add_edge(i, j, bond_type="intermolecular")
        return graph


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