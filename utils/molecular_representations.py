import os
import sys
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform



class MolRep:
    def __init__(self, xyz_file):
        """
        Initialize the MolRep class with an XYZ file.
        
        Parameters:
        xyz_file (str): Path to the XYZ file containing the molecular structure.
        """
        self.xyz_file = xyz_file
        self.atoms = []
        self.positions = []
        self.bonds = []
        self._parse_xyz()
        self._create_bonds()

    def _parse_xyz(self):
        """
        Internal method to parse the XYZ file and extract atoms and their positions.
        """
        with open(self.xyz_file, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:  # Skip the first two lines (header)
                parts = line.split()
                atom = parts[0]
                x, y, z = map(float, parts[1:4])
                self.atoms.append(atom)
                self.positions.append((x, y, z))
        
        self.positions = np.array(self.positions)
        
    def _create_bonds(self):
        """
        Internal method to create bonds based on simple distance criteria.
        """
        n_atoms = len(self.atoms)
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                distance = np.linalg.norm(self.positions[i] - self.positions[j])
                # Simple bond criterion (you can adjust this threshold)
                if distance < 1.6:  # Consider atoms bonded if within 1.6 Å
                    self.bonds.append((i, j))

    def plot(self):
        """
        Plots the molecule in two subplots:
        - Left: Licorice representation of the molecule.
        - Right: Highlighted nodes and edges.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Define color mapping for different atom types
        color_map = {'H': 'white', 'C': 'black', 'O': 'red', 'N': 'blue', 'S': 'yellow'}

        # Get the colors for the atoms
        colors = [color_map.get(atom, 'gray') for atom in self.atoms]

        # Plot 1: Licorice representation
        axes[0].scatter(self.positions[:, 0], self.positions[:, 1], s=200, c=colors, edgecolor='k')
        for bond in self.bonds:
            i, j = bond
            axes[0].plot([self.positions[i, 0], self.positions[j, 0]],
                         [self.positions[i, 1], self.positions[j, 1]], 'k-', lw=2)
        axes[0].set_title('Licorice Representation')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].axis('equal')

        # Plot 2: Highlighted nodes and edges
        axes[1].scatter(self.positions[:, 0], self.positions[:, 1], s=200, c=colors, edgecolor='k')
        for bond in self.bonds:
            i, j = bond
            axes[1].plot([self.positions[i, 0], self.positions[j, 0]],
                         [self.positions[i, 1], self.positions[j, 1]], 'k-', lw=2)
        axes[1].set_title('Highlighted Molecule Representation')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].axis('equal')

        plt.tight_layout()
        plt.show()


molecule = 'C:/Users/ettor/Desktop/NONCOV/scratch/test_structs/caffeine.xyz'

mol = MolRep(molecule)
mol.plot()

