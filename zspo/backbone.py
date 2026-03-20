"""ProteinBackbone: N-CA-C-O coordinate representation."""

import numpy as np


class ProteinBackbone:
    """Protein backbone with N-CA-C-O atom coordinates.

    Attributes
    ----------
    coords : np.ndarray
        Shape (n_residues, 4, 3) — 4 atoms (N, CA, C, O) each with 3D coordinates.
    """

    ATOM_N = 0
    ATOM_CA = 1
    ATOM_C = 2
    ATOM_O = 3

    def __init__(self, n_residues: int):
        """Initialize backbone with zero coordinates."""
        self.n_residues = n_residues
        self.coords = np.zeros((n_residues, 4, 3), dtype=np.float64)

    @classmethod
    def from_random(cls, n_residues: int, seed=None) -> "ProteinBackbone":
        """Create a backbone with random coordinates.

        Parameters
        ----------
        n_residues : int
            Number of residues.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ProteinBackbone
        """
        rng = np.random.default_rng(seed)
        bb = cls(n_residues)
        bb.coords = rng.standard_normal((n_residues, 4, 3))
        return bb

    def bond_lengths(self) -> np.ndarray:
        """Compute CA-CA distances between consecutive residues.

        Returns
        -------
        np.ndarray
            Shape (n_residues - 1,) of CA-CA distances.
        """
        ca = self.coords[:, self.ATOM_CA, :]  # (n_residues, 3)
        diff = ca[1:] - ca[:-1]               # (n_residues-1, 3)
        return np.linalg.norm(diff, axis=-1)

    def to_distance_matrix(self) -> np.ndarray:
        """Compute pairwise CA-CA distance matrix.

        Returns
        -------
        np.ndarray
            Shape (n_residues, n_residues), symmetric.
        """
        ca = self.coords[:, self.ATOM_CA, :]  # (n_residues, 3)
        # Euclidean distance via broadcasting
        diff = ca[:, np.newaxis, :] - ca[np.newaxis, :, :]  # (n, n, 3)
        return np.linalg.norm(diff, axis=-1)
