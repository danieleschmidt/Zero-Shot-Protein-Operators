"""PDEConstraintEncoder: encode folding energy as diffusion PDE features."""

import numpy as np


class PDEConstraintEncoder:
    """Encode a distance matrix as diffusion PDE features.

    Uses discrete Laplacian diffusion on the distance matrix to simulate
    folding energy propagation, then extracts Fourier coefficients as features.

    Parameters
    ----------
    n_modes : int
        Number of Fourier modes (features) to extract per residue.
    diffusion_steps : int
        Number of Laplacian smoothing steps.
    """

    def __init__(self, n_modes: int = 8, diffusion_steps: int = 5):
        self.n_modes = n_modes
        self.diffusion_steps = diffusion_steps

    def encode(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Encode a distance matrix into PDE-diffused features.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Shape (n_residues, n_residues) symmetric distance matrix.

        Returns
        -------
        np.ndarray
            Shape (n_residues, n_modes) feature matrix.
        """
        n = distance_matrix.shape[0]
        x = distance_matrix.copy().astype(np.float64)

        # Discrete diffusion: Laplacian smoothing
        # L = D - A where A is the (normalized) distance-based adjacency
        for _ in range(self.diffusion_steps):
            # Row-normalize to get transition matrix
            row_sums = x.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            a_norm = x / row_sums
            # Laplacian smoothing: x = (I + A_norm) / 2
            x = 0.5 * (x + a_norm @ x)

        # Extract n_modes Fourier coefficients per residue row
        fft_coeffs = np.fft.rfft(x, axis=1)  # (n_residues, n//2 + 1)
        # Take real part of first n_modes coefficients
        n_available = fft_coeffs.shape[1]
        modes = min(self.n_modes, n_available)

        features = np.zeros((n, self.n_modes), dtype=np.float64)
        features[:, :modes] = np.abs(fft_coeffs[:, :modes])

        return features
