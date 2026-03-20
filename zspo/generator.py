"""ZeroShotGenerator: PDE-conditioned backbone generation."""

import numpy as np

from .backbone import ProteinBackbone
from .neural_operator import NeuralOperatorLayer


class ZeroShotGenerator:
    """Generate protein backbone coordinates from PDE features + random latent.

    Architecture:
      1. Concatenate PDE features (n_residues, n_modes) with latent (n_residues, latent_dim)
         → input (n_residues, n_modes + latent_dim)
      2. Pass through NeuralOperatorLayer → (n_residues, hidden_dim)
      3. Second NeuralOperatorLayer → (n_residues, 4*3) = (n_residues, 12)
      4. Reshape to (n_residues, 4, 3) as backbone coords

    Parameters
    ----------
    n_residues : int
        Number of residues in the generated backbone.
    latent_dim : int
        Dimension of the random latent code per residue.
    hidden_dim : int
        Hidden dimension of the neural operator.
    """

    def __init__(
        self,
        n_residues: int,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        n_modes: int = 8,
        seed=None,
    ):
        self.n_residues = n_residues
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_modes = n_modes

        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**31, size=2)

        # Infer input dim: pde features (n_modes) + latent
        in_dim = n_modes + latent_dim

        self.layer1 = NeuralOperatorLayer(in_dim, hidden_dim, n_modes=n_modes, seed=int(seeds[0]))
        self.layer2 = NeuralOperatorLayer(hidden_dim, 12, n_modes=n_modes, seed=int(seeds[1]))

        self._rng = np.random.default_rng(seed)

    def generate(self, pde_features: np.ndarray, latent: np.ndarray = None) -> ProteinBackbone:
        """Generate a backbone from PDE features.

        Parameters
        ----------
        pde_features : np.ndarray
            Shape (n_residues, n_modes).
        latent : np.ndarray, optional
            Shape (n_residues, latent_dim). Sampled from N(0,1) if None.

        Returns
        -------
        ProteinBackbone
        """
        if latent is None:
            latent = self._rng.standard_normal((self.n_residues, self.latent_dim))

        # Concatenate features
        x = np.concatenate([pde_features, latent], axis=-1)  # (n_residues, n_modes+latent_dim)

        # Forward through two neural operator layers
        h = self.layer1.forward(x)          # (n_residues, hidden_dim)
        out = self.layer2.forward(h)         # (n_residues, 12)

        # Reshape to backbone coordinates
        coords = out.reshape(self.n_residues, 4, 3)

        bb = ProteinBackbone(self.n_residues)
        bb.coords = coords
        return bb
