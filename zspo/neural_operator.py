"""NeuralOperatorLayer: kernel integral operator approximation via spectral methods."""

import numpy as np


class NeuralOperatorLayer:
    """Spectral neural operator layer.

    Approximates a kernel integral operator using FFT-based spectral convolution
    plus a linear bypass (residual connection).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    n_modes : int
        Number of Fourier modes to retain in spectral convolution.
    """

    def __init__(self, in_dim: int, out_dim: int, n_modes: int = 8, seed=None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes

        rng = np.random.default_rng(seed)

        # Linear bypass weight: (out_dim, in_dim)
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.standard_normal((out_dim, in_dim)) * scale

        # Complex spectral weights: (n_modes, out_dim, in_dim)
        self.R = (
            rng.standard_normal((n_modes, out_dim, in_dim))
            + 1j * rng.standard_normal((n_modes, out_dim, in_dim))
        ) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply neural operator layer.

        Parameters
        ----------
        x : np.ndarray
            Shape (batch_or_n, in_dim).

        Returns
        -------
        np.ndarray
            Shape (batch_or_n, out_dim).
        """
        n = x.shape[0]

        # --- Spectral branch ---
        # FFT along the batch/spatial dimension (axis 0)
        x_fft = np.fft.rfft(x, axis=0)  # (n//2+1, in_dim)

        n_freq = x_fft.shape[0]
        modes = min(self.n_modes, n_freq)

        # Apply complex weights for kept modes: R[m] @ x_fft[m]
        out_fft = np.zeros((n_freq, self.out_dim), dtype=complex)
        for m in range(modes):
            # x_fft[m]: (in_dim,), R[m]: (out_dim, in_dim)
            out_fft[m] = self.R[m] @ x_fft[m]

        # iFFT back to spatial domain
        spectral_out = np.fft.irfft(out_fft, n=n, axis=0)  # (n, out_dim)

        # --- Linear bypass ---
        linear_out = x @ self.W.T  # (n, out_dim)

        return spectral_out + linear_out
