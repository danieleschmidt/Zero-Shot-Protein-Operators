"""Structure evaluation metrics: RMSD and GDT_TS."""

import numpy as np


def rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two sets of CA coordinates.

    Parameters
    ----------
    coords1 : np.ndarray
        Shape (n_residues, 3) — CA coordinates of structure 1.
    coords2 : np.ndarray
        Shape (n_residues, 3) — CA coordinates of structure 2.

    Returns
    -------
    float
        Root mean square deviation.
    """
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1))))


def gdt_ts(
    coords1: np.ndarray,
    coords2: np.ndarray,
    thresholds: tuple = (1.0, 2.0, 4.0, 8.0),
) -> float:
    """Compute GDT_TS (Global Distance Test Total Score).

    Parameters
    ----------
    coords1 : np.ndarray
        Shape (n_residues, 3) — CA coordinates of structure 1 (prediction).
    coords2 : np.ndarray
        Shape (n_residues, 3) — CA coordinates of structure 2 (reference).
    thresholds : tuple
        Distance cutoffs in Angstroms. Default: (1.0, 2.0, 4.0, 8.0).

    Returns
    -------
    float
        GDT_TS score in [0, 1].
    """
    distances = np.linalg.norm(coords1 - coords2, axis=-1)  # (n_residues,)
    n = len(distances)
    scores = []
    for t in thresholds:
        fraction = np.sum(distances <= t) / n
        scores.append(fraction)
    return float(np.mean(scores))
