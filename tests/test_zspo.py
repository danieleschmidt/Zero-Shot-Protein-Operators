"""Tests for Zero-Shot Protein Operators."""

import numpy as np
import pytest

from zspo.backbone import ProteinBackbone
from zspo.pde_encoder import PDEConstraintEncoder
from zspo.neural_operator import NeuralOperatorLayer
from zspo.generator import ZeroShotGenerator
from zspo.evaluator import rmsd, gdt_ts


# ---------------------------------------------------------------------------
# ProteinBackbone tests
# ---------------------------------------------------------------------------

def test_backbone_shape():
    """ProteinBackbone coords should have shape (n_residues, 4, 3)."""
    n = 20
    bb = ProteinBackbone(n)
    assert bb.coords.shape == (n, 4, 3)


def test_backbone_from_random_valid():
    """from_random should produce finite coordinates."""
    bb = ProteinBackbone.from_random(30, seed=42)
    assert bb.coords.shape == (30, 4, 3)
    assert np.all(np.isfinite(bb.coords))


def test_backbone_from_random_seed_reproducible():
    """Same seed should give identical coordinates."""
    bb1 = ProteinBackbone.from_random(20, seed=7)
    bb2 = ProteinBackbone.from_random(20, seed=7)
    np.testing.assert_array_equal(bb1.coords, bb2.coords)


def test_bond_lengths_count():
    """bond_lengths should return n_residues - 1 values."""
    n = 15
    bb = ProteinBackbone.from_random(n, seed=1)
    bl = bb.bond_lengths()
    assert bl.shape == (n - 1,)


def test_bond_lengths_positive():
    """All bond lengths should be non-negative."""
    bb = ProteinBackbone.from_random(25, seed=2)
    assert np.all(bb.bond_lengths() >= 0)


def test_distance_matrix_symmetric():
    """to_distance_matrix should return a symmetric matrix."""
    bb = ProteinBackbone.from_random(20, seed=3)
    dm = bb.to_distance_matrix()
    assert dm.shape == (20, 20)
    np.testing.assert_allclose(dm, dm.T, atol=1e-10)


def test_distance_matrix_diagonal_zero():
    """Diagonal of distance matrix should be zero."""
    bb = ProteinBackbone.from_random(10, seed=4)
    dm = bb.to_distance_matrix()
    np.testing.assert_allclose(np.diag(dm), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# PDEConstraintEncoder tests
# ---------------------------------------------------------------------------

def test_pde_encoder_output_shape():
    """PDEConstraintEncoder.encode should return (n_residues, n_modes)."""
    n, n_modes = 20, 8
    bb = ProteinBackbone.from_random(n, seed=5)
    dm = bb.to_distance_matrix()
    enc = PDEConstraintEncoder(n_modes=n_modes)
    feats = enc.encode(dm)
    assert feats.shape == (n, n_modes)


def test_pde_encoder_finite():
    """PDE features should be finite."""
    bb = ProteinBackbone.from_random(30, seed=6)
    dm = bb.to_distance_matrix()
    enc = PDEConstraintEncoder()
    feats = enc.encode(dm)
    assert np.all(np.isfinite(feats))


# ---------------------------------------------------------------------------
# NeuralOperatorLayer tests
# ---------------------------------------------------------------------------

def test_neural_operator_preserves_batch_dim():
    """NeuralOperatorLayer should preserve the batch/spatial dimension."""
    layer = NeuralOperatorLayer(in_dim=16, out_dim=16, n_modes=8, seed=0)
    x = np.random.randn(10, 16)
    out = layer.forward(x)
    assert out.shape[0] == 10


def test_neural_operator_changes_dimension():
    """NeuralOperatorLayer should transform in_dim → out_dim."""
    layer = NeuralOperatorLayer(in_dim=16, out_dim=32, n_modes=8, seed=0)
    x = np.random.randn(10, 16)
    out = layer.forward(x)
    assert out.shape == (10, 32)


def test_neural_operator_output_finite():
    """NeuralOperatorLayer output should be finite."""
    layer = NeuralOperatorLayer(in_dim=8, out_dim=12, n_modes=4, seed=1)
    x = np.random.randn(20, 8)
    out = layer.forward(x)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# ZeroShotGenerator tests
# ---------------------------------------------------------------------------

def test_generator_returns_protein_backbone():
    """ZeroShotGenerator.generate should return a ProteinBackbone."""
    n = 20
    bb = ProteinBackbone.from_random(n, seed=10)
    enc = PDEConstraintEncoder(n_modes=8)
    feats = enc.encode(bb.to_distance_matrix())
    gen = ZeroShotGenerator(n_residues=n, latent_dim=16, hidden_dim=32, seed=42)
    result = gen.generate(feats)
    assert isinstance(result, ProteinBackbone)
    assert result.coords.shape == (n, 4, 3)


def test_generator_with_explicit_latent_is_deterministic():
    """Passing the same explicit latent should yield identical outputs."""
    n = 20
    bb = ProteinBackbone.from_random(n, seed=11)
    enc = PDEConstraintEncoder(n_modes=8)
    feats = enc.encode(bb.to_distance_matrix())
    gen = ZeroShotGenerator(n_residues=n, latent_dim=16, hidden_dim=32, seed=42)
    latent = np.random.default_rng(99).standard_normal((n, 16))
    r1 = gen.generate(feats, latent=latent)
    r2 = gen.generate(feats, latent=latent)
    np.testing.assert_array_equal(r1.coords, r2.coords)


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

def test_rmsd_identical_is_zero():
    """RMSD of a structure against itself should be 0."""
    coords = np.random.randn(30, 3)
    assert rmsd(coords, coords) == pytest.approx(0.0, abs=1e-10)


def test_rmsd_different_structures_positive():
    """RMSD of two different structures should be > 0."""
    rng = np.random.default_rng(20)
    c1 = rng.standard_normal((30, 3))
    c2 = rng.standard_normal((30, 3))
    assert rmsd(c1, c2) > 0.0


def test_gdt_ts_identical_is_one():
    """GDT_TS of a structure against itself should be 1.0."""
    coords = np.random.randn(30, 3)
    assert gdt_ts(coords, coords) == pytest.approx(1.0)


def test_gdt_ts_range():
    """GDT_TS should be in [0, 1]."""
    rng = np.random.default_rng(21)
    c1 = rng.standard_normal((30, 3))
    c2 = rng.standard_normal((30, 3))
    score = gdt_ts(c1, c2)
    assert 0.0 <= score <= 1.0
