# Zero-Shot Protein Operators

Neural operators for de novo protein structure generation conditioned on PDE constraints.

## Features
- ProteinBackbone: N-CA-C-O coordinate representation
- PDEConstraintEncoder: Folding energy encoded as diffusion PDE features
- NeuralOperatorLayer: Kernel integral operator approximation
- ZeroShotGenerator: PDE-conditioned structure generation
- RMSD / GDT_TS evaluation metrics

## Install
```
pip install numpy
```

## Usage
```python
from zspo.backbone import ProteinBackbone
from zspo.generator import ZeroShotGenerator
from zspo.pde_encoder import PDEConstraintEncoder
from zspo.evaluator import rmsd

bb = ProteinBackbone.from_random(50)
enc = PDEConstraintEncoder()
feats = enc.encode(bb.to_distance_matrix())
gen = ZeroShotGenerator(n_residues=50)
pred = gen.generate(feats)
print(rmsd(bb.coords[:, 1], pred.coords[:, 1]))
```

## Run Tests
```
~/anaconda3/bin/python3 -m pytest tests/ -v
```
