# Protein Operators Experiments

This directory contains experimental notebooks, scripts, and results for protein operator research.

## Structure

```
experiments/
├── notebooks/           # Jupyter notebooks for exploration
├── configs/            # Experiment configuration files  
├── results/            # Experimental results and outputs
├── baselines/          # Baseline method comparisons
└── analysis/           # Analysis scripts and visualizations
```

## Quick Start

1. **Setup Environment**:
   ```bash
   conda env create -f ../environment.yml
   conda activate protein-operators
   ```

2. **Start Jupyter Lab**:
   ```bash
   jupyter lab
   ```

3. **Run Basic Experiment**:
   ```bash
   python baselines/rosetta_comparison.py
   ```

## Experiment Types

### Neural Operator Training
- `train_deeponet.py` - Train DeepONet models
- `train_fno.py` - Train FNO models  
- `hyperparameter_search.py` - Automated hyperparameter optimization

### Validation Studies
- `physics_validation.py` - Physics consistency checks
- `benchmark_comparison.py` - Compare against existing tools
- `ablation_studies.py` - Component analysis

### Application Examples
- `antibody_design.py` - Antibody design pipeline
- `enzyme_engineering.py` - Enzyme optimization
- `de_novo_design.py` - Novel protein generation

## Configuration

Experiments use Hydra for configuration management. See `configs/` for examples:

```yaml
# configs/train_deeponet.yaml
model:
  type: deeponet
  constraint_dim: 256
  branch_hidden: [512, 1024]
  trunk_hidden: [512, 1024]
  num_basis: 1024

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  physics_loss_weight: 0.1

data:
  dataset: pdb_select95
  max_length: 300
  augmentation: true
```

## Reproducibility

All experiments include:
- Configuration files
- Random seeds
- Environment specifications
- Result logging with MLflow
- Model checkpoints

## Contributing

1. Create new experiment in appropriate subdirectory
2. Use consistent naming: `YYYY-MM-DD_experiment_name`
3. Include configuration file
4. Document results in experiment log
5. Add summary to this README