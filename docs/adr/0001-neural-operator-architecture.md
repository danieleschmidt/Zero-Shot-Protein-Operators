# ADR-0001: Neural Operator Architecture Selection

## Status
Accepted

Date: 2025-01-15

## Context
We need to select the core neural architecture for learning protein structure mappings from biophysical constraints. The system must handle:

- Variable-length constraint specifications
- High-dimensional 3D coordinate outputs
- Physics-informed learning requirements
- Zero-shot generalization to new protein designs
- Computational efficiency for practical deployment

Key alternatives considered:
1. **Transformer-based architectures** (ESM, ProtTrans)
2. **Graph Neural Networks** (GCN, GAT, MessagePassing)
3. **Neural Operators** (DeepONet, FNO, Neural ODE)
4. **Diffusion Models** (DDPM, Score-based)

## Decision
We will implement a **hybrid neural operator architecture** combining:
- **DeepONet** for constraint-to-structure mapping
- **Fourier Neural Operator (FNO)** for PDE-constrained refinement
- **Multi-scale integration** connecting different resolution levels

## Rationale

### Why Neural Operators?
1. **Function-to-Function Learning**: Natural fit for constraint → structure mapping
2. **Resolution Independence**: Can operate at different spatial scales
3. **Physics Integration**: Built-in PDE solving capabilities
4. **Zero-Shot Generalization**: Learn operators, not specific instances

### Why DeepONet + FNO Hybrid?
1. **DeepONet Strengths**:
   - Excellent for constraint encoding (branch network)
   - Handles variable constraint types naturally
   - Strong theoretical foundations for operator learning

2. **FNO Strengths**:
   - Global receptive fields via spectral convolutions
   - Efficient for PDE solving
   - Multi-resolution processing capabilities

3. **Synergistic Combination**:
   - DeepONet handles constraint → initial structure
   - FNO refines via physics-informed PDE solving
   - Multi-scale bridging connects resolutions

### Alternatives Rejected

**Transformers**: 
- Pros: Strong sequence modeling, pre-trained models available
- Cons: Limited physics integration, quadratic scaling, not designed for operator learning

**Graph Neural Networks**:
- Pros: Natural protein graph representation, message passing
- Cons: Fixed topology assumptions, limited long-range interactions, no direct PDE solving

**Diffusion Models**:
- Pros: High-quality generation, controllable sampling
- Cons: Computationally expensive, limited physics constraints, iterative sampling

## Consequences

### Positive
- **Unified Framework**: Single architecture handles multiple scales and physics
- **Strong Theoretical Foundation**: Operator learning theory provides guarantees
- **Computational Efficiency**: Spectral methods scale well with problem size
- **Physics Integration**: Natural PDE constraint handling
- **Extensibility**: Easy to add new constraint types and physics models

### Negative
- **Implementation Complexity**: More complex than standard architectures
- **Training Data Requirements**: Need diverse PDE solution data
- **Memory Usage**: Spectral operations can be memory-intensive
- **Limited Pre-trained Models**: Must train from scratch

### Neutral
- **Novel Architecture**: Less community support but cutting-edge capabilities
- **Research vs Production**: Balance between innovation and stability

## Implementation Notes

### Core Components
1. **Constraint Branch Network**:
   ```python
   constraint_encoder = nn.Sequential(
       ConstraintEmbedding(dim=256),
       nn.Linear(256, 512),
       nn.ReLU(),
       nn.Linear(512, 1024)
   )
   ```

2. **Spatial Trunk Network**:
   ```python
   spatial_encoder = nn.Sequential(
       PositionalEncoding(dim=128),
       nn.Linear(128, 512),
       nn.ReLU(),
       nn.Linear(512, 1024)
   )
   ```

3. **FNO Refiner**:
   ```python
   fno_refiner = FNO2d(
       modes1=32, modes2=32,
       width=64, depth=4,
       in_channels=3, out_channels=3
   )
   ```

### Training Strategy
- **Phase 1**: Train DeepONet on constraint-structure pairs
- **Phase 2**: Train FNO on PDE trajectory data
- **Phase 3**: Joint training with physics-informed losses
- **Phase 4**: Multi-scale integration optimization

### Validation Approach
- **Unit Tests**: Individual operator components
- **Physics Tests**: Energy conservation, force consistency
- **Benchmark Tests**: Comparison with Rosetta, AlphaFold
- **Ablation Studies**: Component contribution analysis

## Related Decisions
- ADR-0002: PDE Formulation for Protein Folding
- ADR-0003: Multi-Scale Architecture Design
- ADR-0004: Training Data Pipeline

## References
- Lu et al. "Learning nonlinear operators via DeepONet" (2021)
- Li et al. "Fourier Neural Operator for Parametric PDEs" (2020)
- Chen et al. "Universal approximation to nonlinear operators" (1995)
- Protein Data Bank structure database
- OpenMM molecular dynamics framework