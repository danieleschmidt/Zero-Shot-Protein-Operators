# Zero-Shot Protein Operators API Documentation

## Overview

The Zero-Shot Protein Operators toolkit provides a comprehensive REST API for PDE-constrained protein design using neural operators. This documentation covers all API endpoints, data models, and integration patterns for production deployment.

## Core API Architecture

### Base Configuration
- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication**: JWT tokens or API keys
- **Content-Type**: `application/json`
- **Rate Limiting**: 1000 requests/hour per API key

### Health Endpoints

#### System Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-08-10T14:30:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "cache": "healthy",
    "gpu": "available",
    "workers": 8
  }
}
```

### Protein Design Endpoints

#### Create Protein Design Task
```
POST /protein/design
```

**Request Body:**
```json
{
  "constraints": {
    "binding_sites": [
      {
        "residues": [10, 20, 30],
        "ligand": "ATP",
        "affinity_nm": 100.0,
        "binding_mode": "competitive"
      }
    ],
    "secondary_structures": [
      {
        "start": 5,
        "end": 15,
        "type": "helix",
        "phi_psi_constraints": [-60, -45]
      }
    ],
    "physics_constraints": {
      "max_energy": -10.0,
      "stability_threshold": 0.7,
      "solubility": "high"
    }
  },
  "generation_params": {
    "sequence_length": 100,
    "temperature": 0.8,
    "top_p": 0.9,
    "num_candidates": 5
  },
  "validation_level": "STRICT"
}
```

**Response:**
```json
{
  "task_id": "protein_design_abc123",
  "status": "queued",
  "estimated_completion": "2024-08-10T14:35:00Z",
  "queue_position": 3
}
```

#### Get Design Task Status
```
GET /protein/design/{task_id}
```

**Response (In Progress):**
```json
{
  "task_id": "protein_design_abc123",
  "status": "running",
  "progress": 0.65,
  "stage": "structure_optimization",
  "eta_seconds": 120,
  "intermediate_results": {
    "constraint_satisfaction": 0.85,
    "energy_score": -8.2,
    "stability_prediction": 0.72
  }
}
```

**Response (Completed):**
```json
{
  "task_id": "protein_design_abc123",
  "status": "completed",
  "results": {
    "designs": [
      {
        "design_id": "design_001",
        "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUUBUON",
        "structure": {
          "coordinates": [[1.0, 2.0, 3.0], ...],
          "secondary_structure": "HHHHEEEELLLHHHHEEEE...",
          "confidence_scores": [0.95, 0.87, 0.92, ...]
        },
        "validation": {
          "constraint_satisfaction": 0.92,
          "physics_validation": {
            "energy": -12.5,
            "stability": 0.89,
            "clash_score": 0.02
          },
          "quality_metrics": {
            "ramachandran_favored": 0.98,
            "side_chain_accuracy": 0.85,
            "backbone_accuracy": 0.94
          }
        },
        "metadata": {
          "generation_time": 45.2,
          "neural_operator_used": "DeepONet_v2.1",
          "pde_solver": "finite_element",
          "optimization_steps": 1000
        }
      }
    ],
    "analysis": {
      "success_rate": 0.80,
      "average_energy": -11.2,
      "constraint_violations": [],
      "recommendations": [
        "Consider adding hydrophobic constraints for membrane proteins",
        "Validate binding affinity experimentally"
      ]
    }
  }
}
```

### Constraint Management

#### Validate Constraints
```
POST /constraints/validate
```

**Request Body:**
```json
{
  "constraints": {
    "binding_sites": [...],
    "secondary_structures": [...],
    "physics_constraints": {...}
  },
  "validation_level": "RESEARCH"
}
```

**Response:**
```json
{
  "is_valid": true,
  "validation_report": {
    "structural_validity": {
      "score": 0.95,
      "issues": []
    },
    "physics_consistency": {
      "score": 0.88,
      "warnings": ["High energy binding site at residue 25"]
    },
    "constraint_conflicts": [],
    "suggestions": [
      "Add flexibility constraints for loop regions",
      "Consider allosteric effects for multi-site binding"
    ]
  }
}
```

### Structure Analysis

#### Analyze Existing Structure
```
POST /structure/analyze
```

**Request Body:**
```json
{
  "structure_data": {
    "format": "pdb",
    "content": "ATOM      1  N   MET A   1      20.154  16.967  15.691  1.00 15.00           N\n..."
  },
  "analysis_type": "comprehensive",
  "include_predictions": true
}
```

**Response:**
```json
{
  "structure_id": "struct_xyz789",
  "analysis": {
    "basic_properties": {
      "num_residues": 120,
      "num_chains": 1,
      "molecular_weight": 13245.7,
      "isoelectric_point": 6.8
    },
    "structural_features": {
      "secondary_structure": {
        "alpha_helix": 0.45,
        "beta_sheet": 0.30,
        "loops": 0.25
      },
      "binding_sites": [
        {
          "residues": [45, 67, 89],
          "predicted_ligand": "nucleotide",
          "confidence": 0.87
        }
      ],
      "domains": [
        {
          "start": 1,
          "end": 80,
          "type": "catalytic",
          "confidence": 0.92
        }
      ]
    },
    "quality_assessment": {
      "ramachandran": {
        "favored": 0.96,
        "allowed": 0.03,
        "outliers": 0.01
      },
      "clash_score": 2.3,
      "overall_quality": "high"
    },
    "predictions": {
      "function": "kinase",
      "subcellular_location": "cytoplasm",
      "stability": 0.78,
      "aggregation_propensity": "low"
    }
  }
}
```

## Mathematical Formulations

### Neural Operator Architecture

The system implements DeepONet and Fourier Neural Operators for protein design:

#### DeepONet Formulation
```
G(u)(y) = Σᵢ₌₁ᵖ bᵢ(u) × tᵢ(y)
```
Where:
- `G(u)(y)`: Neural operator mapping input function u to output at point y
- `bᵢ(u)`: Branch network encoding input constraints
- `tᵢ(y)`: Trunk network encoding spatial coordinates
- `p`: Latent dimension

#### PDE Constraint Integration
```
∂u/∂t + ∇ · (D∇u) = f(u, c)
```
Where:
- `u`: Protein structure field
- `D`: Diffusion tensor (structure flexibility)
- `f(u, c)`: Constraint force function
- `c`: Applied constraints (binding sites, secondary structure)

### Optimization Objective

```
L = λ₁L_pde + λ₂L_constraint + λ₃L_physics + λ₄L_quality
```

Where:
- `L_pde`: PDE residual loss
- `L_constraint`: Constraint satisfaction loss  
- `L_physics`: Physical validity loss
- `L_quality`: Structural quality loss
- `λᵢ`: Weighting parameters

## Error Codes

| Code | Description | HTTP Status | Recovery Strategy |
|------|-------------|-------------|-------------------|
| `CONSTRAINT_INVALID` | Invalid constraint specification | 400 | Validate constraints using `/constraints/validate` |
| `COMPUTATION_TIMEOUT` | Design generation timed out | 408 | Reduce complexity or increase timeout |
| `RESOURCE_EXHAUSTED` | Insufficient computational resources | 503 | Wait for resources or upgrade plan |
| `MODEL_ERROR` | Neural operator model failure | 500 | Retry with fallback model |
| `STRUCTURE_INVALID` | Generated structure failed validation | 422 | Adjust constraints or validation level |

## Rate Limits and Quotas

### API Rate Limits
- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1,000 requests/hour  
- **Enterprise**: 10,000 requests/hour

### Computational Quotas
- **GPU Hours**: Tracked per billing period
- **Storage**: 10GB for structures and results
- **Concurrent Tasks**: 3 for Pro, 10 for Enterprise

## Authentication

### API Key Authentication
```bash
curl -H "Authorization: Bearer your-api-key-here" \
     -H "Content-Type: application/json" \
     https://api.proteinoperators.ai/v1/protein/design
```

### JWT Token Authentication  
```bash
# Get token
curl -X POST https://api.proteinoperators.ai/v1/auth/login \
     -d '{"email": "user@example.com", "password": "password"}'

# Use token
curl -H "Authorization: JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
     https://api.proteinoperators.ai/v1/protein/design
```

## SDK Examples

### Python SDK
```python
from protein_operators import ProteinDesignerAPI

# Initialize client
client = ProteinDesignerAPI(api_key="your-key")

# Create constraints
constraints = client.constraints.create()
constraints.add_binding_site(residues=[10, 20], ligand="ATP", affinity_nm=50)
constraints.add_secondary_structure(start=5, end=15, ss_type="helix")

# Submit design task
task = client.protein.design(
    constraints=constraints,
    sequence_length=100,
    num_candidates=3
)

# Monitor progress
while not task.is_complete():
    print(f"Progress: {task.progress:.1%}")
    time.sleep(10)

# Get results
results = task.get_results()
for design in results.designs:
    print(f"Design {design.id}: Energy = {design.energy:.2f}")
```

### JavaScript SDK
```javascript
import { ProteinOperatorsAPI } from '@protein-operators/sdk';

const client = new ProteinOperatorsAPI({
  apiKey: 'your-api-key',
  baseURL: 'https://api.proteinoperators.ai/v1'
});

// Create design task
const task = await client.protein.design({
  constraints: {
    bindingSites: [{
      residues: [10, 20, 30],
      ligand: 'ATP',
      affinityNm: 100
    }]
  },
  generationParams: {
    sequenceLength: 120,
    numCandidates: 5
  }
});

// Poll for completion
const results = await task.waitForCompletion();
console.log(`Generated ${results.designs.length} designs`);
```

## Webhook Integration

### Configure Webhooks
```
POST /webhooks
```

**Request:**
```json
{
  "url": "https://your-app.com/protein-webhook",
  "events": ["task.completed", "task.failed", "task.progress"],
  "secret": "webhook-secret-key"
}
```

### Webhook Payload Example
```json
{
  "event": "task.completed",
  "timestamp": "2024-08-10T14:35:00Z",
  "task_id": "protein_design_abc123",
  "data": {
    "status": "completed",
    "results": {...},
    "metadata": {
      "completion_time": 45.2,
      "resource_usage": {
        "gpu_hours": 0.5,
        "memory_mb": 2048
      }
    }
  }
}
```

## Performance Optimization

### Caching Strategy
- **Structure Cache**: 24-hour TTL for common structures
- **Constraint Validation**: 1-hour TTL for validated constraints
- **Model Predictions**: 6-hour TTL for repeated inputs

### Batch Processing
```json
{
  "batch_requests": [
    {"constraints": {...}, "params": {...}},
    {"constraints": {...}, "params": {...}}
  ],
  "batch_options": {
    "parallel": true,
    "priority": "high"
  }
}
```

## Monitoring and Analytics

### Metrics Endpoint
```
GET /metrics
```

**Response:**
```json
{
  "system_metrics": {
    "gpu_utilization": 0.75,
    "memory_usage": 0.60,
    "active_tasks": 12,
    "queue_depth": 3
  },
  "api_metrics": {
    "requests_per_second": 25.3,
    "average_response_time": 145.2,
    "success_rate": 0.987
  },
  "business_metrics": {
    "designs_generated": 1547,
    "constraint_satisfaction_rate": 0.92,
    "user_satisfaction": 4.7
  }
}
```

---

*For additional support, consult the comprehensive examples in `/examples` or contact support@proteinoperators.ai*