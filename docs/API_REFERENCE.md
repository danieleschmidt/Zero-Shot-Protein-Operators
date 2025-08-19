# 📚 API Reference Guide

## 🚀 Enhanced DeepONet API

### Class: `EnhancedProteinDeepONet`

Advanced neural operator for protein structure prediction with adaptive capabilities.

#### Constructor

```python
EnhancedProteinDeepONet(
    constraint_dim: int = 256,
    coordinate_dim: int = 3,
    output_dim: int = 3,
    branch_hidden: List[int] = [512, 1024],
    trunk_hidden: List[int] = [512, 1024],
    num_basis: int = 1024,
    activation: str = "gelu",
    dropout_rate: float = 0.1,
    adaptive_basis: bool = True,
    multi_scale_attention: bool = True,
    uncertainty_quantification: bool = True,
    num_ensemble: int = 5,
    physics_regularization: float = 0.1
)
```

#### Methods

##### `forward_with_uncertainty(constraints, coordinates)`

Forward pass with uncertainty quantification.

**Parameters:**
- `constraints` (torch.Tensor): Constraint tensor [batch, num_constraints, 4]
- `coordinates` (torch.Tensor): Coordinate tensor [batch, num_points, 3]

**Returns:**
- `output` (torch.Tensor): Predicted coordinates [batch, num_points, 3]
- `uncertainties` (Dict[str, torch.Tensor]): Uncertainty estimates

**Example:**
```python
model = EnhancedProteinDeepONet()
output, uncertainties = model.forward_with_uncertainty(constraints, coordinates)
print(f"Epistemic uncertainty: {uncertainties['epistemic'].mean():.4f}")
```

##### `get_feature_importance(constraints, coordinates)`

Compute feature importance using gradient-based attribution.

**Returns:**
- `Dict[str, torch.Tensor]`: Importance scores for constraints and coordinates

## 🎯 Advanced Design Service API

### Class: `AdvancedDesignService`

Multi-strategy protein design service with optimization capabilities.

#### Constructor

```python
AdvancedDesignService(
    model: BaseNeuralOperator,
    optimization_strategies: List[str] = ["evolutionary", "gradient"],
    max_concurrent_designs: int = 4,
    cache_size: int = 1000
)
```

#### Methods

##### `design_protein_async(constraints, length, optimization_steps)`

Asynchronous protein design with multiple optimization strategies.

**Parameters:**
- `constraints` (torch.Tensor): Design constraints
- `length` (int): Target protein length
- `optimization_steps` (int): Number of optimization iterations

**Returns:**
- `DesignResult`: Result object with structure, score, and metadata

##### `multi_objective_design(constraints, objectives)`

Multi-objective optimization with Pareto front exploration.

**Parameters:**
- `objectives` (List[Tuple]): List of (name, type, target) objectives

**Example:**
```python
objectives = [
    ("stability", "maximize"),
    ("flexibility", "target", 0.5),
    ("compactness", "minimize")
]
result = designer.multi_objective_design(constraints, objectives)
```

## ✅ Validation Framework API

### Class: `AdvancedValidationFramework`

Comprehensive validation with AI-powered quality assessment.

#### Constructor

```python
AdvancedValidationFramework(
    enable_ai_predictor: bool = True,
    ai_model_path: Optional[str] = None,
    max_concurrent_validations: int = 4
)
```

#### Methods

##### `validate_structure_async(structure, validation_level, structure_id)`

Asynchronous structure validation with multiple levels.

**Parameters:**
- `structure` (ProteinStructure): Structure to validate
- `validation_level` (ValidationLevel): Validation stringency level
- `structure_id` (Optional[str]): Unique identifier

**Returns:**
- `ValidationReport`: Comprehensive validation report

##### `get_validation_statistics()`

Get statistics about completed validations.

**Returns:**
- `Dict[str, Any]`: Validation statistics and metrics

## 📊 Monitoring System API

### Class: `AdvancedMonitoringSystem`

Real-time monitoring and performance profiling.

#### Methods

##### `get_real_time_metrics()`

Get current system metrics.

**Returns:**
- `Dict[str, float]`: Real-time performance metrics

##### `profile(operation_name)`

Context manager for performance profiling.

**Example:**
```python
with monitor.performance_profiler.profile("protein_design"):
    result = designer.design_protein(constraints, length=100)
```

## 🔒 Security Manager API

### Class: `SecurityManager`

Comprehensive security management and access control.

#### Methods

##### `validate_input(data, schema_name)`

Validate input data against predefined schemas.

##### `check_permission(user_id, permission)`

Check if user has required permission.

##### `log_action(user_id, action, details)`

Log security-relevant actions for audit trails.

## 🏗️ Distributed Coordinator API

### Class: `DistributedCoordinator`

Coordinate distributed tasks across multiple nodes.

#### Methods

##### `submit_task_async(task_type, task_data, priority)`

Submit task for distributed execution.

##### `get_node_status()`

Get status of all worker nodes.

##### `balance_load()`

Manually trigger load balancing across nodes.

## 🗄️ Adaptive Cache API

### Class: `AdaptiveCacheSystem`

Multi-level adaptive caching with intelligent preloading.

#### Methods

##### `get(key, level)`

Retrieve cached value from specified cache level.

##### `put(key, value, level, ttl)`

Store value in cache with time-to-live.

##### `get_cache_statistics()`

Get detailed cache performance statistics.

## 🔬 Research Framework API

### Class: `AdvancedResearchFramework`

Scientific experiment design and analysis.

#### Methods

##### `design_experiment(hypothesis, variables, metrics)`

Design scientific experiment with proper controls.

##### `run_experiment(experiment_config)`

Execute experiment with statistical rigor.

##### `generate_report(results)`

Generate publication-ready research report.

## 📡 REST API Endpoints

### Authentication

#### POST `/auth/login`
Authenticate user and receive JWT token.

**Request:**
```json
{
  "username": "researcher@lab.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 86400,
  "user_id": "user123"
}
```

### Protein Design

#### POST `/api/v1/design`
Design new protein structure.

**Headers:**
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request:**
```json
{
  "constraints": [[0, 0, 0, 1], [10, 5, 2, 0.5]],
  "length": 50,
  "optimization_steps": 1000,
  "include_uncertainty": true
}
```

**Response:**
```json
{
  "design_id": "design_12345",
  "structure": {
    "coordinates": [...],
    "sequence": "MKLLAFVGLS..."
  },
  "score": 0.85,
  "uncertainties": {
    "epistemic": 0.12,
    "aleatoric": 0.08
  }
}
```

#### GET `/api/v1/design/{design_id}`
Retrieve design by ID.

#### POST `/api/v1/design/batch`
Submit multiple designs for batch processing.

### Validation

#### POST `/api/v1/validate`
Validate protein structure.

**Request:**
```json
{
  "structure": {
    "coordinates": [...],
    "sequence": "MKLLAFVGLS..."
  },
  "validation_level": "comprehensive"
}
```

**Response:**
```json
{
  "validation_id": "val_67890",
  "overall_score": 0.78,
  "passed": true,
  "metrics": [
    {
      "name": "bond_lengths",
      "value": 0.85,
      "passed": true
    }
  ],
  "recommendations": [
    "Structure passed all validation checks"
  ]
}
```

### Monitoring

#### GET `/api/v1/metrics`
Get system performance metrics.

**Response:**
```json
{
  "cpu_percent": 45.2,
  "memory_percent": 68.5,
  "active_tasks": 12,
  "cache_hit_rate": 0.85
}
```

#### GET `/api/v1/health`
Health check endpoint.

#### GET `/api/v1/ready`
Readiness check endpoint.

## 🔧 Configuration API

### Environment Variables

```python
# Database configuration
POSTGRES_HOST = "postgres-service"
POSTGRES_PORT = 5432
POSTGRES_DB = "protein_operators"

# Cache configuration  
REDIS_HOST = "redis-service"
REDIS_PORT = 6379

# Model configuration
MODEL_PATH = "/models/enhanced_deeponet.pt"
MODEL_BATCH_SIZE = 32

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# Security configuration
JWT_SECRET = "your-secret-key"
RATE_LIMIT_REQUESTS = 1000
RATE_LIMIT_WINDOW = 3600
```

### Configuration Classes

```python
from protein_operators.config import Config

# Load configuration
config = Config.from_env()

# Override specific settings
config.model.batch_size = 64
config.api.workers = 8
```

## 📊 Error Handling

### Exception Classes

```python
class ProteinOperatorError(Exception):
    """Base exception for protein operator errors."""
    pass

class ValidationError(ProteinOperatorError):
    """Raised when validation fails."""
    pass

class DesignError(ProteinOperatorError):
    """Raised when design process fails."""
    pass

class SecurityError(ProteinOperatorError):
    """Raised for security violations."""
    pass
```

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Structure validation failed",
    "details": {
      "failed_metrics": ["bond_lengths", "clashes"],
      "suggestions": ["Run geometry optimization"]
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## 🚀 Performance Guidelines

### Optimization Best Practices

1. **Batch Processing**: Group multiple designs for efficient processing
2. **Caching**: Use cache for frequently accessed structures
3. **Async Operations**: Use async methods for I/O-bound operations
4. **Memory Management**: Monitor memory usage during large batch jobs

### Rate Limiting

- **Default Limits**: 1000 requests per hour per user
- **Burst Capacity**: Up to 100 requests per minute
- **Premium Limits**: Configurable based on subscription tier

### Response Times

- **Design Endpoint**: < 5 seconds for structures up to 100 residues
- **Validation Endpoint**: < 2 seconds for comprehensive validation
- **Metrics Endpoint**: < 100ms for real-time monitoring data

---

*This API reference provides comprehensive documentation for all components of the Enhanced Protein Operators system. For additional examples and tutorials, see the main documentation.*