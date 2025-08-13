"""
Advanced configuration management system for protein operators.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union, Type

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
from pathlib import Path
from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import copy

@dataclass
class ModelConfig:
    """Configuration for neural operator models."""
    operator_type: str = "deeponet"
    checkpoint_path: Optional[str] = None
    device: str = "auto"
    
    # DeepONet specific
    constraint_dim: int = 256
    branch_hidden: List[int] = field(default_factory=lambda: [512, 1024])
    trunk_hidden: List[int] = field(default_factory=lambda: [512, 1024])
    num_basis: int = 1024
    activation: str = "relu"
    dropout_rate: float = 0.1
    
    # FNO specific
    modes1: int = 16
    modes2: int = 16
    modes3: int = 16
    width: int = 64
    depth: int = 4
    in_channels: int = 20
    out_channels: int = 3

@dataclass
class ComputeConfig:
    """Configuration for computational resources."""
    device: str = "auto"
    precision: str = "float32"  # float16, float32, float64
    max_memory_gb: Optional[float] = None
    enable_mixed_precision: bool = False
    num_threads: Optional[int] = None
    batch_size: int = 32
    
    # GPU specific
    gpu_ids: List[int] = field(default_factory=list)
    enable_gpu_memory_growth: bool = True
    gpu_memory_limit_gb: Optional[float] = None

@dataclass
class ValidationConfig:
    """Configuration for structure validation."""
    enable_stereochemistry: bool = True
    enable_clash_detection: bool = True
    enable_ramachandran: bool = True
    enable_constraint_validation: bool = True
    
    # Thresholds
    clash_threshold: float = 2.0
    bond_length_tolerance: float = 0.3
    angle_tolerance: float = 15.0
    ramachandran_outlier_threshold: float = 0.01
    
    # Scoring weights
    stereochemistry_weight: float = 0.25
    clash_weight: float = 0.25
    ramachandran_weight: float = 0.15
    constraint_weight: float = 0.20
    compactness_weight: float = 0.10
    ss_consistency_weight: float = 0.05

@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    log_dir: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console: bool = True
    enable_structured: bool = True
    enable_performance_tracking: bool = True

@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    enable_input_validation: bool = True
    max_protein_length: int = 5000
    max_constraint_count: int = 100
    max_file_size_mb: int = 100
    allowed_file_extensions: List[str] = field(default_factory=lambda: [".pdb", ".fasta", ".json", ".yaml"])
    enable_rate_limiting: bool = False
    max_requests_per_minute: int = 60

@dataclass
class CacheConfig:
    """Configuration for caching system."""
    enable_caching: bool = True
    cache_dir: Optional[str] = None
    max_cache_size_gb: float = 1.0
    cache_ttl_hours: int = 24
    enable_model_cache: bool = True
    enable_structure_cache: bool = True
    enable_validation_cache: bool = True

@dataclass
class ProteinOperatorsConfig:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Metadata
    config_version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    user_id: Optional[str] = None
    description: str = ""

class ConfigurationValidator:
    """Validates configuration settings."""
    
    @staticmethod
    def validate_model_config(config: ModelConfig) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        if config.operator_type not in ["deeponet", "fno"]:
            errors.append(f"Invalid operator_type: {config.operator_type}")
        
        if config.constraint_dim <= 0:
            errors.append("constraint_dim must be positive")
        
        if config.dropout_rate < 0 or config.dropout_rate > 1:
            errors.append("dropout_rate must be between 0 and 1")
        
        if config.activation not in ["relu", "gelu", "swish"]:
            errors.append(f"Invalid activation: {config.activation}")
        
        return errors
    
    @staticmethod
    def validate_compute_config(config: ComputeConfig) -> List[str]:
        """Validate compute configuration."""
        errors = []
        
        if config.precision not in ["float16", "float32", "float64"]:
            errors.append(f"Invalid precision: {config.precision}")
        
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if config.max_memory_gb is not None and config.max_memory_gb <= 0:
            errors.append("max_memory_gb must be positive")
        
        return errors
    
    @staticmethod
    def validate_security_config(config: SecurityConfig) -> List[str]:
        """Validate security configuration."""
        errors = []
        
        if config.max_protein_length <= 0:
            errors.append("max_protein_length must be positive")
        
        if config.max_constraint_count <= 0:
            errors.append("max_constraint_count must be positive")
        
        if config.max_requests_per_minute <= 0:
            errors.append("max_requests_per_minute must be positive")
        
        return errors
    
    @staticmethod
    def validate_full_config(config: ProteinOperatorsConfig) -> List[str]:
        """Validate complete configuration."""
        errors = []
        
        errors.extend(ConfigurationValidator.validate_model_config(config.model))
        errors.extend(ConfigurationValidator.validate_compute_config(config.compute))
        errors.extend(ConfigurationValidator.validate_security_config(config.security))
        
        return errors

class ConfigurationManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self):
        self.config: Optional[ProteinOperatorsConfig] = None
        self.config_file: Optional[Path] = None
        self.logger = logging.getLogger(__name__)
    
    def load_config(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        validate: bool = True
    ) -> ProteinOperatorsConfig:
        """Load configuration from file or environment."""
        
        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            self.config_file = config_path
            
            # Load from file
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML configuration files. Install with: pip install pyyaml")
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            # Convert dict to config object
            self.config = self._dict_to_config(data)
        else:
            # Load from environment or use defaults
            self.config = self._load_from_environment()
        
        # Validate configuration
        if validate:
            errors = ConfigurationValidator.validate_full_config(self.config)
            if errors:
                raise ValueError(f"Configuration validation errors: {'; '.join(errors)}")
        
        self.logger.info(f"Configuration loaded successfully from {config_path or 'environment'}")
        return self.config
    
    def save_config(
        self, 
        config_path: Union[str, Path],
        config: Optional[ProteinOperatorsConfig] = None
    ):
        """Save configuration to file."""
        config = config or self.config
        if not config:
            raise ValueError("No configuration to save")
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dict
        data = self._config_to_dict(config)
        
        # Save based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML configuration files. Install with: pip install pyyaml")
            with open(config_path, 'w') as f:
                yaml.dump(data, f, indent=2, default_flow_style=False)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        self.logger.info(f"Configuration saved to {config_path}")
    
    def get_config(self) -> ProteinOperatorsConfig:
        """Get current configuration."""
        if not self.config:
            self.config = self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        if not self.config:
            self.config = self.load_config()
        
        # Apply updates
        self._update_nested_dict(self._config_to_dict(self.config), updates)
        
        # Convert back to config object
        updated_data = self._config_to_dict(self.config)
        self._apply_updates(updated_data, updates)
        self.config = self._dict_to_config(updated_data)
        
        # Validate updated configuration
        errors = ConfigurationValidator.validate_full_config(self.config)
        if errors:
            raise ValueError(f"Configuration validation errors after update: {'; '.join(errors)}")
    
    def create_default_config(self) -> ProteinOperatorsConfig:
        """Create a default configuration."""
        return ProteinOperatorsConfig()
    
    def export_config_template(self, output_path: Union[str, Path]):
        """Export a configuration template with comments."""
        template = {
            "model": {
                "operator_type": "deeponet",  # Options: deeponet, fno
                "checkpoint_path": None,      # Path to pre-trained model
                "device": "auto",            # auto, cpu, cuda
                "constraint_dim": 256,       # Constraint encoding dimension
                "branch_hidden": [512, 1024], # Hidden layers for branch network
                "trunk_hidden": [512, 1024],  # Hidden layers for trunk network
                "num_basis": 1024,           # Number of basis functions
                "activation": "relu",        # relu, gelu, swish
                "dropout_rate": 0.1          # Dropout probability
            },
            "compute": {
                "device": "auto",            # Computing device
                "precision": "float32",      # Numerical precision
                "batch_size": 32,           # Batch size for processing
                "enable_mixed_precision": False, # Enable automatic mixed precision
                "num_threads": None,         # Number of CPU threads (auto if None)
                "gpu_memory_limit_gb": None  # GPU memory limit
            },
            "validation": {
                "enable_stereochemistry": True,    # Check bond lengths and angles
                "enable_clash_detection": True,    # Check atomic clashes
                "enable_ramachandran": True,       # Check backbone conformations
                "clash_threshold": 2.0,            # Minimum distance for clashes (Å)
                "bond_length_tolerance": 0.3       # Bond length tolerance (Å)
            },
            "logging": {
                "level": "INFO",            # DEBUG, INFO, WARNING, ERROR
                "log_dir": None,           # Directory for log files
                "enable_console": True,    # Enable console logging
                "enable_structured": True  # Enable JSON structured logging
            },
            "security": {
                "max_protein_length": 5000,      # Maximum allowed protein length
                "max_constraint_count": 100,     # Maximum number of constraints
                "enable_input_validation": True,  # Enable input validation
                "max_file_size_mb": 100          # Maximum file size for uploads
            },
            "cache": {
                "enable_caching": True,     # Enable result caching
                "cache_dir": None,         # Cache directory (auto if None)
                "max_cache_size_gb": 1.0,  # Maximum cache size
                "cache_ttl_hours": 24      # Cache time-to-live
            }
        }
        
        output_path = Path(output_path)
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            if not HAS_YAML:
                # Fallback to JSON if YAML not available
                output_path = output_path.with_suffix('.json')
                with open(output_path, 'w') as f:
                    json.dump(template, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    yaml.dump(template, f, indent=2, default_flow_style=False)
        else:
            with open(output_path, 'w') as f:
                json.dump(template, f, indent=2)
        
        self.logger.info(f"Configuration template saved to {output_path}")
    
    def _load_from_environment(self) -> ProteinOperatorsConfig:
        """Load configuration from environment variables."""
        config = ProteinOperatorsConfig()
        
        # Model configuration
        if os.getenv('PROTEIN_OPERATORS_MODEL_TYPE'):
            config.model.operator_type = os.getenv('PROTEIN_OPERATORS_MODEL_TYPE')
        if os.getenv('PROTEIN_OPERATORS_CHECKPOINT'):
            config.model.checkpoint_path = os.getenv('PROTEIN_OPERATORS_CHECKPOINT')
        if os.getenv('PROTEIN_OPERATORS_DEVICE'):
            config.model.device = os.getenv('PROTEIN_OPERATORS_DEVICE')
        
        # Compute configuration
        if os.getenv('PROTEIN_OPERATORS_BATCH_SIZE'):
            config.compute.batch_size = int(os.getenv('PROTEIN_OPERATORS_BATCH_SIZE'))
        if os.getenv('PROTEIN_OPERATORS_PRECISION'):
            config.compute.precision = os.getenv('PROTEIN_OPERATORS_PRECISION')
        
        # Logging configuration
        if os.getenv('PROTEIN_OPERATORS_LOG_LEVEL'):
            config.logging.level = os.getenv('PROTEIN_OPERATORS_LOG_LEVEL')
        if os.getenv('PROTEIN_OPERATORS_LOG_DIR'):
            config.logging.log_dir = os.getenv('PROTEIN_OPERATORS_LOG_DIR')
        
        return config
    
    def _dict_to_config(self, data: Dict[str, Any]) -> ProteinOperatorsConfig:
        """Convert dictionary to configuration object."""
        # Handle nested configuration objects
        model_data = data.get('model', {})
        compute_data = data.get('compute', {})
        validation_data = data.get('validation', {})
        logging_data = data.get('logging', {})
        security_data = data.get('security', {})
        cache_data = data.get('cache', {})
        
        return ProteinOperatorsConfig(
            model=ModelConfig(**model_data),
            compute=ComputeConfig(**compute_data),
            validation=ValidationConfig(**validation_data),
            logging=LoggingConfig(**logging_data),
            security=SecurityConfig(**security_data),
            cache=CacheConfig(**cache_data),
            config_version=data.get('config_version', '1.0'),
            created_at=data.get('created_at', datetime.utcnow().isoformat()),
            user_id=data.get('user_id'),
            description=data.get('description', '')
        )
    
    def _config_to_dict(self, config: ProteinOperatorsConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for field in fields(obj):
                    value = getattr(obj, field.name)
                    if hasattr(value, '__dataclass_fields__'):
                        result[field.name] = dataclass_to_dict(value)
                    else:
                        result[field.name] = value
                return result
            return obj
        
        return dataclass_to_dict(config)
    
    def _update_nested_dict(self, target: Dict[str, Any], updates: Dict[str, Any]):
        """Update nested dictionary with new values."""
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_nested_dict(target[key], value)
            else:
                target[key] = value
    
    def _apply_updates(self, data: Dict[str, Any], updates: Dict[str, Any]):
        """Apply updates to configuration data."""
        for key, value in updates.items():
            keys = key.split('.')
            current = data
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value

# Global configuration manager
_global_config_manager: Optional[ConfigurationManager] = None

def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigurationManager()
    return _global_config_manager

def load_config(config_path: Optional[Union[str, Path]] = None) -> ProteinOperatorsConfig:
    """Load configuration from file or environment."""
    manager = get_config_manager()
    return manager.load_config(config_path)

def get_config() -> ProteinOperatorsConfig:
    """Get current configuration."""
    manager = get_config_manager()
    return manager.get_config()

def save_config(config_path: Union[str, Path], config: Optional[ProteinOperatorsConfig] = None):
    """Save configuration to file."""
    manager = get_config_manager()
    manager.save_config(config_path, config)

def create_config_template(output_path: Union[str, Path]):
    """Create a configuration template file."""
    manager = get_config_manager()
    manager.export_config_template(output_path)