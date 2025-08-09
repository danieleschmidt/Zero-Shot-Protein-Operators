"""
Configuration management system for protein design framework.

This module provides centralized configuration management with validation,
environment-specific settings, and dynamic configuration updates.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List, Type
from pathlib import Path
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import logging
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration sources in order of precedence."""
    ENVIRONMENT = "environment"
    FILE = "file"
    DEFAULT = "default"


@dataclass
class ModelConfig:
    """Configuration for neural operator models."""
    operator_type: str = "deeponet"
    checkpoint_path: Optional[str] = None
    device: str = "auto"
    precision: str = "float32"
    
    # DeepONet specific
    branch_hidden: List[int] = field(default_factory=lambda: [512, 1024])
    trunk_hidden: List[int] = field(default_factory=lambda: [512, 1024])
    num_basis: int = 1024
    
    # FNO specific
    modes: List[int] = field(default_factory=lambda: [16, 16, 16])
    width: int = 64
    depth: int = 4
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class ValidationConfig:
    """Configuration for validation systems."""
    strict_mode: bool = False
    max_constraint_density: float = 0.8
    min_bond_length: float = 1.0
    max_bond_length: float = 8.0
    clash_threshold: float = 2.0
    max_coordinate_range: float = 1000.0
    validation_timeout: float = 30.0


@dataclass
class GenerationConfig:
    """Configuration for protein generation."""
    max_length: int = 500
    min_length: int = 10
    num_samples: int = 1
    physics_guided: bool = True
    optimization_steps: int = 100
    convergence_threshold: float = 1e-6
    temperature: float = 300.0
    force_field: str = "amber99sb"


@dataclass
class ComputeConfig:
    """Configuration for computational resources."""
    max_cpu_cores: int = -1  # -1 means use all available
    max_memory_gb: float = -1.0  # -1 means use system limit
    gpu_enabled: bool = True
    mixed_precision: bool = True
    compile_mode: bool = False
    parallel_workers: int = 4


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    console_output: bool = True


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    enable_input_validation: bool = True
    max_input_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: [".pdb", ".json", ".yaml", ".pt"])
    sandbox_mode: bool = False
    audit_logging: bool = True


@dataclass
class ProteinOperatorConfig:
    """Main configuration class for the protein operator framework."""
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Metadata
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False


class ConfigValidator(ABC):
    """Abstract base class for configuration validators."""
    
    @abstractmethod
    def validate(self, config: Any) -> List[str]:
        """
        Validate configuration object.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        pass


class ModelConfigValidator(ConfigValidator):
    """Validator for model configuration."""
    
    def validate(self, config: ModelConfig) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        # Validate operator type
        valid_operators = ["deeponet", "fno"]
        if config.operator_type not in valid_operators:
            errors.append(f"Invalid operator_type: {config.operator_type}. Must be one of {valid_operators}")
        
        # Validate device
        valid_devices = ["cpu", "cuda", "auto"]
        if config.device not in valid_devices:
            errors.append(f"Invalid device: {config.device}. Must be one of {valid_devices}")
        
        # Validate precision
        valid_precisions = ["float16", "float32", "float64"]
        if config.precision not in valid_precisions:
            errors.append(f"Invalid precision: {config.precision}. Must be one of {valid_precisions}")
        
        # Validate numeric parameters
        if config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if config.max_epochs <= 0:
            errors.append("max_epochs must be positive")
        
        if config.num_basis <= 0:
            errors.append("num_basis must be positive")
        
        # Validate hidden layer sizes
        if not config.branch_hidden or any(h <= 0 for h in config.branch_hidden):
            errors.append("branch_hidden must contain positive integers")
        
        if not config.trunk_hidden or any(h <= 0 for h in config.trunk_hidden):
            errors.append("trunk_hidden must contain positive integers")
        
        return errors


class ValidationConfigValidator(ConfigValidator):
    """Validator for validation configuration."""
    
    def validate(self, config: ValidationConfig) -> List[str]:
        """Validate validation configuration."""
        errors = []
        
        if not 0 < config.max_constraint_density <= 1:
            errors.append("max_constraint_density must be between 0 and 1")
        
        if config.min_bond_length <= 0:
            errors.append("min_bond_length must be positive")
        
        if config.max_bond_length <= config.min_bond_length:
            errors.append("max_bond_length must be greater than min_bond_length")
        
        if config.clash_threshold <= 0:
            errors.append("clash_threshold must be positive")
        
        if config.validation_timeout <= 0:
            errors.append("validation_timeout must be positive")
        
        return errors


class GenerationConfigValidator(ConfigValidator):
    """Validator for generation configuration."""
    
    def validate(self, config: GenerationConfig) -> List[str]:
        """Validate generation configuration."""
        errors = []
        
        if config.min_length <= 0:
            errors.append("min_length must be positive")
        
        if config.max_length <= config.min_length:
            errors.append("max_length must be greater than min_length")
        
        if config.num_samples <= 0:
            errors.append("num_samples must be positive")
        
        if config.optimization_steps < 0:
            errors.append("optimization_steps must be non-negative")
        
        if config.temperature <= 0:
            errors.append("temperature must be positive")
        
        valid_force_fields = ["amber99sb", "charmm36m", "opls"]
        if config.force_field not in valid_force_fields:
            errors.append(f"Invalid force_field: {config.force_field}. Must be one of {valid_force_fields}")
        
        return errors


class ConfigManager:
    """
    Central configuration manager.
    
    Handles loading, validation, and management of configuration from
    multiple sources with proper precedence.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[ProteinOperatorConfig] = None
        self._validators = {
            ModelConfig: ModelConfigValidator(),
            ValidationConfig: ValidationConfigValidator(),
            GenerationConfig: GenerationConfigValidator()
        }
        
        # Environment variable prefix
        self.env_prefix = "PROTEIN_OPERATORS_"
    
    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        validate: bool = True
    ) -> ProteinOperatorConfig:
        """
        Load configuration from multiple sources.
        
        Args:
            config_path: Path to configuration file
            validate: Whether to validate the configuration
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file not found
        """
        if config_path:
            self.config_path = Path(config_path)
        
        # Start with default configuration
        config = ProteinOperatorConfig()
        
        # Load from file if provided
        if self.config_path and self.config_path.exists():
            file_config = self._load_from_file(self.config_path)
            config = self._merge_configs(config, file_config)
            logger.info(f"Loaded configuration from {self.config_path}")
        
        # Override with environment variables
        env_config = self._load_from_environment()
        config = self._merge_configs(config, env_config)
        
        # Validate configuration
        if validate:
            errors = self._validate_config(config)
            if errors:
                raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        self._config = config
        logger.info("Configuration loaded successfully")
        return config
    
    def get_config(self) -> ProteinOperatorConfig:
        """
        Get current configuration.
        
        Returns:
            Current configuration
            
        Raises:
            RuntimeError: If no configuration has been loaded
        """
        if self._config is None:
            # Try to load default configuration
            self._config = self.load_config()
        return self._config
    
    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration (uses current config_path if None)
            
        Raises:
            RuntimeError: If no configuration has been loaded
        """
        if self._config is None:
            raise RuntimeError("No configuration to save")
        
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ValueError("No save path specified")
        
        # Convert to dictionary
        config_dict = asdict(self._config)
        
        # Save as YAML for readability
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> None:
        """
        Update configuration dynamically.
        
        Args:
            updates: Configuration updates as nested dictionary
            validate: Whether to validate after update
            
        Raises:
            ValueError: If configuration becomes invalid after update
        """
        if self._config is None:
            raise RuntimeError("No configuration loaded")
        
        # Apply updates
        self._apply_updates(self._config, updates)
        
        # Validate if requested
        if validate:
            errors = self._validate_config(self._config)
            if errors:
                raise ValueError(f"Configuration update validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        logger.info(f"Configuration updated with {len(updates)} changes")
    
    def _load_from_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f) or {}
                elif path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    logger.warning(f"Unknown configuration file format: {path.suffix}")
                    return {}
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            raise
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Map environment variables to configuration structure
        env_mappings = {
            f"{self.env_prefix}MODEL_OPERATOR_TYPE": ("model", "operator_type"),
            f"{self.env_prefix}MODEL_DEVICE": ("model", "device"),
            f"{self.env_prefix}MODEL_BATCH_SIZE": ("model", "batch_size", int),
            f"{self.env_prefix}MODEL_LEARNING_RATE": ("model", "learning_rate", float),
            f"{self.env_prefix}VALIDATION_STRICT_MODE": ("validation", "strict_mode", bool),
            f"{self.env_prefix}GENERATION_MAX_LENGTH": ("generation", "max_length", int),
            f"{self.env_prefix}COMPUTE_GPU_ENABLED": ("compute", "gpu_enabled", bool),
            f"{self.env_prefix}LOG_LEVEL": ("logging", "level"),
            f"{self.env_prefix}DEBUG": ("debug", bool),
        }
        
        for env_var, mapping in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Parse value type
                if len(mapping) > 2:
                    converter = mapping[2]
                    if converter == bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = converter(value)
                
                # Set nested value
                self._set_nested_value(config, mapping[:-1] if len(mapping) > 2 else mapping, value)
        
        return config
    
    def _merge_configs(
        self,
        base_config: ProteinOperatorConfig,
        updates: Dict[str, Any]
    ) -> ProteinOperatorConfig:
        """Merge configuration updates into base configuration."""
        # Convert base config to dict
        config_dict = asdict(base_config)
        
        # Deep merge updates
        self._deep_merge(config_dict, updates)
        
        # Convert back to config object
        return self._dict_to_config(config_dict)
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep merge updates into base dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, config: Dict[str, Any], keys: tuple, value: Any) -> None:
        """Set nested value in configuration dictionary."""
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ProteinOperatorConfig:
        """Convert dictionary to configuration object."""
        # This is a simplified conversion - in practice, you'd use a more robust method
        try:
            return ProteinOperatorConfig(
                model=ModelConfig(**config_dict.get('model', {})),
                validation=ValidationConfig(**config_dict.get('validation', {})),
                generation=GenerationConfig(**config_dict.get('generation', {})),
                compute=ComputeConfig(**config_dict.get('compute', {})),
                logging=LoggingConfig(**config_dict.get('logging', {})),
                security=SecurityConfig(**config_dict.get('security', {})),
                **{k: v for k, v in config_dict.items() if k not in 
                   ['model', 'validation', 'generation', 'compute', 'logging', 'security']}
            )
        except TypeError as e:
            logger.error(f"Failed to convert dictionary to configuration: {e}")
            logger.error(f"Config dictionary: {config_dict}")
            raise
    
    def _validate_config(self, config: ProteinOperatorConfig) -> List[str]:
        """Validate complete configuration."""
        all_errors = []
        
        # Validate each section
        for attr_name in ['model', 'validation', 'generation']:
            attr_value = getattr(config, attr_name)
            attr_type = type(attr_value)
            
            if attr_type in self._validators:
                errors = self._validators[attr_type].validate(attr_value)
                if errors:
                    all_errors.extend([f"{attr_name}.{error}" for error in errors])
        
        return all_errors
    
    def _apply_updates(self, config: Any, updates: Dict[str, Any]) -> None:
        """Apply updates to configuration object."""
        for key, value in updates.items():
            if hasattr(config, key):
                current_value = getattr(config, key)
                if isinstance(value, dict) and hasattr(current_value, '__dict__'):
                    self._apply_updates(current_value, value)
                else:
                    setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")


# Global configuration manager instance
_global_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    return _global_config_manager


def get_config() -> ProteinOperatorConfig:
    """Get the current global configuration."""
    return _global_config_manager.get_config()


def load_config(config_path: Optional[Union[str, Path]] = None) -> ProteinOperatorConfig:
    """Load configuration from file or environment."""
    return _global_config_manager.load_config(config_path)


def update_config(updates: Dict[str, Any]) -> None:
    """Update global configuration."""
    _global_config_manager.update_config(updates)