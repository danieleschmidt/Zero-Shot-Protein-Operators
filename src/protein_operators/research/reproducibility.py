"""
Reproducibility management for neural operator protein research.

This module provides comprehensive tools for ensuring reproducibility
in neural operator research, including experiment configuration,
result archiving, and environment management.
"""

import os
import sys
import json
import pickle
import hashlib
import time
import shutil
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import platform
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
try:
    import torch
    import numpy as np
except ImportError:
    import mock_torch as torch
    np = None


@dataclass
class ExperimentConfig:
    """
    Comprehensive experiment configuration for reproducibility.
    """
    # Experiment metadata
    experiment_name: str
    description: str
    author: str
    timestamp: str
    
    # Model configuration
    model_type: str
    model_parameters: Dict[str, Any]
    
    # Data configuration
    dataset_name: str
    dataset_version: str
    data_splits: Dict[str, float]
    
    # Training configuration
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int
    loss_function: str
    
    # Hardware configuration
    device: str
    gpu_model: Optional[str]
    memory_gb: Optional[float]
    
    # Software environment
    python_version: str
    pytorch_version: str
    numpy_version: Optional[str]
    cuda_version: Optional[str]
    
    # Random seeds
    random_seed: int
    numpy_seed: int
    torch_seed: int
    
    # Additional configuration
    additional_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    @classmethod
    def from_current_environment(
        cls,
        experiment_name: str,
        description: str,
        author: str,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        **kwargs
    ) -> 'ExperimentConfig':
        """Create configuration from current environment."""
        
        # Get system information
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        
        # Get software versions
        python_version = platform.python_version()
        pytorch_version = torch.__version__
        numpy_version = np.__version__ if np is not None else None
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        
        # Memory information
        memory_gb = None
        try:
            if torch.cuda.is_available():
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
        
        return cls(
            experiment_name=experiment_name,
            description=description,
            author=author,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            model_type=model_config.get('type', 'unknown'),
            model_parameters=model_config,
            dataset_name=training_config.get('dataset', 'unknown'),
            dataset_version=training_config.get('dataset_version', '1.0'),
            data_splits=training_config.get('data_splits', {'train': 0.8, 'val': 0.1, 'test': 0.1}),
            optimizer=training_config.get('optimizer', 'adam'),
            learning_rate=training_config.get('learning_rate', 1e-3),
            batch_size=training_config.get('batch_size', 32),
            epochs=training_config.get('epochs', 100),
            loss_function=training_config.get('loss_function', 'mse'),
            device=device,
            gpu_model=gpu_model,
            memory_gb=memory_gb,
            python_version=python_version,
            pytorch_version=pytorch_version,
            numpy_version=numpy_version,
            cuda_version=cuda_version,
            random_seed=kwargs.get('random_seed', 42),
            numpy_seed=kwargs.get('numpy_seed', 42),
            torch_seed=kwargs.get('torch_seed', 42),
            additional_config=kwargs.get('additional_config', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def get_hash(self) -> str:
        """Get hash of configuration for versioning."""
        # Create reproducible hash
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class ResultsArchiver:
    """
    Archive and manage experimental results for reproducibility.
    """
    
    def __init__(self, base_archive_path: Union[str, Path] = "research_archive"):
        """
        Initialize results archiver.
        
        Args:
            base_archive_path: Base path for archived results
        """
        self.base_path = Path(base_archive_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for archiver."""
        logger = logging.getLogger('ResultsArchiver')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.base_path / 'archive.log')
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(fh)
        
        return logger
    
    def archive_experiment(
        self,
        config: ExperimentConfig,
        results: Dict[str, Any],
        model_state: Optional[Dict[str, Any]] = None,
        source_code: Optional[List[str]] = None,
        additional_files: Optional[List[Union[str, Path]]] = None
    ) -> str:
        """
        Archive complete experiment.
        
        Args:
            config: Experiment configuration
            results: Experiment results
            model_state: Model state dict
            source_code: List of source code files to archive
            additional_files: Additional files to archive
            
        Returns:
            Archive ID
        """
        # Create archive ID
        archive_id = f"{config.experiment_name}_{config.get_hash()}"
        archive_path = self.base_path / archive_id
        archive_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Archiving experiment {archive_id}")
        
        # Save configuration
        config.save(archive_path / "config.json")
        
        # Save results
        with open(archive_path / "results.json", 'w') as f:
            # Convert tensors to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save model state
        if model_state is not None:
            torch.save(model_state, archive_path / "model_state.pth")
        
        # Archive source code
        if source_code is not None:
            source_dir = archive_path / "source_code"
            source_dir.mkdir(exist_ok=True)
            
            for source_file in source_code:
                source_path = Path(source_file)
                if source_path.exists():
                    shutil.copy2(source_path, source_dir / source_path.name)
        
        # Archive additional files
        if additional_files is not None:
            additional_dir = archive_path / "additional_files"
            additional_dir.mkdir(exist_ok=True)
            
            for file_path in additional_files:
                file_path = Path(file_path)
                if file_path.exists():
                    if file_path.is_file():
                        shutil.copy2(file_path, additional_dir / file_path.name)
                    elif file_path.is_dir():
                        shutil.copytree(file_path, additional_dir / file_path.name)
        
        # Create archive manifest
        manifest = {
            'archive_id': archive_id,
            'experiment_name': config.experiment_name,
            'timestamp': config.timestamp,
            'author': config.author,
            'files': [str(f.relative_to(archive_path)) for f in archive_path.rglob('*') if f.is_file()],
            'config_hash': config.get_hash()
        }
        
        with open(archive_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Experiment archived successfully: {archive_id}")
        
        return archive_id
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def load_experiment(self, archive_id: str) -> Tuple[ExperimentConfig, Dict[str, Any]]:
        """
        Load archived experiment.
        
        Args:
            archive_id: Archive identifier
            
        Returns:
            Tuple of (config, results)
        """
        archive_path = self.base_path / archive_id
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive {archive_id} not found")
        
        # Load configuration
        config = ExperimentConfig.load(archive_path / "config.json")
        
        # Load results
        with open(archive_path / "results.json", 'r') as f:
            results = json.load(f)
        
        return config, results
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all archived experiments."""
        experiments = []
        
        for archive_dir in self.base_path.iterdir():
            if archive_dir.is_dir():
                manifest_path = archive_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    experiments.append(manifest)
        
        return experiments
    
    def compare_experiments(
        self,
        archive_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple archived experiments.
        
        Args:
            archive_ids: List of archive IDs to compare
            metrics: Specific metrics to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            'experiments': {},
            'metric_comparison': {},
            'configuration_diff': {}
        }
        
        configs = []
        results = []
        
        for archive_id in archive_ids:
            config, result = self.load_experiment(archive_id)
            configs.append(config)
            results.append(result)
            comparison['experiments'][archive_id] = {
                'config': config.to_dict(),
                'results_summary': self._summarize_results(result)
            }
        
        # Compare metrics
        if metrics is not None:
            for metric in metrics:
                metric_values = []
                for result in results:
                    if metric in result:
                        metric_values.append(result[metric])
                
                if metric_values:
                    comparison['metric_comparison'][metric] = {
                        'values': metric_values,
                        'mean': np.mean(metric_values) if np else None,
                        'std': np.std(metric_values) if np else None,
                        'min': min(metric_values),
                        'max': max(metric_values)
                    }
        
        return comparison
    
    def _summarize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of results."""
        summary = {}
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                summary[key] = value
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                summary[f"{key}_mean"] = np.mean(value) if np else sum(value) / len(value)
                summary[f"{key}_std"] = np.std(value) if np else 0.0
        
        return summary


class ReproducibilityManager:
    """
    Comprehensive reproducibility management system.
    
    Provides tools for ensuring reproducible research in neural
    operator protein design, including seed management, environment
    tracking, and result verification.
    """
    
    def __init__(
        self,
        base_path: Union[str, Path] = "reproducible_research",
        strict_mode: bool = True
    ):
        """
        Initialize reproducibility manager.
        
        Args:
            base_path: Base path for reproducibility files
            strict_mode: Whether to enforce strict reproducibility
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.strict_mode = strict_mode
        
        # Initialize components
        self.archiver = ResultsArchiver(self.base_path / "archives")
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Environment information
        self.environment_info = self._collect_environment_info()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for reproducibility manager."""
        logger = logging.getLogger('ReproducibilityManager')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.base_path / 'reproducibility.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect comprehensive environment information."""
        env_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # PyTorch information
        if torch is not None:
            env_info['pytorch_version'] = torch.__version__
            env_info['cuda_available'] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                env_info['cuda_version'] = torch.version.cuda
                env_info['cudnn_version'] = torch.backends.cudnn.version()
                env_info['gpu_count'] = torch.cuda.device_count()
                env_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        
        # NumPy information
        if np is not None:
            env_info['numpy_version'] = np.__version__
        
        # Git information (if available)
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info['git_commit'] = git_commit
            
            git_status = subprocess.check_output(
                ['git', 'status', '--porcelain'], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info['git_status'] = git_status
            env_info['git_clean'] = len(git_status) == 0
        except:
            env_info['git_available'] = False
        
        return env_info
    
    def set_global_seeds(self, seed: int = 42):
        """Set all random seeds for reproducibility."""
        self.logger.info(f"Setting global seeds to {seed}")
        
        # Python random
        import random
        random.seed(seed)
        
        # NumPy
        if np is not None:
            np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Set deterministic behavior
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        
        if hasattr(torch.backends.cudnn, 'deterministic'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def create_experiment_config(
        self,
        experiment_name: str,
        description: str,
        author: str,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        **kwargs
    ) -> ExperimentConfig:
        """Create experiment configuration."""
        config = ExperimentConfig.from_current_environment(
            experiment_name=experiment_name,
            description=description,
            author=author,
            model_config=model_config,
            training_config=training_config,
            **kwargs
        )
        
        # Validate configuration if in strict mode
        if self.strict_mode:
            self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: ExperimentConfig):
        """Validate experiment configuration."""
        # Check for required fields
        required_fields = ['experiment_name', 'description', 'author']
        for field in required_fields:
            if not getattr(config, field):
                raise ValueError(f"Required field '{field}' is empty")
        
        # Check for reproducibility requirements
        if config.random_seed is None:
            self.logger.warning("No random seed specified")
        
        # Check git status
        if not self.environment_info.get('git_clean', True):
            self.logger.warning("Git repository has uncommitted changes")
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        experiment_function: callable,
        archive_results: bool = True
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Run experiment with full reproducibility tracking.
        
        Args:
            config: Experiment configuration
            experiment_function: Function that runs the experiment
            archive_results: Whether to archive results
            
        Returns:
            Tuple of (results, archive_id)
        """
        self.logger.info(f"Starting experiment: {config.experiment_name}")
        
        # Set seeds
        self.set_global_seeds(config.random_seed)
        
        # Run experiment
        start_time = time.time()
        
        try:
            results = experiment_function(config)
            success = True
            error_msg = None
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            results = {'error': str(e)}
            success = False
            error_msg = str(e)
        
        end_time = time.time()
        
        # Add metadata to results
        results['_metadata'] = {
            'success': success,
            'duration_seconds': end_time - start_time,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
            'environment_info': self.environment_info,
            'config_hash': config.get_hash(),
            'error': error_msg
        }
        
        archive_id = None
        if archive_results:
            archive_id = self.archiver.archive_experiment(config, results)
        
        self.logger.info(f"Experiment completed: {config.experiment_name}")
        
        return results, archive_id
    
    def verify_reproducibility(
        self,
        archive_id: str,
        experiment_function: callable,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Verify reproducibility by re-running archived experiment.
        
        Args:
            archive_id: Archive ID of experiment to verify
            experiment_function: Original experiment function
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Verification results
        """
        self.logger.info(f"Verifying reproducibility for {archive_id}")
        
        # Load original experiment
        original_config, original_results = self.archiver.load_experiment(archive_id)
        
        # Re-run experiment
        new_results, _ = self.run_experiment(
            original_config, experiment_function, archive_results=False
        )
        
        # Compare results
        verification = {
            'reproducible': True,
            'differences': [],
            'numerical_differences': {},
            'verification_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Compare numerical results
        for key in original_results:
            if key.startswith('_'):  # Skip metadata
                continue
                
            if key in new_results:
                orig_val = original_results[key]
                new_val = new_results[key]
                
                if isinstance(orig_val, (int, float)) and isinstance(new_val, (int, float)):
                    diff = abs(orig_val - new_val)
                    if diff > tolerance:
                        verification['reproducible'] = False
                        verification['differences'].append(f"{key}: {orig_val} vs {new_val}")
                        verification['numerical_differences'][key] = diff
                elif orig_val != new_val:
                    verification['reproducible'] = False
                    verification['differences'].append(f"{key}: values differ")
            else:
                verification['reproducible'] = False
                verification['differences'].append(f"Missing key in new results: {key}")
        
        return verification
    
    def generate_reproducibility_report(
        self,
        archive_ids: Optional[List[str]] = None,
        output_file: str = "reproducibility_report.md"
    ):
        """Generate comprehensive reproducibility report."""
        if archive_ids is None:
            experiments = self.archiver.list_experiments()
            archive_ids = [exp['archive_id'] for exp in experiments]
        
        report_lines = []
        report_lines.append("# Reproducibility Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Environment information
        report_lines.append("## Environment Information")
        for key, value in self.environment_info.items():
            report_lines.append(f"- **{key}**: {value}")
        report_lines.append("")
        
        # Experiment summaries
        report_lines.append("## Experiments")
        for archive_id in archive_ids:
            try:
                config, results = self.archiver.load_experiment(archive_id)
                
                report_lines.append(f"### {config.experiment_name}")
                report_lines.append(f"- **Archive ID**: {archive_id}")
                report_lines.append(f"- **Author**: {config.author}")
                report_lines.append(f"- **Timestamp**: {config.timestamp}")
                report_lines.append(f"- **Description**: {config.description}")
                report_lines.append(f"- **Config Hash**: {config.get_hash()}")
                
                # Key results
                if '_metadata' in results and results['_metadata']['success']:
                    report_lines.append("- **Status**: Success")
                    duration = results['_metadata'].get('duration_seconds', 0)
                    report_lines.append(f"- **Duration**: {duration:.2f} seconds")
                else:
                    report_lines.append("- **Status**: Failed")
                
                report_lines.append("")
                
            except Exception as e:
                report_lines.append(f"### Error loading {archive_id}")
                report_lines.append(f"Error: {str(e)}")
                report_lines.append("")
        
        # Write report
        report_path = self.base_path / output_file
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Reproducibility report saved to {report_path}")
    
    def export_experiment_package(
        self,
        archive_id: str,
        output_path: Union[str, Path],
        include_environment: bool = True
    ):
        """
        Export experiment as a complete package for sharing.
        
        Args:
            archive_id: Archive ID to export
            output_path: Output path for package
            include_environment: Whether to include environment setup files
        """
        output_path = Path(output_path)
        archive_path = self.archiver.base_path / archive_id
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive {archive_id} not found")
        
        # Create package
        shutil.copytree(archive_path, output_path)
        
        if include_environment:
            # Add environment files
            env_dir = output_path / "environment"
            env_dir.mkdir(exist_ok=True)
            
            # Save environment info
            with open(env_dir / "environment_info.json", 'w') as f:
                json.dump(self.environment_info, f, indent=2)
            
            # Create requirements.txt if possible
            try:
                requirements = subprocess.check_output(['pip', 'freeze']).decode()
                with open(env_dir / "requirements.txt", 'w') as f:
                    f.write(requirements)
            except:
                pass
        
        self.logger.info(f"Experiment package exported to {output_path}")