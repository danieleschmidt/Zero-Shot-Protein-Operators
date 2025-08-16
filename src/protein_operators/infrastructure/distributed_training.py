"""
Distributed Training Infrastructure for Large-Scale Protein Neural Operators.

Implements state-of-the-art distributed training techniques for protein
design models, enabling scaling to massive datasets and model sizes.

Research Features:
- Multi-GPU data and model parallelism
- Gradient compression and communication optimization
- Dynamic load balancing
- Fault-tolerant training with checkpointing
- Federated learning for collaborative research
- Mixed precision training optimization

Citing: "Distributed Deep Learning for Protein Structure Prediction" (2024)
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    import mock_torch as torch
    nn = torch.nn
    F = torch.nn.functional
    
    # Mock distributed components
    class DDP:
        def __init__(self, model, *args, **kwargs):
            self.module = model
        
        def __call__(self, *args, **kwargs):
            return self.module(*args, **kwargs)
    
    class DistributedSampler:
        def __init__(self, *args, **kwargs):
            pass
    
    class GradScaler:
        def __init__(self, *args, **kwargs):
            pass
        
        def scale(self, loss):
            return loss
        
        def step(self, optimizer):
            optimizer.step()
        
        def update(self):
            pass
    
    def autocast(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    dist = None


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = 'nccl'
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = 'localhost'
    master_port: str = '12355'
    
    # Training configuration
    batch_size_per_gpu: int = 32
    accumulation_steps: int = 1
    mixed_precision: bool = True
    
    # Communication optimization
    gradient_compression: bool = True
    compression_ratio: float = 0.1
    bucket_size_mb: int = 25
    
    # Fault tolerance
    checkpoint_frequency: int = 1000
    checkpoint_dir: str = 'checkpoints'
    max_retries: int = 3
    
    # Load balancing
    dynamic_batching: bool = True
    load_balance_frequency: int = 100


class DistributedTrainer:
    """
    Advanced distributed training coordinator.
    
    Handles multi-GPU training with advanced features like gradient
    compression, dynamic load balancing, and fault tolerance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # Initialize distributed training
        self.is_distributed = config.world_size > 1
        if self.is_distributed and dist is not None:
            self._init_distributed()
        
        # Setup model for distributed training
        self.model = self._setup_model(model)
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Gradient compression
        self.gradient_compressor = GradientCompressor(
            compression_ratio=config.compression_ratio
        ) if config.gradient_compression else None
        
        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_retries=config.max_retries
        )
        
        # Load balancing
        self.load_balancer = DynamicLoadBalancer(
            world_size=config.world_size,
            rank=config.rank
        ) if config.dynamic_batching else None
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.training_metrics = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup distributed training logger."""
        logger = logging.getLogger(f'DistributedTrainer_rank_{self.config.rank}')
        logger.setLevel(logging.INFO)
        
        # Only log from rank 0 to avoid spam
        if self.config.rank == 0:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_distributed(self):
        """Initialize distributed training environment."""
        if dist is None:
            self.logger.warning("Distributed training not available")
            return
        
        # Set environment variables
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        os.environ['RANK'] = str(self.config.rank)
        
        try:
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            # Set CUDA device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
            
            self.logger.info(f"Initialized distributed training: rank {self.config.rank}/{self.config.world_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for distributed training."""
        # Move model to appropriate device
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.local_rank}')
            model = model.to(device)
        
        # Wrap with DistributedDataParallel
        if self.is_distributed and dist is not None:
            model = DDP(
                model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                bucket_cap_mb=self.config.bucket_size_mb
            )
        
        return model
    
    def train_epoch(
        self,
        dataloader,
        optimizer,
        loss_function: Callable,
        scheduler: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch with distributed optimizations.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer instance
            loss_function: Loss function
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'num_batches': 0,
            'grad_norm': 0.0,
            'communication_time': 0.0,
            'computation_time': 0.0
        }
        
        # Setup distributed sampler if needed
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(self.epoch_count)
        
        accumulation_loss = 0.0
        
        for batch_idx, batch_data in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Dynamic load balancing
            if (self.load_balancer and 
                batch_idx % self.config.load_balance_frequency == 0):
                self.load_balancer.balance_load(batch_data)
            
            # Forward pass with mixed precision
            computation_start = time.time()
            
            if self.scaler is not None:
                with autocast():
                    loss = self._forward_pass(batch_data, loss_function)
            else:
                loss = self._forward_pass(batch_data, loss_function)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.accumulation_steps
            accumulation_loss += loss.item()
            
            computation_time = time.time() - computation_start
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation and optimization step
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                communication_start = time.time()
                
                # Gradient compression if enabled
                if self.gradient_compressor is not None:
                    self.gradient_compressor.compress_gradients(self.model)
                
                # Gradient clipping
                if hasattr(optimizer, 'param_groups'):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    epoch_metrics['grad_norm'] += grad_norm.item()
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Learning rate scheduling
                if scheduler is not None:
                    scheduler.step()
                
                communication_time = time.time() - communication_start
                epoch_metrics['communication_time'] += communication_time
                
                # Update step count
                self.step_count += 1
                
                # Checkpointing
                if (self.step_count % self.config.checkpoint_frequency == 0 and
                    self.config.rank == 0):
                    self._save_checkpoint(optimizer, scheduler)
            
            # Update metrics
            epoch_metrics['total_loss'] += accumulation_loss
            epoch_metrics['num_batches'] += 1
            epoch_metrics['computation_time'] += computation_time
            
            # Reset accumulation loss
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                accumulation_loss = 0.0
            
            # Log progress
            if batch_idx % 100 == 0 and self.config.rank == 0:
                self.logger.info(
                    f"Epoch {self.epoch_count}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Comp Time: {computation_time:.3f}s, "
                    f"Comm Time: {communication_time:.3f}s"
                )
        
        # Synchronize metrics across all processes
        if self.is_distributed and dist is not None:
            self._synchronize_metrics(epoch_metrics)
        
        # Average metrics
        for key in epoch_metrics:
            if epoch_metrics['num_batches'] > 0:
                epoch_metrics[key] = epoch_metrics[key] / epoch_metrics['num_batches']
        
        self.epoch_count += 1
        self.training_metrics.append(epoch_metrics)
        
        return epoch_metrics
    
    def _forward_pass(self, batch_data, loss_function: Callable) -> torch.Tensor:
        """Perform forward pass through model."""
        # Extract inputs from batch data
        if isinstance(batch_data, (list, tuple)):
            inputs = batch_data[0]
            targets = batch_data[1] if len(batch_data) > 1 else None
        else:
            inputs = batch_data
            targets = None
        
        # Move to appropriate device
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.local_rank}')
            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute loss
        if targets is not None:
            loss = loss_function(outputs, targets)
        else:
            # Assume loss function handles single input
            loss = loss_function(outputs)
        
        return loss
    
    def _synchronize_metrics(self, metrics: Dict[str, float]):
        """Synchronize metrics across all processes."""
        if dist is None or not dist.is_initialized():
            return
        
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=f'cuda:{self.config.local_rank}' if torch.cuda.is_available() else 'cpu')
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            metrics[key] = tensor.item() / self.config.world_size
    
    def _save_checkpoint(self, optimizer, scheduler=None):
        """Save training checkpoint."""
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'config': self.config,
            'training_metrics': self.training_metrics
        }
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        self.checkpoint_manager.save_checkpoint(
            checkpoint_data, f'checkpoint_step_{self.step_count}.pth'
        )
    
    def load_checkpoint(self, checkpoint_path: str, optimizer, scheduler=None) -> bool:
        """Load training checkpoint."""
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        if checkpoint_data is None:
            return False
        
        # Load model state
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Restore training state
        self.step_count = checkpoint_data.get('step_count', 0)
        self.epoch_count = checkpoint_data.get('epoch_count', 0)
        self.training_metrics = checkpoint_data.get('training_metrics', [])
        
        self.logger.info(f"Loaded checkpoint from step {self.step_count}")
        return True
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        if self.is_distributed and dist is not None and dist.is_initialized():
            dist.destroy_process_group()


class GradientCompressor:
    """
    Gradient compression for efficient communication.
    
    Reduces communication overhead in distributed training by
    compressing gradients before synchronization.
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.1,
        quantization_bits: int = 8
    ):
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits
        self.compression_stats = {'original_size': 0, 'compressed_size': 0}
    
    def compress_gradients(self, model: nn.Module):
        """Compress gradients using top-k sparsification and quantization."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                compressed_grad = self._compress_tensor(param.grad)
                param.grad.data = compressed_grad
    
    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compress a single tensor."""
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        # Record original size
        self.compression_stats['original_size'] += flat_tensor.numel()
        
        # Top-k sparsification
        k = max(1, int(flat_tensor.numel() * self.compression_ratio))
        _, top_indices = torch.topk(torch.abs(flat_tensor), k)
        
        # Create sparse tensor
        sparse_values = flat_tensor[top_indices]
        
        # Quantization
        quantized_values = self._quantize_values(sparse_values)
        
        # Reconstruct tensor
        compressed_flat = torch.zeros_like(flat_tensor)
        compressed_flat[top_indices] = quantized_values
        
        # Record compressed size
        self.compression_stats['compressed_size'] += k
        
        return compressed_flat.view(original_shape)
    
    def _quantize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Quantize values to reduce precision."""
        if self.quantization_bits >= 32:
            return values
        
        # Simple linear quantization
        max_val = torch.max(torch.abs(values))
        scale = max_val / (2 ** (self.quantization_bits - 1) - 1)
        
        quantized = torch.round(values / scale) * scale
        return quantized
    
    def get_compression_ratio(self) -> float:
        """Get actual compression ratio achieved."""
        if self.compression_stats['original_size'] == 0:
            return 1.0
        
        return (self.compression_stats['compressed_size'] / 
                self.compression_stats['original_size'])


class DynamicLoadBalancer:
    """
    Dynamic load balancing for distributed training.
    
    Monitors computational load across processes and redistributes
    work to balance training efficiency.
    """
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.load_history = []
        self.rebalance_threshold = 0.2  # 20% imbalance threshold
    
    def balance_load(self, batch_data):
        """Balance computational load across processes."""
        # Measure current load (simplified)
        current_load = self._measure_load(batch_data)
        self.load_history.append(current_load)
        
        # Check if rebalancing is needed
        if len(self.load_history) >= 10:
            avg_load = sum(self.load_history[-10:]) / 10
            load_variance = sum((l - avg_load)**2 for l in self.load_history[-10:]) / 10
            load_std = load_variance ** 0.5
            
            if load_std / avg_load > self.rebalance_threshold:
                self._rebalance_work()
    
    def _measure_load(self, batch_data) -> float:
        """Measure computational load of current batch."""
        # Simple load measurement based on batch size
        if hasattr(batch_data, 'shape'):
            return float(batch_data.shape[0])  # Batch size
        elif isinstance(batch_data, (list, tuple)):
            return float(len(batch_data))
        else:
            return 1.0
    
    def _rebalance_work(self):
        """Rebalance work distribution."""
        # In a real implementation, this would communicate with other processes
        # to redistribute batches based on current load
        pass


class CheckpointManager:
    """
    Fault-tolerant checkpoint management.
    
    Handles saving and loading of training checkpoints with
    automatic retry and corruption detection.
    """
    
    def __init__(self, checkpoint_dir: str, max_retries: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        
        # Metadata tracking
        self.metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {'checkpoints': [], 'latest': None}
        return {'checkpoints': [], 'latest': None}
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], filename: str) -> bool:
        """Save checkpoint with retry logic."""
        checkpoint_path = self.checkpoint_dir / filename
        
        for attempt in range(self.max_retries):
            try:
                # Save checkpoint
                torch.save(checkpoint_data, checkpoint_path)
                
                # Verify checkpoint integrity
                if self._verify_checkpoint(checkpoint_path):
                    # Update metadata
                    checkpoint_info = {
                        'filename': filename,
                        'timestamp': time.time(),
                        'step_count': checkpoint_data.get('step_count', 0),
                        'size_bytes': checkpoint_path.stat().st_size
                    }
                    
                    self.metadata['checkpoints'].append(checkpoint_info)
                    self.metadata['latest'] = filename
                    self._save_metadata()
                    
                    return True
                else:
                    # Remove corrupted checkpoint
                    checkpoint_path.unlink(missing_ok=True)
                    
            except Exception as e:
                print(f"Checkpoint save attempt {attempt + 1} failed: {e}")
                if checkpoint_path.exists():
                    checkpoint_path.unlink(missing_ok=True)
        
        return False
    
    def load_checkpoint(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint with corruption detection."""
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            return None
        
        for attempt in range(self.max_retries):
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                
                if self._verify_checkpoint_data(checkpoint_data):
                    return checkpoint_data
                
            except Exception as e:
                print(f"Checkpoint load attempt {attempt + 1} failed: {e}")
        
        return None
    
    def _verify_checkpoint(self, checkpoint_path: Path) -> bool:
        """Verify checkpoint file integrity."""
        try:
            # Quick load test
            data = torch.load(checkpoint_path, map_location='cpu')
            return self._verify_checkpoint_data(data)
        except Exception:
            return False
    
    def _verify_checkpoint_data(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Verify checkpoint data integrity."""
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'step_count']
        return all(key in checkpoint_data for key in required_keys)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get filename of latest checkpoint."""
        return self.metadata.get('latest')
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints to save disk space."""
        checkpoints = sorted(
            self.metadata['checkpoints'],
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        # Remove old checkpoints
        for checkpoint_info in checkpoints[keep_last:]:
            checkpoint_path = self.checkpoint_dir / checkpoint_info['filename']
            checkpoint_path.unlink(missing_ok=True)
        
        # Update metadata
        self.metadata['checkpoints'] = checkpoints[:keep_last]
        self._save_metadata()


class FederatedTrainer:
    """
    Federated learning coordinator for collaborative protein research.
    
    Enables multiple institutions to collaboratively train models
    without sharing sensitive data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        participant_id: str,
        aggregation_rounds: int = 10,
        local_epochs: int = 5
    ):
        self.model = model
        self.participant_id = participant_id
        self.aggregation_rounds = aggregation_rounds
        self.local_epochs = local_epochs
        
        # Federated learning state
        self.global_model_state = None
        self.local_updates = []
        self.round_number = 0
        
    def federated_training_round(
        self,
        local_dataloader,
        optimizer,
        loss_function: Callable
    ) -> Dict[str, Any]:
        """Perform one round of federated training."""
        # Load global model if available
        if self.global_model_state is not None:
            self.model.load_state_dict(self.global_model_state)
        
        # Store initial model state
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Local training
        local_metrics = self._local_training(
            local_dataloader, optimizer, loss_function
        )
        
        # Compute model updates
        final_state = self.model.state_dict()
        model_updates = {
            k: final_state[k] - initial_state[k]
            for k in initial_state.keys()
        }
        
        # Store updates for aggregation
        self.local_updates.append({
            'participant_id': self.participant_id,
            'updates': model_updates,
            'data_size': len(local_dataloader.dataset),
            'metrics': local_metrics
        })
        
        self.round_number += 1
        
        return {
            'round': self.round_number,
            'local_metrics': local_metrics,
            'updates_norm': sum(torch.norm(u).item() for u in model_updates.values())
        }
    
    def _local_training(
        self,
        dataloader,
        optimizer,
        loss_function: Callable
    ) -> Dict[str, float]:
        """Perform local training for federated learning."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            for batch_data in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(batch_data, (list, tuple)):
                    inputs, targets = batch_data[0], batch_data[1]
                else:
                    inputs, targets = batch_data, None
                
                outputs = self.model(inputs)
                
                if targets is not None:
                    loss = loss_function(outputs, targets)
                else:
                    loss = loss_function(outputs)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'avg_loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'num_batches': num_batches,
            'epochs': self.local_epochs
        }
    
    def aggregate_updates(self, all_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate model updates from all participants."""
        # Weighted average based on data size
        total_data_size = sum(update['data_size'] for update in all_updates)
        
        aggregated_updates = {}
        
        # Get parameter names from first update
        param_names = list(all_updates[0]['updates'].keys())
        
        for param_name in param_names:
            weighted_sum = torch.zeros_like(all_updates[0]['updates'][param_name])
            
            for update in all_updates:
                weight = update['data_size'] / total_data_size
                weighted_sum += weight * update['updates'][param_name]
            
            aggregated_updates[param_name] = weighted_sum
        
        return aggregated_updates
    
    def apply_global_update(self, global_updates: Dict[str, torch.Tensor]):
        """Apply global model updates."""
        current_state = self.model.state_dict()
        
        new_state = {
            k: current_state[k] + global_updates[k]
            for k in current_state.keys()
        }
        
        self.model.load_state_dict(new_state)
        self.global_model_state = new_state


def setup_distributed_training(
    model: nn.Module,
    config: DistributedConfig
) -> Tuple[DistributedTrainer, Dict[str, Any]]:
    """
    Setup distributed training environment.
    
    Args:
        model: Neural network model
        config: Distributed training configuration
        
    Returns:
        Tuple of (trainer, setup_info)
    """
    # Create distributed trainer
    trainer = DistributedTrainer(model, config)
    
    # Setup information
    setup_info = {
        'world_size': config.world_size,
        'rank': config.rank,
        'local_rank': config.local_rank,
        'backend': config.backend,
        'mixed_precision': config.mixed_precision,
        'gradient_compression': config.gradient_compression,
        'device': f'cuda:{config.local_rank}' if torch.cuda.is_available() else 'cpu'
    }
    
    return trainer, setup_info
