#!/usr/bin/env python3
"""
Complete training script for neural operators in protein design.

This script demonstrates the full training pipeline from data generation
to model training with real PyTorch integration and comprehensive validation.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from protein_operators.utils.torch_integration import (
    print_system_info, get_device_info, TORCH_AVAILABLE
)
from protein_operators.training import (
    TrainingConfig, ProteinDataset, NeuralOperatorTrainer
)
from protein_operators.data import TrainingDataGenerator
from protein_operators.models import ProteinDeepONet, ProteinFNO

# PyTorch imports with fallback
if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader
else:
    torch = None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural operators for protein design"
    )
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='deeponet',
                       choices=['deeponet', 'fno'],
                       help='Type of neural operator model')
    
    # Training parameters  
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='Device for training')
    
    # Data configuration
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of training samples to generate')
    parser.add_argument('--synthetic_ratio', type=float, default=0.8,
                       help='Ratio of synthetic to real data')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Physics-informed training
    parser.add_argument('--physics_guided', action='store_true',
                       help='Enable physics-guided training')
    parser.add_argument('--physics_weight', type=float, default=0.1,
                       help='Weight for physics loss term')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    
    # Advanced options
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Enable mixed precision training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


class DataTransform:
    """Transform raw data for neural network training."""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def __call__(self, sample):
        """Transform a data sample."""
        constraints = sample['constraints']
        structure = sample['structure']
        
        # Convert coordinates to tensor
        if TORCH_AVAILABLE:
            coords = torch.tensor(structure['ca_coords'], dtype=torch.float32)
        else:
            # Fallback for mock environment
            from protein_operators.utils.torch_integration import tensor
            coords = tensor(structure['ca_coords'])
        
        # Create constraint encoding (simplified)
        constraint_encoding = self._encode_constraints(constraints)
        
        return {
            'constraints': constraint_encoding,
            'structure': coords,
            'sequence': structure['sequence'],
            'length': structure['length'],
            'pdb_id': structure['pdb_id']
        }
    
    def _encode_constraints(self, constraints):
        """Encode constraints as feature vector."""
        # Simple encoding - in practice would be more sophisticated
        encoding = [0.0] * 64  # 64-dimensional constraint vector
        
        if constraints['constraint_type'] == 'binding_site':
            encoding[0] = 1.0
            if 'binding_affinity' in constraints:
                encoding[1] = min(constraints['binding_affinity'] * 1e6, 10.0)  # Scale to reasonable range
        elif constraints['constraint_type'] == 'secondary_structure':
            encoding[2] = 1.0
        elif constraints['constraint_type'] == 'stability':
            encoding[3] = 1.0
            if 'target_tm' in constraints:
                encoding[4] = constraints['target_tm'] / 100.0  # Normalize
        
        if TORCH_AVAILABLE:
            return torch.tensor(encoding, dtype=torch.float32)
        else:
            from protein_operators.utils.torch_integration import tensor
            return tensor(encoding)


def create_model(model_type: str, config: dict):
    """Create neural operator model."""
    if model_type == 'deeponet':
        model = ProteinDeepONet(
            constraint_dim=config.get('constraint_dim', 64),
            hidden_dim=config.get('hidden_dim', 256),
            output_dim=config.get('output_dim', 3),
            num_basis=config.get('num_basis', 512)
        )
    elif model_type == 'fno':
        model = ProteinFNO(
            modes=config.get('modes', 32),
            width=config.get('width', 64),
            depth=config.get('depth', 4),
            in_channels=config.get('in_channels', 64),
            out_channels=config.get('out_channels', 3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Print system information
    print("=== Protein Operator Training ===")
    print_system_info()
    
    # Set random seeds
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    print("\n=== Data Generation ===")
    data_generator = TrainingDataGenerator(
        cache_dir=f"{args.data_dir}/cache",
        synthetic_ratio=args.synthetic_ratio
    )
    
    constraint_data, structure_data = data_generator.generate_training_dataset(
        num_samples=args.num_samples,
        length_range=(50, 200)
    )
    
    # Create train/validation split
    (train_constraints, train_structures), (val_constraints, val_structures) = \
        data_generator.create_validation_split(
            constraint_data, structure_data, args.val_ratio
        )
    
    print(f"Training samples: {len(train_constraints)}")
    print(f"Validation samples: {len(val_constraints)}")
    
    # Create datasets
    transform = DataTransform(device=args.device)
    
    train_dataset = ProteinDataset(train_constraints, train_structures, transform)
    val_dataset = ProteinDataset(val_constraints, val_structures, transform)
    
    # Create data loaders
    if TORCH_AVAILABLE:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    else:
        # Mock data loader for testing
        class MockDataLoader:
            def __init__(self, dataset, batch_size):
                self.dataset = dataset
                self.batch_size = batch_size
            
            def __iter__(self):
                for i in range(0, len(self.dataset), self.batch_size):
                    batch = []
                    for j in range(i, min(i + self.batch_size, len(self.dataset))):
                        batch.append(self.dataset[j])
                    yield self._collate_batch(batch)
            
            def _collate_batch(self, batch):
                # Simple collation - in practice would stack tensors
                return {
                    'constraints': [b['constraints'] for b in batch],
                    'structure': [b['structure'] for b in batch],
                    'sequence': [b['sequence'] for b in batch]
                }
        
        train_loader = MockDataLoader(train_dataset, args.batch_size)
        val_loader = MockDataLoader(val_dataset, args.batch_size)
    
    # Create model
    print(f"\n=== Model Creation ({args.model_type.upper()}) ===")
    model_config = {
        'constraint_dim': 64,
        'hidden_dim': 256,
        'output_dim': 3
    }
    
    model = create_model(args.model_type, model_config)
    print(f"Model created: {model.__class__.__name__}")
    
    # Create training configuration
    training_config = TrainingConfig(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        physics_guided=args.physics_guided,
        physics_loss_weight=args.physics_weight,
        mixed_precision=args.mixed_precision and TORCH_AVAILABLE,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Create trainer
    trainer = NeuralOperatorTrainer(model, training_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = trainer.load_checkpoint(args.resume)
        trainer.current_epoch = checkpoint.get('epoch', 0)
        trainer.best_val_loss = checkpoint.get('loss', float('inf'))
    
    # Start training
    print(f"\n=== Training Started ===")
    print(f"Device: {get_device_info()['device']}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Physics Guided: {args.physics_guided}")
    
    try:
        history = trainer.train(train_loader, val_loader)
        
        print(f"\n=== Training Completed ===")
        print(f"Final Train Loss: {history['train_losses'][-1]:.4f}")
        if history['val_losses']:
            print(f"Final Val Loss: {history['val_losses'][-1]:.4f}")
        print(f"Best Val Loss: {trainer.best_val_loss:.4f}")
        
        # Save final model
        final_model_path = Path(args.checkpoint_dir) / "final_model.pt"
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            val_loss=trainer.best_val_loss,
            is_best=True
        )
        print(f"Final model saved: {final_model_path}")
        
        # Save training history
        history_path = Path(args.log_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': [float(x) for x in history['train_losses']],
                'val_losses': [float(x) for x in history['val_losses']],
                'config': vars(args)
            }, f, indent=2)
        print(f"Training history saved: {history_path}")
        
    except KeyboardInterrupt:
        print(f"\n=== Training Interrupted ===")
        # Save current state
        interrupt_path = Path(args.checkpoint_dir) / "interrupted_model.pt"
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            val_loss=trainer.best_val_loss
        )
        print(f"Model saved: {interrupt_path}")
    
    except Exception as e:
        print(f"\n=== Training Error ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Training Script Completed ===")


if __name__ == "__main__":
    main()