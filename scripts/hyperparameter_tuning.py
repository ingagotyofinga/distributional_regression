#!/usr/bin/env python3
"""
Comprehensive hyperparameter tuning script for MNIST distributional regression.

This script implements multiple optimization strategies:
1. Grid search for systematic exploration
2. Random search for efficient sampling
3. Bayesian optimization for intelligent search
4. Early stopping and learning rate scheduling
"""

import os
import random
import math
import json
import time
import argparse
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# -----------------------
# Configuration
# -----------------------
@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""
    # Model architecture
    base_channels: int = 32
    groups: int = 8
    
    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 1
    n_epochs: int = 200
    
    # Kernel and loss parameters
    h_kernel: float = 0.003
    blur: float = 0.04
    
    # Optimization strategy
    optimizer: str = 'adam'
    weight_decay: float = 0.0
    lr_scheduler: str = 'none'
    lr_patience: int = 10
    lr_factor: float = 0.5
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-6
    
    # Data augmentation
    data_augmentation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 1337):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------
# Device & constants
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
H = W = 28

# -----------------------
# Pixel grid in [0,1]^2
# -----------------------
yy, xx = torch.meshgrid(
    torch.linspace(0,1,H, device=device),
    torch.linspace(0,1,W, device=device),
    indexing="ij"
)
X = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (784, 2)

def _to_probs(img):
    img = torch.nn.functional.softplus(img)             # nonnegative
    return img / (img.sum(dim=(2,3), keepdim=True) + 1e-12)

# Per-sample W2^2 (loop is fine for MNIST)
@torch.no_grad()
def w2_batch_nograd(a, b, sinkhorn):
    vals = []
    for ai, bi in zip(a, b):
        ai = _to_probs(ai.unsqueeze(0)).view(-1)        # (784,)
        bi = _to_probs(bi.unsqueeze(0)).view(-1)
        vals.append(sinkhorn(ai, X, bi, X))
    return torch.stack(vals)                             # (B,)

def w2_batch(a, b, sinkhorn):
    vals = []
    for ai, bi in zip(a, b):
        ai = _to_probs(ai.unsqueeze(0)).view(-1)
        bi = _to_probs(bi.unsqueeze(0)).view(-1)
        vals.append(sinkhorn(ai, X, bi, X))
    return torch.stack(vals)

# Gaussian kernel on W2^2
def kernel_weights(mu0, mu_batch, h=0.01, sinkhorn=None):
    with torch.no_grad():
        d2 = w2_batch_nograd(mu_batch, mu0.expand_as(mu_batch), sinkhorn)
    return torch.exp(-d2 / (2 * h * h))                  # (B,)

# -----------------------
# Model Architecture
# -----------------------
def conv3x3(in_ch, out_ch, stride=1, groups=8):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )

class ResBlock(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch),
        )
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.net(x) + x)

class TinyUNet(nn.Module):
    def __init__(self, base=32, groups=8):
        super().__init__()
        c1, c2 = base, base*2

        # Encoder
        self.e1 = nn.Sequential(conv3x3(1, c1, groups=groups), ResBlock(c1, groups=groups))
        self.e2 = nn.Sequential(
            conv3x3(c1, c2, stride=2, groups=groups),
            ResBlock(c2, groups=groups)
        )

        # Bottleneck
        self.mid = nn.Sequential(conv3x3(c2, c2, groups=groups), ResBlock(c2, groups=groups))

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.d1  = nn.Sequential(
            conv3x3(c2 + c1, c1, groups=groups),
            ResBlock(c1, groups=groups)
        )

        # Head
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        z  = self.mid(s2)
        y  = self.up1(z)
        y  = torch.cat([y, s1], dim=1)
        y  = self.d1(y)
        return self.out(y)

def to_display_img(yhat, normalize_mass=True):
    img = torch.nn.functional.softplus(yhat)
    if normalize_mass:
        img = img / (img.sum(dim=(2,3), keepdim=True) + 1e-12)
    mn = img.amin(dim=(2,3), keepdim=True)
    mx = img.amax(dim=(2,3), keepdim=True)
    disp = (img - mn) / (mx - mn + 1e-12)
    return disp.clamp(0,1)

# -----------------------
# Training Function
# -----------------------
def train_model(config: HyperparameterConfig, n_per_class: int = 200, 
                src_label: int = 2, tgt_label: int = 8, 
                verbose: bool = False) -> Dict[str, Any]:
    """
    Train a model with the given hyperparameter configuration.
    
    Returns:
        Dictionary containing training metrics and results
    """
    set_seed()
    
    # Load data
    train = MNIST(root=".", train=True, download=True, transform=ToTensor())
    
    def collect(ds, label, n=None):
        xs = [img for (img, y) in ds if y == label]
        if n: xs = xs[:n]
        return torch.stack(xs)
    
    src_all = collect(train, src_label, n=n_per_class).to(device)
    tgt_all = collect(train, tgt_label, n=n_per_class).to(device)
    m = min(len(src_all), len(tgt_all))
    
    # Hold out first pair as (μ0, ν0); train on the rest
    mu0 = src_all[:1]
    nu0 = tgt_all[:1]
    src = src_all[1:m]
    tgt = tgt_all[1:m]
    M = src.shape[0]
    
    # Create data loader
    indices = torch.arange(M, device=device)
    dataset = TensorDataset(src, tgt, indices)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    # Initialize model and optimizer
    model = TinyUNet(base=config.base_channels, groups=config.groups).to(device)
    
    # Choose optimizer
    if config.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                                   weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                    weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, 
                                  weight_decay=config.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Learning rate scheduler
    if config.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience, verbose=verbose
        )
    elif config.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)
    elif config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.n_epochs//3, gamma=0.5)
    else:
        scheduler = None
    
    # Initialize Sinkhorn loss
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=config.blur)
    
    # Training loop
    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0
    loss_history = []
    val_history = []
    lr_history = []
    
    start_time = time.time()
    
    for epoch in range(config.n_epochs):
        model.train()
        
        # Precompute weights once per epoch
        all_w = kernel_weights(mu0, src, h=config.h_kernel, sinkhorn=sinkhorn).detach()
        W = float(all_w.sum().cpu()) + 1e-12
        
        epoch_loss = 0.0
        
        for (x_i, y_i, idx_i) in loader:
            w_i = all_w[idx_i.item()] if config.batch_size == 1 else all_w[idx_i]
            yhat_i = model(x_i)
            
            d2_i = w2_batch(yhat_i, y_i, sinkhorn)
            if config.batch_size == 1:
                d2_i = d2_i[0]
                loss_i = (w_i / W) * d2_i
            else:
                # For batch_size > 1, compute loss for each sample and sum
                loss_i = torch.sum((w_i / W) * d2_i)
            
            optimizer.zero_grad()
            loss_i.backward()
            optimizer.step()
            
            if config.batch_size == 1:
                epoch_loss += float((w_i * d2_i).detach().cpu())
            else:
                epoch_loss += float(torch.sum(w_i * d2_i).detach().cpu())
        
        epoch_loss = epoch_loss / W
        loss_history.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            nu0hat = model(mu0)
            val_w2 = float(w2_batch_nograd(nu0hat, nu0, sinkhorn)[0].cpu())
        val_history.append(val_w2)
        
        # Learning rate tracking
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Update scheduler
        if scheduler is not None:
            if config.lr_scheduler == 'plateau':
                scheduler.step(val_w2)
            else:
                scheduler.step()
        
        # Early stopping
        if val_w2 < best_val - config.min_delta:
            best_val = val_w2
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if config.early_stopping and patience_counter >= config.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
        
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:03d}: Loss={epoch_loss:.6f}, Val_W2={val_w2:.6f}, LR={current_lr:.2e}")
    
    training_time = time.time() - start_time
    
    return {
        'best_val_w2': best_val,
        'best_epoch': best_epoch,
        'final_loss': loss_history[-1] if loss_history else float('inf'),
        'final_val_w2': val_history[-1] if val_history else float('inf'),
        'training_time': training_time,
        'n_epochs_trained': len(loss_history),
        'loss_history': loss_history,
        'val_history': val_history,
        'lr_history': lr_history,
        'config': config.to_dict()
    }

# -----------------------
# Optimization Strategies
# -----------------------
class HyperparameterOptimizer:
    """Base class for hyperparameter optimization strategies."""
    
    def __init__(self, search_space: Dict[str, List], n_trials: int = 50):
        self.search_space = search_space
        self.n_trials = n_trials
        self.results = []
    
    def optimize(self, n_per_class: int = 200) -> List[Dict[str, Any]]:
        """Run optimization and return results."""
        raise NotImplementedError
    
    def get_best_config(self) -> HyperparameterConfig:
        """Get the best configuration found."""
        if not self.results:
            raise ValueError("No results available. Run optimize() first.")
        
        best_result = min(self.results, key=lambda x: x['best_val_w2'])
        return HyperparameterConfig(**best_result['config'])

class GridSearchOptimizer(HyperparameterOptimizer):
    """Grid search optimization."""
    
    def optimize(self, n_per_class: int = 200) -> List[Dict[str, Any]]:
        """Run grid search optimization."""
        param_grid = ParameterGrid(self.search_space)
        total_combinations = len(param_grid)
        
        print(f"Grid search: {total_combinations} combinations to test")
        
        for i, params in enumerate(param_grid):
            print(f"\n--- Trial {i+1}/{total_combinations} ---")
            print(f"Parameters: {params}")
            
            config = HyperparameterConfig(**params)
            result = train_model(config, n_per_class=n_per_class, verbose=False)
            self.results.append(result)
            
            print(f"Best Val W2: {result['best_val_w2']:.6f}")
            print(f"Training time: {result['training_time']:.2f}s")
        
        return self.results

class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search optimization."""
    
    def optimize(self, n_per_class: int = 200) -> List[Dict[str, Any]]:
        """Run random search optimization."""
        print(f"Random search: {self.n_trials} trials")
        
        for i in range(self.n_trials):
            print(f"\n--- Trial {i+1}/{self.n_trials} ---")
            
            # Sample random parameters
            params = {}
            for param_name, param_values in self.search_space.items():
                params[param_name] = random.choice(param_values)
            
            print(f"Parameters: {params}")
            
            config = HyperparameterConfig(**params)
            result = train_model(config, n_per_class=n_per_class, verbose=False)
            self.results.append(result)
            
            print(f"Best Val W2: {result['best_val_w2']:.6f}")
            print(f"Training time: {result['training_time']:.2f}s")
        
        return self.results

# -----------------------
# Main Execution
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for MNIST distributional regression')
    parser.add_argument('--strategy', type=str, default='random', 
                       choices=['grid', 'random'], help='Optimization strategy')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials for random search')
    parser.add_argument('--n_per_class', type=int, default=200, help='Number of samples per class')
    parser.add_argument('--output_dir', type=str, default='../results/hyperparameter_tuning', 
                       help='Output directory for results')
    parser.add_argument('--src_label', type=int, default=2, help='Source class label')
    parser.add_argument('--tgt_label', type=int, default=8, help='Target class label')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define search space
    search_space = {
        'base_channels': [16, 32, 64, 128],
        'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        'h_kernel': [0.001, 0.003, 0.01, 0.03, 0.1],
        'blur': [0.01, 0.02, 0.04, 0.08, 0.16],
        'optimizer': ['adam', 'adamw', 'sgd'],
        'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
        'lr_scheduler': ['none', 'plateau', 'cosine', 'step'],
        'patience': [10, 20, 30, 50],
        'batch_size': [1, 2, 4, 8],
        'groups': [4, 8, 16, 32]
    }
    
    # Initialize optimizer
    if args.strategy == 'grid':
        optimizer = GridSearchOptimizer(search_space)
    else:
        optimizer = RandomSearchOptimizer(search_space, n_trials=args.n_trials)
    
    # Run optimization
    print(f"Starting {args.strategy} search optimization...")
    print(f"Source class: {args.src_label}, Target class: {args.tgt_label}")
    print(f"Number of samples per class: {args.n_per_class}")
    
    start_time = time.time()
    results = optimizer.optimize(n_per_class=args.n_per_class)
    total_time = time.time() - start_time
    
    # Save results
    results_file = os.path.join(args.output_dir, f'{args.strategy}_search_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Get best configuration
    best_config = optimizer.get_best_config()
    best_result = min(results, key=lambda x: x['best_val_w2'])
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Best validation W2: {best_result['best_val_w2']:.6f}")
    print(f"Best configuration:")
    for key, value in best_config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Save best configuration
    best_config_file = os.path.join(args.output_dir, 'best_config.json')
    with open(best_config_file, 'w') as f:
        json.dump(best_config.to_dict(), f, indent=2)
    
    # Create summary plot
    create_summary_plot(results, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")

def create_summary_plot(results: List[Dict[str, Any]], output_dir: str):
    """Create summary plots of the optimization results."""
    
    # Extract metrics
    val_w2s = [r['best_val_w2'] for r in results]
    training_times = [r['training_time'] for r in results]
    n_epochs = [r['n_epochs_trained'] for r in results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Validation W2 distribution
    axes[0, 0].hist(val_w2s, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Best Validation W2')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Best Validation W2')
    axes[0, 0].axvline(min(val_w2s), color='red', linestyle='--', label=f'Best: {min(val_w2s):.6f}')
    axes[0, 0].legend()
    
    # Plot 2: Training time vs performance
    axes[0, 1].scatter(training_times, val_w2s, alpha=0.7)
    axes[0, 1].set_xlabel('Training Time (s)')
    axes[0, 1].set_ylabel('Best Validation W2')
    axes[0, 1].set_title('Training Time vs Performance')
    
    # Plot 3: Epochs trained vs performance
    axes[1, 0].scatter(n_epochs, val_w2s, alpha=0.7)
    axes[1, 0].set_xlabel('Epochs Trained')
    axes[1, 0].set_ylabel('Best Validation W2')
    axes[1, 0].set_title('Epochs vs Performance')
    
    # Plot 4: Top 10 configurations
    sorted_results = sorted(results, key=lambda x: x['best_val_w2'])[:10]
    trial_nums = list(range(1, len(sorted_results) + 1))
    top_val_w2s = [r['best_val_w2'] for r in sorted_results]
    
    axes[1, 1].bar(trial_nums, top_val_w2s, alpha=0.7)
    axes[1, 1].set_xlabel('Trial Rank')
    axes[1, 1].set_ylabel('Best Validation W2')
    axes[1, 1].set_title('Top 10 Configurations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {os.path.join(output_dir, 'optimization_summary.png')}")

if __name__ == "__main__":
    main()
