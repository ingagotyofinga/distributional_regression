#!/usr/bin/env python3
"""
Advanced hyperparameter tuning with Bayesian optimization and advanced strategies.

This script implements:
1. Bayesian optimization using Gaussian processes
2. Multi-objective optimization (performance vs training time)
3. Hyperband for efficient resource allocation
4. Advanced learning rate scheduling
5. Model ensemble evaluation
"""

import os
import random
import math
import json
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional
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
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("Warning: scikit-optimize not available. Bayesian optimization disabled.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Advanced optimization disabled.")

# Import base classes from the main hyperparameter tuning script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hyperparameter_tuning import (
    HyperparameterConfig, set_seed, device, H, W, X, 
    _to_probs, w2_batch_nograd, w2_batch, kernel_weights,
    conv3x3, ResBlock, TinyUNet, to_display_img, train_model
)

# -----------------------
# Advanced Configuration
# -----------------------
@dataclass
class AdvancedHyperparameterConfig(HyperparameterConfig):
    """Extended configuration with advanced hyperparameters."""
    # Advanced training
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5
    warmup_factor: float = 0.1
    
    # Model architecture enhancements
    dropout_rate: float = 0.0
    use_attention: bool = False
    activation: str = 'silu'  # silu, gelu, relu, leaky_relu
    
    # Data augmentation
    rotation_degrees: float = 0.0
    translation: float = 0.0
    scale_range: Tuple[float, float] = (1.0, 1.0)
    
    # Ensemble parameters
    ensemble_size: int = 1
    diversity_weight: float = 0.0

# -----------------------
# Advanced Optimizers
# -----------------------
class BayesianOptimizer:
    """Bayesian optimization using Gaussian processes."""
    
    def __init__(self, search_space: Dict[str, Any], n_trials: int = 50):
        if not BAYESIAN_OPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
        
        self.n_trials = n_trials
        self.results = []
        self._setup_search_space(search_space)
    
    def _setup_search_space(self, search_space: Dict[str, Any]):
        """Convert search space to scikit-optimize format."""
        self.dimensions = []
        self.param_names = []
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                if param_config['type'] == 'real':
                    self.dimensions.append(Real(
                        param_config['low'], 
                        param_config['high'], 
                        name=param_name
                    ))
                elif param_config['type'] == 'integer':
                    self.dimensions.append(Integer(
                        param_config['low'], 
                        param_config['high'], 
                        name=param_name
                    ))
                elif param_config['type'] == 'categorical':
                    self.dimensions.append(Categorical(
                        param_config['choices'], 
                        name=param_name
                    ))
            else:
                # Simple list of choices
                self.dimensions.append(Categorical(param_config, name=param_name))
            
            self.param_names.append(param_name)
    
    def optimize(self, n_per_class: int = 200) -> List[Dict[str, Any]]:
        """Run Bayesian optimization."""
        print(f"Bayesian optimization: {self.n_trials} trials")
        
        @use_named_args(dimensions=self.dimensions)
        def objective(**params):
            # Convert parameters to config
            config_dict = {}
            for name, value in params.items():
                if name in ['base_channels', 'groups', 'n_epochs', 'patience', 'warmup_epochs']:
                    config_dict[name] = int(value)
                elif name in ['ensemble_size']:
                    config_dict[name] = int(value)
                else:
                    config_dict[name] = value
            
            config = AdvancedHyperparameterConfig(**config_dict)
            
            # Train model
            result = train_model(config, n_per_class=n_per_class, verbose=False)
            
            # Store result
            result['config'] = config.to_dict()
            self.results.append(result)
            
            print(f"Trial {len(self.results)}: Val W2 = {result['best_val_w2']:.6f}")
            
            return result['best_val_w2']
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=self.dimensions,
            n_calls=self.n_trials,
            random_state=42
        )
        
        return self.results

class OptunaOptimizer:
    """Advanced optimization using Optuna."""
    
    def __init__(self, search_space: Dict[str, Any], n_trials: int = 50):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for advanced optimization")
        
        self.n_trials = n_trials
        self.results = []
        self.search_space = search_space
    
    def optimize(self, n_per_class: int = 200) -> List[Dict[str, Any]]:
        """Run Optuna optimization."""
        print(f"Optuna optimization: {self.n_trials} trials")
        
        def objective(trial):
            # Sample parameters
            config_dict = {}
            
            for param_name, param_config in self.search_space.items():
                if isinstance(param_config, dict):
                    if param_config['type'] == 'real':
                        config_dict[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'integer':
                        config_dict[param_name] = trial.suggest_int(
                            param_name, 
                            param_config['low'], 
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        config_dict[param_name] = trial.suggest_categorical(
                            param_name, 
                            param_config['choices']
                        )
                else:
                    # Simple list of choices
                    config_dict[param_name] = trial.suggest_categorical(param_name, param_config)
            
            config = AdvancedHyperparameterConfig(**config_dict)
            
            # Train model
            result = train_model(config, n_per_class=n_per_class, verbose=False)
            
            # Store result
            result['config'] = config.to_dict()
            self.results.append(result)
            
            print(f"Trial {len(self.results)}: Val W2 = {result['best_val_w2']:.6f}")
            
            return result['best_val_w2']
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return self.results

class HyperbandOptimizer:
    """Hyperband for efficient resource allocation."""
    
    def __init__(self, search_space: Dict[str, Any], max_epochs: int = 200, eta: float = 3):
        self.search_space = search_space
        self.max_epochs = max_epochs
        self.eta = eta
        self.results = []
    
    def _get_hyperband_configs(self, n_trials: int) -> List[Dict[str, Any]]:
        """Generate configurations for Hyperband."""
        configs = []
        
        for _ in range(n_trials):
            config_dict = {}
            for param_name, param_values in self.search_space.items():
                if isinstance(param_values, list):
                    config_dict[param_name] = random.choice(param_values)
                else:
                    # Handle dict format
                    if param_values['type'] == 'real':
                        config_dict[param_name] = random.uniform(
                            param_values['low'], param_values['high']
                        )
                    elif param_values['type'] == 'integer':
                        config_dict[param_name] = random.randint(
                            param_values['low'], param_values['high']
                        )
                    elif param_values['type'] == 'categorical':
                        config_dict[param_name] = random.choice(param_values['choices'])
            
            configs.append(config_dict)
        
        return configs
    
    def optimize(self, n_per_class: int = 200) -> List[Dict[str, Any]]:
        """Run Hyperband optimization."""
        print(f"Hyperband optimization with max_epochs={self.max_epochs}")
        
        # Calculate Hyperband brackets
        s_max = int(math.log(self.max_epochs) / math.log(self.eta))
        B = (s_max + 1) * self.max_epochs
        
        configs = self._get_hyperband_configs(100)  # Generate more configs than needed
        
        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / self.max_epochs * (self.eta ** s) / (s + 1)))
            r = self.max_epochs * (self.eta ** (-s))
            
            print(f"\nBracket {s}: {n} configs, {int(r)} epochs each")
            
            # Evaluate configurations
            bracket_results = []
            for i in range(min(n, len(configs))):
                config_dict = configs[i]
                config = AdvancedHyperparameterConfig(**config_dict)
                config.n_epochs = int(r)  # Override epochs for this bracket
                
                result = train_model(config, n_per_class=n_per_class, verbose=False)
                result['config'] = config.to_dict()
                bracket_results.append(result)
                
                print(f"  Config {i+1}: Val W2 = {result['best_val_w2']:.6f}")
            
            # Sort by performance and keep best
            bracket_results.sort(key=lambda x: x['best_val_w2'])
            self.results.extend(bracket_results)
            
            # Keep best configs for next bracket
            if s > 0:
                best_configs = [r['config'] for r in bracket_results[:n//self.eta]]
                configs = best_configs + self._get_hyperband_configs(n - len(best_configs))
        
        return self.results

# -----------------------
# Multi-Objective Optimization
# -----------------------
class MultiObjectiveOptimizer:
    """Multi-objective optimization considering both performance and efficiency."""
    
    def __init__(self, search_space: Dict[str, Any], n_trials: int = 50):
        self.search_space = search_space
        self.n_trials = n_trials
        self.results = []
    
    def optimize(self, n_per_class: int = 200) -> List[Dict[str, Any]]:
        """Run multi-objective optimization."""
        print(f"Multi-objective optimization: {self.n_trials} trials")
        
        # Generate random configurations
        for i in range(self.n_trials):
            print(f"\n--- Trial {i+1}/{self.n_trials} ---")
            
            # Sample parameters
            config_dict = {}
            for param_name, param_values in self.search_space.items():
                if isinstance(param_values, list):
                    config_dict[param_name] = random.choice(param_values)
                else:
                    if param_values['type'] == 'real':
                        config_dict[param_name] = random.uniform(
                            param_values['low'], param_values['high']
                        )
                    elif param_values['type'] == 'integer':
                        config_dict[param_name] = random.randint(
                            param_values['low'], param_values['high']
                        )
                    elif param_values['type'] == 'categorical':
                        config_dict[param_name] = random.choice(param_values['choices'])
            
            config = AdvancedHyperparameterConfig(**config_dict)
            
            # Train model
            result = train_model(config, n_per_class=n_per_class, verbose=False)
            result['config'] = config.to_dict()
            
            # Calculate composite score (weighted combination)
            performance_score = 1.0 / (1.0 + result['best_val_w2'])  # Higher is better
            efficiency_score = 1.0 / (1.0 + result['training_time'] / 1000)  # Higher is better
            result['composite_score'] = 0.7 * performance_score + 0.3 * efficiency_score
            
            self.results.append(result)
            
            print(f"Val W2: {result['best_val_w2']:.6f}, Time: {result['training_time']:.2f}s, Score: {result['composite_score']:.4f}")
        
        return self.results

# -----------------------
# Main Execution
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='Advanced hyperparameter tuning for MNIST distributional regression')
    parser.add_argument('--strategy', type=str, default='optuna', 
                       choices=['bayesian', 'optuna', 'hyperband', 'multi_objective'], 
                       help='Optimization strategy')
    parser.add_argument('--n_trials', type=int, default=30, help='Number of trials')
    parser.add_argument('--n_per_class', type=int, default=200, help='Number of samples per class')
    parser.add_argument('--output_dir', type=str, default='../results/advanced_hyperparameter_tuning', 
                       help='Output directory for results')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum epochs for Hyperband')
    parser.add_argument('--src_label', type=int, default=2, help='Source class label')
    parser.add_argument('--tgt_label', type=int, default=8, help='Target class label')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define advanced search space
    search_space = {
        'base_channels': {'type': 'integer', 'low': 16, 'high': 128},
        'learning_rate': {'type': 'real', 'low': 1e-5, 'high': 1e-2, 'log': True},
        'h_kernel': {'type': 'real', 'low': 0.0001, 'high': 0.1, 'log': True},
        'blur': {'type': 'real', 'low': 0.01, 'high': 0.2, 'log': True},
        'optimizer': ['adam', 'adamw', 'sgd'],
        'weight_decay': {'type': 'real', 'low': 0.0, 'high': 1e-2, 'log': True},
        'lr_scheduler': ['none', 'plateau', 'cosine', 'step'],
        'patience': {'type': 'integer', 'low': 5, 'high': 50},
        'batch_size': [1, 2, 4, 8, 16],
        'groups': {'type': 'integer', 'low': 4, 'high': 32},
        'gradient_clip_norm': {'type': 'real', 'low': 0.1, 'high': 10.0, 'log': True},
        'warmup_epochs': {'type': 'integer', 'low': 0, 'high': 20},
        'dropout_rate': {'type': 'real', 'low': 0.0, 'high': 0.5},
        'activation': ['silu', 'gelu', 'relu', 'leaky_relu'],
        'rotation_degrees': {'type': 'real', 'low': 0.0, 'high': 15.0},
        'translation': {'type': 'real', 'low': 0.0, 'high': 0.1},
        'diversity_weight': {'type': 'real', 'low': 0.0, 'high': 0.1}
    }
    
    # Initialize optimizer
    if args.strategy == 'bayesian':
        if not BAYESIAN_OPT_AVAILABLE:
            print("Error: scikit-optimize not available. Falling back to random search.")
            args.strategy = 'multi_objective'
        else:
            optimizer = BayesianOptimizer(search_space, n_trials=args.n_trials)
    elif args.strategy == 'optuna':
        if not OPTUNA_AVAILABLE:
            print("Error: Optuna not available. Falling back to random search.")
            args.strategy = 'multi_objective'
        else:
            optimizer = OptunaOptimizer(search_space, n_trials=args.n_trials)
    elif args.strategy == 'hyperband':
        optimizer = HyperbandOptimizer(search_space, max_epochs=args.max_epochs)
    else:  # multi_objective
        optimizer = MultiObjectiveOptimizer(search_space, n_trials=args.n_trials)
    
    # Run optimization
    print(f"Starting {args.strategy} optimization...")
    print(f"Source class: {args.src_label}, Target class: {args.tgt_label}")
    print(f"Number of samples per class: {args.n_per_class}")
    
    start_time = time.time()
    results = optimizer.optimize(n_per_class=args.n_per_class)
    total_time = time.time() - start_time
    
    # Save results
    results_file = os.path.join(args.output_dir, f'{args.strategy}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Get best configuration
    if args.strategy == 'multi_objective':
        best_result = max(results, key=lambda x: x['composite_score'])
    else:
        best_result = min(results, key=lambda x: x['best_val_w2'])
    
    best_config = AdvancedHyperparameterConfig(**best_result['config'])
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Best validation W2: {best_result['best_val_w2']:.6f}")
    if 'composite_score' in best_result:
        print(f"Composite score: {best_result['composite_score']:.4f}")
    print(f"Best configuration:")
    for key, value in best_config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Save best configuration
    best_config_file = os.path.join(args.output_dir, 'best_config.json')
    with open(best_config_file, 'w') as f:
        json.dump(best_config.to_dict(), f, indent=2)
    
    # Create advanced summary plots
    create_advanced_summary_plots(results, args.output_dir, args.strategy)
    
    print(f"\nResults saved to: {args.output_dir}")

def create_advanced_summary_plots(results: List[Dict[str, Any]], output_dir: str, strategy: str):
    """Create advanced summary plots."""
    
    # Extract metrics
    val_w2s = [r['best_val_w2'] for r in results]
    training_times = [r['training_time'] for r in results]
    n_epochs = [r['n_epochs_trained'] for r in results]
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Validation W2 distribution
    axes[0, 0].hist(val_w2s, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Best Validation W2')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Best Validation W2')
    axes[0, 0].axvline(min(val_w2s), color='red', linestyle='--', label=f'Best: {min(val_w2s):.6f}')
    axes[0, 0].legend()
    
    # Plot 2: Training time vs performance
    scatter = axes[0, 1].scatter(training_times, val_w2s, alpha=0.7, c=n_epochs, cmap='viridis')
    axes[0, 1].set_xlabel('Training Time (s)')
    axes[0, 1].set_ylabel('Best Validation W2')
    axes[0, 1].set_title('Training Time vs Performance')
    plt.colorbar(scatter, ax=axes[0, 1], label='Epochs Trained')
    
    # Plot 3: Epochs vs performance
    axes[0, 2].scatter(n_epochs, val_w2s, alpha=0.7)
    axes[0, 2].set_xlabel('Epochs Trained')
    axes[0, 2].set_ylabel('Best Validation W2')
    axes[0, 2].set_title('Epochs vs Performance')
    
    # Plot 4: Learning curves for top 5 configurations
    top_5 = sorted(results, key=lambda x: x['best_val_w2'])[:5]
    for i, result in enumerate(top_5):
        if 'loss_history' in result and result['loss_history']:
            axes[1, 0].plot(result['loss_history'], alpha=0.7, label=f'Config {i+1}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Loss')
    axes[1, 0].set_title('Learning Curves (Top 5)')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    
    # Plot 5: Validation curves for top 5 configurations
    for i, result in enumerate(top_5):
        if 'val_history' in result and result['val_history']:
            axes[1, 1].plot(result['val_history'], alpha=0.7, label=f'Config {i+1}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation W2')
    axes[1, 1].set_title('Validation Curves (Top 5)')
    axes[1, 1].legend()
    
    # Plot 6: Hyperparameter importance (if available)
    if strategy in ['optuna', 'bayesian']:
        # Create a simple importance plot based on variance
        param_importance = {}
        for result in results:
            config = result['config']
            for param, value in config.items():
                if param not in param_importance:
                    param_importance[param] = []
                param_importance[param].append((value, result['best_val_w2']))
        
        # Calculate correlation with performance
        param_corrs = {}
        for param, values in param_importance.items():
            if len(set([v[0] for v in values])) > 1:  # Only if parameter varies
                vals = [v[0] for v in values]
                perfs = [v[1] for v in values]
                corr = np.corrcoef(vals, perfs)[0, 1]
                param_corrs[param] = abs(corr)
        
        if param_corrs:
            sorted_params = sorted(param_corrs.items(), key=lambda x: x[1], reverse=True)[:10]
            params, corrs = zip(*sorted_params)
            axes[1, 2].barh(range(len(params)), corrs)
            axes[1, 2].set_yticks(range(len(params)))
            axes[1, 2].set_yticklabels(params)
            axes[1, 2].set_xlabel('|Correlation with Performance|')
            axes[1, 2].set_title('Parameter Importance')
        else:
            axes[1, 2].text(0.5, 0.5, 'No parameter variation', ha='center', va='center')
            axes[1, 2].set_title('Parameter Importance')
    else:
        axes[1, 2].text(0.5, 0.5, 'Not available for this strategy', ha='center', va='center')
        axes[1, 2].set_title('Parameter Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{strategy}_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Advanced summary plot saved to: {os.path.join(output_dir, f'{strategy}_summary.png')}")

if __name__ == "__main__":
    main()
