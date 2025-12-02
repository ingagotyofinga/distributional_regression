#!/usr/bin/env python3
"""
Simple script to run hyperparameter optimization with recommended settings.

This script provides an easy interface to run optimization with sensible defaults
and automatically selects the best available optimization strategy.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check which optimization strategies are available."""
    strategies = {}
    
    # Check for basic dependencies
    try:
        import torch
        import matplotlib
        import numpy
        strategies['basic'] = True
    except ImportError as e:
        print(f"Error: Missing basic dependency: {e}")
        return {}
    
    # Check for scikit-optimize (Bayesian optimization)
    try:
        import skopt
        strategies['bayesian'] = True
        print("✓ scikit-optimize available for Bayesian optimization")
    except ImportError:
        strategies['bayesian'] = False
        print("✗ scikit-optimize not available (install with: pip install scikit-optimize)")
    
    # Check for Optuna (advanced optimization)
    try:
        import optuna
        strategies['optuna'] = True
        print("✓ Optuna available for advanced optimization")
    except ImportError:
        strategies['optuna'] = False
        print("✗ Optuna not available (install with: pip install optuna)")
    
    return strategies

def install_dependencies():
    """Install recommended dependencies."""
    print("Installing recommended dependencies...")
    
    dependencies = [
        'scikit-optimize',
        'optuna',
        'plotly',  # For better visualization
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✓ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {dep}")

def run_optimization(strategy='auto', n_trials=30, n_per_class=200, output_dir=None):
    """Run hyperparameter optimization with the specified strategy."""
    
    if output_dir is None:
        output_dir = f"../results/optimization_{strategy}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which script to run
    if strategy == 'auto':
        strategies = check_dependencies()
        if strategies.get('optuna', False):
            strategy = 'optuna'
            script = 'advanced_hyperparameter_tuning.py'
        elif strategies.get('bayesian', False):
            strategy = 'bayesian'
            script = 'advanced_hyperparameter_tuning.py'
        else:
            strategy = 'random'
            script = 'hyperparameter_tuning.py'
        print(f"Auto-selected strategy: {strategy}")
    
    # Build command
    if script == 'advanced_hyperparameter_tuning.py':
        cmd = [
            sys.executable, script,
            '--strategy', strategy,
            '--n_trials', str(n_trials),
            '--n_per_class', str(n_per_class),
            '--output_dir', output_dir
        ]
    else:
        cmd = [
            sys.executable, script,
            '--strategy', strategy,
            '--n_trials', str(n_trials),
            '--n_per_class', str(n_per_class),
            '--output_dir', output_dir
        ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")
    
    # Run optimization
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Optimization completed successfully!")
        print(f"Results saved to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Optimization failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization for MNIST distributional regression')
    parser.add_argument('--strategy', type=str, default='auto', 
                       choices=['auto', 'random', 'grid', 'bayesian', 'optuna', 'hyperband', 'multi_objective'],
                       help='Optimization strategy (auto selects best available)')
    parser.add_argument('--n_trials', type=int, default=30, 
                       help='Number of trials to run')
    parser.add_argument('--n_per_class', type=int, default=200, 
                       help='Number of samples per class')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--install_deps', action='store_true',
                       help='Install recommended dependencies first')
    parser.add_argument('--quick', action='store_true',
                       help='Run a quick test with fewer trials')
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        install_dependencies()
        return
    
    # Adjust trials for quick mode
    if args.quick:
        args.n_trials = min(args.n_trials, 10)
        print(f"Quick mode: Running {args.n_trials} trials")
    
    # Check dependencies
    strategies = check_dependencies()
    if not strategies.get('basic', False):
        print("Error: Missing basic dependencies. Please install required packages.")
        return
    
    # Run optimization
    success = run_optimization(
        strategy=args.strategy,
        n_trials=args.n_trials,
        n_per_class=args.n_per_class,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE!")
        print("="*60)
        print("Next steps:")
        print("1. Check the results directory for detailed outputs")
        print("2. Look at the summary plots to understand performance")
        print("3. Use the best configuration found for your final model")
        print("4. Consider running with more trials for better results")
    else:
        print("\nOptimization failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

