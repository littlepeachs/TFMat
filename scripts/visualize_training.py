#!/usr/bin/env python
"""
Visualize training metrics from run.metrics.log files.

Usage:
    python scripts/visualize_training.py --runs CSP-mp20 crystalmf-imf-mp20-baseline
"""

import argparse
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def parse_metrics_log(log_path):
    """Parse metrics from run.metrics.log file."""
    metrics = defaultdict(lambda: defaultdict(list))

    with open(log_path, 'r') as f:
        for line in f:
            # Extract the dictionary from the log line
            match = re.search(r'\{.*\}', line)
            if not match:
                continue

            try:
                data = eval(match.group())  # Safe here since we control the log format
                epoch = data.get('epoch')

                if epoch is None:
                    continue

                # Collect all metrics
                for key, value in data.items():
                    if key != 'epoch' and isinstance(value, (int, float)):
                        metrics[key]['epochs'].append(epoch)
                        metrics[key]['values'].append(value)
            except:
                continue

    return metrics


def plot_training_curves(runs_data, output_dir, show=False):
    """Plot training curves comparing multiple runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define metrics to plot
    metric_groups = {
        'Total Loss': ['train_loss_epoch', 'val_loss'],
        'Lattice Loss': ['lattice_loss_epoch', 'val_lattice_loss'],
        'Coord Loss': ['coord_loss_epoch', 'val_coord_loss'],
    }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for group_name, metric_keys in metric_groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(group_name, fontsize=16, fontweight='bold')

        for ax_idx, metric_key in enumerate(metric_keys):
            ax = axes[ax_idx]

            for run_idx, (run_name, metrics) in enumerate(runs_data.items()):
                if metric_key in metrics:
                    epochs = metrics[metric_key]['epochs']
                    values = metrics[metric_key]['values']

                    color = colors[run_idx % len(colors)]
                    ax.plot(epochs, values, label=run_name, color=color, linewidth=2, alpha=0.8)

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(metric_key.replace('_', ' ').title(), fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f'{group_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

        if show:
            plt.show()
        plt.close()

    # Plot all losses together for comparison
    fig, ax = plt.subplots(figsize=(12, 7))

    for run_idx, (run_name, metrics) in enumerate(runs_data.items()):
        color = colors[run_idx % len(colors)]

        # Plot train and val loss
        if 'train_loss_epoch' in metrics:
            epochs = metrics['train_loss_epoch']['epochs']
            values = metrics['train_loss_epoch']['values']
            ax.plot(epochs, values, label=f'{run_name} (train)',
                   color=color, linewidth=2, linestyle='-', alpha=0.8)

        if 'val_loss' in metrics:
            epochs = metrics['val_loss']['epochs']
            values = metrics['val_loss']['values']
            ax.plot(epochs, values, label=f'{run_name} (val)',
                   color=color, linewidth=2, linestyle='--', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training and Validation Loss Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'all_losses_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def print_summary_stats(runs_data):
    """Print summary statistics for each run."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for run_name, metrics in runs_data.items():
        print(f"\n{run_name}:")
        print("-" * 60)

        # Get final epoch metrics
        if 'train_loss_epoch' in metrics and len(metrics['train_loss_epoch']['values']) > 0:
            final_train = metrics['train_loss_epoch']['values'][-1]
            final_epoch = metrics['train_loss_epoch']['epochs'][-1]
            print(f"  Final Epoch: {final_epoch}")
            print(f"  Final Train Loss: {final_train:.6f}")

        if 'val_loss' in metrics and len(metrics['val_loss']['values']) > 0:
            final_val = metrics['val_loss']['values'][-1]
            best_val = min(metrics['val_loss']['values'])
            best_epoch = metrics['val_loss']['epochs'][np.argmin(metrics['val_loss']['values'])]
            print(f"  Final Val Loss: {final_val:.6f}")
            print(f"  Best Val Loss: {best_val:.6f} (epoch {best_epoch})")

        if 'lattice_loss_epoch' in metrics and len(metrics['lattice_loss_epoch']['values']) > 0:
            final_lattice = metrics['lattice_loss_epoch']['values'][-1]
            print(f"  Final Lattice Loss: {final_lattice:.6f}")

        if 'coord_loss_epoch' in metrics and len(metrics['coord_loss_epoch']['values']) > 0:
            final_coord = metrics['coord_loss_epoch']['values'][-1]
            print(f"  Final Coord Loss: {final_coord:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--runs', nargs='+', required=True,
                       help='Run names (subdirectories in hydra_jobs/singlerun/)')
    parser.add_argument('--base-dir', default='/ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun',
                       help='Base directory containing run folders')
    parser.add_argument('--output-dir', default='./training_plots',
                       help='Output directory for plots')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    runs_data = {}

    # Load metrics from each run
    for run_name in args.runs:
        log_path = base_dir / run_name / 'run.metrics.log'

        if not log_path.exists():
            print(f"Warning: {log_path} not found, skipping...")
            continue

        print(f"Loading metrics from: {log_path}")
        metrics = parse_metrics_log(log_path)
        runs_data[run_name] = metrics

    if not runs_data:
        print("Error: No valid run data found!")
        return

    # Plot training curves
    plot_training_curves(runs_data, args.output_dir, show=args.show)

    # Print summary statistics
    print_summary_stats(runs_data)

    print(f"\n✓ All plots saved to: {Path(args.output_dir).absolute()}")


if __name__ == '__main__':
    main()
