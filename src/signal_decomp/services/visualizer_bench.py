"""Benchmark visualizations (Loss, Performance, Noise)."""

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_loss(all_histories, save_path):
    """Plot training loss curves for all models."""
    plt.figure(figsize=(10, 5))
    for name, history in all_histories.items():
        plt.plot(history, label=name)
    plt.title("Training Loss per Epoch — All Models")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_noise_sensitivity(noise_levels, noise_results, save_path):
    """Plot reconstruction error vs noise level."""
    plt.figure(figsize=(10, 5))
    noise_pct = [n * 100 for n in noise_levels]
    for name, results in noise_results.items():
        plt.plot(noise_pct, results, marker="o", label=name)
    plt.title("Reconstruction Error vs Noise Level")
    plt.xlabel("Amplitude Noise Level (%)")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_test_performance_comparison(test_results, save_path):
    """Plot a bar chart comparing the final Test MSE of all models."""
    plt.style.use("dark_background")
    models = list(test_results.keys())
    mse_values = list(test_results.values())
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF007F', '#00E5FF', '#FFD600']
    bars = plt.bar(models, mse_values, color=colors, alpha=0.8)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', 
                 ha='center', va='bottom', color='white', fontweight='bold')
    
    plt.title("Final Test Performance Comparison (MSE)", fontsize=14)
    plt.ylabel("Mean Squared Error (Lower is Better)")
    plt.grid(True, axis='y', alpha=0.1)
    plt.savefig(save_path, dpi=200)
    plt.close()
    plt.style.use("default")
