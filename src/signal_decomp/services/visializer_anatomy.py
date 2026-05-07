"""Anatomy and diagnostic visualizations (Dual-Scale, Micro-Diagnostic, Clean Sins)."""

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_dual_scale_anatomy(t, base_signals, clean_sum, noisy_sum, save_path):
    """Plot signals at both 10s (Macro) and 1s (Micro) scales."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, clean_sum, color='#00FF41', label="Clean Composite")
    ax0.set_title("Macro View: Full 10s Horizon")
    ax0.legend()
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, noisy_sum, color='#FF007F', alpha=0.7, label="Noisy Input")
    ax1.legend()
    z_s, z_e = 5000, 6000
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t[z_s:z_e], clean_sum[z_s:z_e], color='#00FF41')
    ax2.set_title("Micro View: 1s Zoom (5s-6s)")
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t[z_s:z_e], noisy_sum[z_s:z_e], color='#FF007F', alpha=0.8)
    ax_logic = fig.add_subplot(gs[2, :])
    colors = ['#00E5FF', '#FFD600', '#FF007F', '#FFFFFF']
    for i in range(base_signals.shape[0]):
        ax_logic.plot(t[z_s:z_e], base_signals[i, z_s:z_e], color=colors[i], label=f"{i+1}Hz Component")
    ax_logic.plot(t[z_s:z_e], clean_sum[z_s:z_e], color='#00FF41', linestyle='--', label="Sum")
    ax_logic.legend(loc='upper right', ncol=5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    plt.style.use('default')

def plot_clean_components(t, signals, save_path):
    """Plot the 4 pure sinusoids in a high-fidelity reference chart."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    colors = ['#00E5FF', '#FFD600', '#FF007F', '#FFFFFF']
    freqs = [1, 3, 5, 7]
    fs, zoom_end = 1000, 2000 
    for i in range(4):
        axes[i].plot(t[:zoom_end], signals[i, :zoom_end], color=colors[i], linewidth=2)
        axes[i].set_title(f"Pure Component {i+1}: {freqs[i]}Hz Sine Wave", color="white", fontsize=12)
        axes[i].grid(True, alpha=0.1)
        axes[i].set_ylabel("Amplitude")
    axes[3].set_xlabel("Time (Seconds)")
    plt.suptitle("The Ground Truth: Baseline Periodic Components", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close()
    plt.style.use("default")

def plot_window_micro_diagnostic(trained_models, datasets, save_path):
    """Show exactly one 10-sample window across all models."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    idx = 42 
    x_fc_all, x_fc_test, y_all, y_test = datasets["FC"]
    x_seq_all, x_seq_test, _, _ = datasets["RNN"]
    s_fc, s_seq, target = x_fc_test[idx], x_seq_test[idx], y_test[idx]
    one_hot, noisy = s_fc[:4], s_fc[4:]
    preds = {}
    for name in ["FC", "RNN", "LSTM"]:
        model, _, _ = trained_models[name]
        model.eval()
        with torch.no_grad():
            inp = torch.tensor(s_fc if name=="FC" else s_seq).float().unsqueeze(0)
            preds[name] = model(inp).numpy().squeeze()
    t10 = np.arange(10)
    axes[0].plot(t10, noisy, 'o--', color='#FF007F', label="Input")
    axes[0].set_title(f"Input Window\nOne-Hot: {one_hot}")
    axes[1].plot(t10, target, 'o-', color='#00FF41', linewidth=3, label="Target")
    axes[2].plot(t10, target, color='#00FF41', alpha=0.2)
    axes[2].plot(t10, preds["RNN"], 'x-', color='#00E5FF', label="RNN")
    axes[2].plot(t10, preds["LSTM"], 's-', color='#FFD600', label="LSTM")
    axes[3].plot(t10, target, color='#00FF41', alpha=0.2)
    axes[3].plot(t10, preds["FC"], 'd-', color='#FFFFFF', label="FC")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    plt.style.use("default")
