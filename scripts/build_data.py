"""
Build the dataset once and save to data/ folder.
All three models (FC, RNN, LSTM) will load from these files
to ensure they train and are evaluated on identical data.

Run once with:
    uv run python build_data.py

Saves:
    data/X_FC.npy    shape: (10000, 14)    FC flat input
    data/X_seq.npy   shape: (10000, 10, 5) RNN/LSTM sequence input
    data/Y.npy       shape: (10000, 10)    clean target (all models)
    data/meta.json   signal metadata (phases, config used)
"""

import json
from pathlib import Path

import numpy as np

from signal_decomp.services.dataset import build_dataset_with_fresh_noise
from signal_decomp.services.generator import apply_noise, generate_base_signals
from signal_decomp.services.visualizer_anatomy import plot_dual_scale_anatomy, plot_clean_components
from signal_decomp.shared.config import load_config
from signal_decomp.shared.constants import FREQUENCIES


def main():
    """Generate signals, build dataset, save to data/ folder."""
    config = load_config()
    Path("data").mkdir(exist_ok=True)

    amplitude = config["signal"]["amplitude"]
    duration = config["signal"]["duration"]
    fs = config["signal"]["fs"]
    noise_amp = config["noise"]["amplitude_pct"]
    noise_phi = config["noise"]["phase_range"]
    num_samples = config["dataset"]["num_samples"]
    window_size = config["dataset"]["window_size"]

    # Step 1 — Time vector
    t = np.arange(0, duration, 1 / fs)
    print(f"Time vector: {len(t)} samples ({duration}s at {fs}Hz)")

    # Step 2 — Generate clean signals with fixed phases
    # Phases drawn ONCE and saved so experiments are reproducible
    clean, phases = generate_base_signals(t, FREQUENCIES, amplitude)
    print(f"Clean signals shape: {clean.shape}")
    print(f"Phases (rad): {phases.round(3)}")

    # Step 3 — Apply noise and get composite
    noisy, composite = apply_noise(clean, phases, FREQUENCIES, t, amplitude, noise_amp, noise_phi)
    print(f"Composite shape: {composite.shape}")
    print(f"Composite range: {composite.min():.3f} to {composite.max():.3f}")

    # Generate Dual-Scale Signal Anatomy Plot for debugging
    Path("results").mkdir(exist_ok=True)
    plot_dual_scale_anatomy(t, clean, clean.sum(axis=0), composite, "results/signal_anatomy.png")
    plot_clean_components(t, clean, "results/clean_components.png")
    print("Saved: results/signal_anatomy.png (Dual-Scale)")
    print("Saved: results/clean_components.png (Clean Baseline)")

    # Step 4 — Build synchronized dataset for all models
    print(f"\nBuilding synchronized dataset ({num_samples} samples)...")
    x_fc, x_seq, y = build_dataset_with_fresh_noise(
        clean_signals=clean,
        t=t,
        frequencies=FREQUENCIES,
        amplitude=amplitude,
        phases=phases,
        noise_amplitude_pct=noise_amp,
        noise_phase_range=noise_phi,
        num_samples=num_samples,
        window_size=window_size,
    )
    print(f"X_FC shape:  {x_fc.shape}")
    print(f"X_seq shape: {x_seq.shape}")
    print(f"Y shape:     {y.shape}")

    # Step 6 — Save everything
    np.save("data/X_FC.npy", x_fc)
    np.save("data/X_seq.npy", x_seq)
    np.save("data/Y.npy", y)
    print("\nSaved:")
    print("  data/X_FC.npy  ", x_fc.shape)
    print("  data/X_seq.npy ", x_seq.shape)
    print("  data/Y.npy     ", y.shape)

    # Step 7 — Save metadata so we know exactly what generated this data
    meta = {
        "frequencies": FREQUENCIES,
        "amplitude": amplitude,
        "phases": phases.tolist(),
        "fs": fs,
        "duration": duration,
        "noise_amplitude_pct": noise_amp,
        "noise_phase_range": noise_phi,
        "num_samples": num_samples,
        "window_size": window_size,
    }
    with open("data/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  data/meta.json")

    # Step 8 — Quick sanity check
    print("\n--- Sanity Check ---")
    print("Sample 0:")
    print(f"  c (one-hot):     {x_fc[0][:4]}")
    print(f"  noisy composite: {x_fc[0][4:].round(3)}")
    print(f"  clean target:    {y[0].round(3)}")
    print("\nSample 1:")
    print(f"  c (one-hot):     {x_fc[1][:4]}")
    print(f"  noisy composite: {x_fc[1][4:].round(3)}")
    print(f"  clean target:    {y[1].round(3)}")

    print("\nDataset built successfully!")


if __name__ == "__main__":
    main()
