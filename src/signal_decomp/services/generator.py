"""Signal generation: clean sinusoids and noisy versions."""
import numpy as np


def generate_base_signals(t, frequencies, amplitude):
    """
    Generate 4 clean sinusoidal signals with fixed random phases.

    Phases are drawn ONCE and fixed for the entire project so that
    all three models (FC, RNN, LSTM) see identical data — ensuring
    a fair comparison.

    Formula: Sk(t) = A * sin(2*pi*fk*t + phi_k)

    Args:
        t:           time vector, shape (N,)
        frequencies: list of 4 frequencies in Hz
        amplitude:   signal amplitude A

    Returns:
        clean_signals: shape (4, N) — the 4 pure sinusoids
        phases:        shape (4,)  — fixed phases, shared across models
    """
    num_freq = len(frequencies)
    clean_signals = np.zeros((num_freq, len(t)))
    phases = np.random.uniform(0, 2 * np.pi, num_freq)

    for i, freq in enumerate(frequencies):
        clean_signals[i] = amplitude * np.sin(
            2 * np.pi * freq * t + phases[i]
        )

    return clean_signals, phases


def apply_noise(clean_signals, phases, frequencies, t,
                amplitude, noise_amplitude_pct, noise_phase_range):
    """
    Apply independent random noise to each of the 4 sinusoids.

    For each sinusoid k, noise is re-drawn independently per sample:
        epsilon_A   ~ Uniform(-noise_amplitude_pct*A, +noise_amplitude_pct*A)
        epsilon_phi ~ Uniform(-noise_phase_range,     +noise_phase_range)

    Noisy formula:
        S_noisy_k(t) = (A + epsilon_A) * sin(2*pi*fk*t + phi_k + epsilon_phi)

    Then the composite is the sum of all 4 noisy sinusoids:
        Sigma(t) = S_noisy_1(t) + S_noisy_2(t) + S_noisy_3(t) + S_noisy_4(t)

    Args:
        clean_signals:      shape (4, N)
        phases:             shape (4,) — same phases as generate_base_signals
        frequencies:        list of 4 frequencies in Hz
        t:                  time vector, shape (N,)
        amplitude:          signal amplitude A
        noise_amplitude_pct: e.g. 0.02 means +/- 2% of A
        noise_phase_range:  e.g. 0.1 means +/- 0.1 rad

    Returns:
        noisy_signals: shape (4, N) — 4 individually noisy sinusoids
        composite:     shape (N,)   — sum of all 4 noisy sinusoids
    """
    n_samples = clean_signals.shape[1]
    noisy_signals = np.zeros_like(clean_signals)

    for i, freq in enumerate(frequencies):
        # Noise re-drawn independently for every single sample
        epsilon_A = np.random.uniform(
            -noise_amplitude_pct * amplitude,
            +noise_amplitude_pct * amplitude,
            n_samples
        )
        epsilon_phi = np.random.uniform(
            -noise_phase_range,
            +noise_phase_range,
            n_samples
        )

        # Apply the noisy formula
        noisy_signals[i] = (amplitude + epsilon_A) * np.sin(
            2 * np.pi * freq * t + phases[i] + epsilon_phi
        )

    # Sum all 4 noisy sinusoids → the composite input signal
    composite = np.sum(noisy_signals, axis=0)

    return noisy_signals, composite