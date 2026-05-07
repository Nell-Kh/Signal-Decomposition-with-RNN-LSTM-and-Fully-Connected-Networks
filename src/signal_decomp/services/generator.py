import numpy as np


def generate_base_signals(t, frequencies, amplitude, fixed_phases=None):
    """Generate 4 clean signals with fixed random phases."""
    num_freq = len(frequencies)
    clean_signals = np.zeros((num_freq, len(t)))

    phases = fixed_phases if fixed_phases is not None else np.random.uniform(0, 2 * np.pi, num_freq)

    for i, freq in enumerate(frequencies):
        clean_signals[i] = amplitude * np.sin(2 * np.pi * freq * t + phases[i])
    return clean_signals, phases

def apply_noise(clean_signals, phases, frequencies, t, amplitude, noise_pct, noise_phi):
    """Apply random noise per sample using configured amplitude[cite: 13]."""
    n_samples = clean_signals.shape[1]
    noisy_signals = np.zeros_like(clean_signals)

    for i, freq in enumerate(frequencies):
        epsilon_a = np.random.uniform(-noise_pct * amplitude, noise_pct * amplitude, n_samples)
        epsilon_phi = np.random.uniform(-noise_phi, noise_phi, n_samples)
        # Use the amplitude variable here, not hardcoded 1.0
        noisy_signals[i] = (amplitude + epsilon_a) * np.sin(
            2 * np.pi * freq * t + phases[i] + epsilon_phi
        )

    composite = np.sum(noisy_signals, axis=0)
    return noisy_signals, composite
