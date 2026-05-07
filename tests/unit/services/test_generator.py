import numpy as np

from signal_decomp.services.generator import apply_noise, generate_base_signals


def test_generate_base_signals():
    t = np.linspace(0, 1, 1000)
    freqs = [1, 2, 3]
    amp = 1.0

    signals, phases = generate_base_signals(t, freqs, amp)

    assert signals.shape == (3, 1000)
    assert len(phases) == 3
    assert np.all(phases >= 0) and np.all(phases <= 2 * np.pi)
    # Check max amplitude
    assert np.isclose(np.max(np.abs(signals)), amp, atol=1e-5)


def test_generate_base_signals_fixed_phases():
    t = np.linspace(0, 1, 10)
    freqs = [1]
    amp = 1.0
    fixed_phases = np.array([0.0])

    signals, phases = generate_base_signals(t, freqs, amp, fixed_phases=fixed_phases)

    assert np.array_equal(phases, fixed_phases)
    assert np.allclose(signals[0], amp * np.sin(2 * np.pi * freqs[0] * t))


def test_apply_noise():
    t = np.linspace(0, 1, 1000)
    freqs = [1, 2]
    amp = 1.0
    phases = np.array([0.0, 0.0])
    clean_signals, _ = generate_base_signals(t, freqs, amp, fixed_phases=phases)

    noise_pct = 0.1
    noise_phi = 0.1

    noisy_signals, composite = apply_noise(
        clean_signals, phases, freqs, t, amp, noise_pct, noise_phi
    )

    assert noisy_signals.shape == clean_signals.shape
    assert composite.shape == (1000,)
    # Composite should be roughly sum of clean signals plus some noise
    clean_sum = np.sum(clean_signals, axis=0)
    assert not np.array_equal(composite, clean_sum)
    assert np.all(np.abs(composite - clean_sum) < 2.0)  # Broad check
