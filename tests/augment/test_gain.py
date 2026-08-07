import numpy as np

from vad.augment.gain import apply_gain_db, random_gain


def test_apply_gain_db_zero_is_noop():
    x = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.allclose(apply_gain_db(x, 0.0), x)


def test_apply_gain_db_doubles_amplitude_at_plus6db():
    x = np.array([0.1, 0.2], dtype=np.float32)
    y = apply_gain_db(x, 6.0206)  # +6.0206dB ~= x2
    assert np.allclose(y, x * 2.0, atol=1e-3)


def test_apply_gain_db_halves_amplitude_at_minus6db():
    x = np.array([0.4, 0.2], dtype=np.float32)
    y = apply_gain_db(x, -6.0206)
    assert np.allclose(y, x * 0.5, atol=1e-3)


def test_random_gain_within_range_and_finite():
    rng = np.random.default_rng(0)
    x = np.random.default_rng(1).standard_normal(1000).astype(np.float32)
    for _ in range(50):
        y = random_gain(rng, x, range_db=(-3.0, 3.0))
        assert np.all(np.isfinite(y))
        # gain factor must be within [10^(-3/20), 10^(3/20)]
        ratio = np.abs(y[x != 0] / x[x != 0])
        assert np.all(ratio >= 10 ** (-3.0 / 20.0) - 1e-6)
        assert np.all(ratio <= 10 ** (3.0 / 20.0) + 1e-6)
