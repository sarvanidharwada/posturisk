"""Unit tests for posturisk.features."""

import numpy as np
import pytest

from posturisk.features import (
    _bandpower,
    _dominant_freq,
    _rms,
    _spectral_entropy,
    calc_postural_sway_features,
    extract_signal_features,
)


class TestRms:
    """Tests for _rms()."""

    def test_known_value(self):
        x = np.array([1.0, -1.0, 1.0, -1.0])
        assert _rms(x) == pytest.approx(1.0)

    def test_zeros(self):
        assert _rms(np.zeros(100)) == pytest.approx(0.0)


class TestFrequencies:
    """Tests for frequency domain functions."""

    def test_bandpower_sine_wave(self):
        fs = 100
        t = np.arange(1000) / fs
        # 5 Hz sine wave -> should be purely in the mid band (3-10 Hz)
        x = np.sin(2 * np.pi * 5 * t)
        
        low = _bandpower(x, fs, 0.1, 3.0)
        mid = _bandpower(x, fs, 3.0, 10.0)
        high = _bandpower(x, fs, 10.0, 20.0)
        
        assert mid > 0.9  # Nearly all power
        assert low < 0.1
        assert high < 0.1

    def test_dominant_freq(self):
        fs = 100
        t = np.arange(1000) / fs
        x = np.sin(2 * np.pi * 7.5 * t)
        dom = _dominant_freq(x, fs)
        assert dom == pytest.approx(7.5, abs=0.5)


class TestPosturalSway:
    """Tests for calc_postural_sway_features()."""

    def test_postural_sway_zero(self):
        ml = np.zeros(100)
        ap = np.zeros(100)
        sway = calc_postural_sway_features(ml, ap, 100)
        assert sway["sway_path_length"] == pytest.approx(0.0)
        assert sway["sway_area"] == pytest.approx(0.0)
        assert sway["sway_mean_velocity"] == pytest.approx(0.0)

    def test_postural_sway_circle(self):
        fs = 100
        t = np.arange(100) / fs
        # Circle radius 1, frequency 1 Hz -> 1 revolution per second
        ml = np.cos(2 * np.pi * 1 * t)
        ap = np.sin(2 * np.pi * 1 * t)
        sway = calc_postural_sway_features(ml, ap, fs)
        
        # Path length around a circle of radius 1 is 2*pi approx
        assert sway["sway_path_length"] > 6.0
        assert sway["sway_mean_velocity"] > 6.0


class TestExtractSignalFeatures:
    """Tests for extract_signal_features()."""

    def test_returns_dict(self):
        rng = np.random.default_rng(0)
        signals = rng.standard_normal((1000, 3))
        features = extract_signal_features(signals, fs=100, signal_names=["a", "b", "c"])
        assert isinstance(features, dict)

    def test_feature_count(self):
        rng = np.random.default_rng(0)
        signals = rng.standard_normal((1000, 6))
        features = extract_signal_features(signals, fs=100)
        # 17 features per signal * 6 signals + 7 cross-channel = 109 features
        assert len(features) == 109

    def test_feature_names_prefix(self):
        signals = np.random.default_rng(0).standard_normal((500, 2))
        features = extract_signal_features(signals, fs=100, signal_names=["x", "y"])
        assert "x_mean" in features
        assert "x_var" in features
        assert "y_jerk_rms" in features
        assert "y_power_high" in features

    def test_no_nan_features(self):
        signals = np.random.default_rng(0).standard_normal((500, 3))
        features = extract_signal_features(signals, fs=100, signal_names=["a", "b", "c"])
        for name, val in features.items():
            assert np.isfinite(val), f"Non-finite value for feature '{name}': {val}"
