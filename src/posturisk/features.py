"""Feature extraction module for accelerometer and posturography signals.

Extracts time-domain, frequency-domain, and postural sway features from
preprocessed 3D acceleration and angular velocity signals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

# Expected signal index if using the default 6-channel LTMM format
# ['v_acc', 'ml_acc', 'ap_acc', 'yaw_vel', 'pitch_vel', 'roll_vel']
IDX_V_ACC = 0
IDX_ML_ACC = 1
IDX_AP_ACC = 2


def _rms(x: NDArray) -> float:
    """Root mean square of array *x*."""
    return float(np.sqrt(np.mean(x**2)))


def _dominant_freq(x: NDArray, fs: int) -> float:
    """Dominant frequency (Hz) of signal *x* sampled at *fs* Hz."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(x))
    # Ignore DC component
    fft_mag[0] = 0
    return float(freqs[np.argmax(fft_mag)])


def _spectral_entropy(x: NDArray, fs: int) -> float:
    """Normalised spectral entropy of signal *x*."""
    freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    psd = psd / (psd.sum() + 1e-12)
    entropy = -np.sum(psd * np.log2(psd + 1e-12))
    max_entropy = np.log2(len(psd))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _bandpower(x: NDArray, fs: int, fmin: float, fmax: float) -> float:
    """Relative power in frequency band [fmin, fmax]."""
    freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    total_power = np.sum(psd)
    if total_power == 0:
        return 0.0
    return float(np.sum(psd[idx_band]) / total_power)


def calc_postural_sway_features(ml: NDArray, ap: NDArray, fs: int) -> dict[str, float]:
    """Calculate postural sway features from ML and AP signals.

    Parameters
    ----------
    ml : NDArray
        Medio-lateral signal (e.g. acceleration or position).
    ap : NDArray
        Antero-posterior signal.
    fs : int
        Sample rate in Hz.

    Returns
    -------
    dict
        Dictionary containing path length, sway area, and mean velocity.
    """
    # Exclude mean to center the trajectory
    ml_centered = ml - np.mean(ml)
    ap_centered = ap - np.mean(ap)

    # Path length (sum of distances between consecutive points)
    dp = np.sqrt(np.diff(ml_centered)**2 + np.diff(ap_centered)**2)
    path_length = float(np.sum(dp))

    # Mean velocity
    duration = len(ml) / fs
    mean_velocity = path_length / duration if duration > 0 else 0.0

    # Sway area (95% confidence ellipse area approximation)
    # Area = pi * F * sqrt(var_ml * var_ap - cov_ml_ap^2)
    # Using F0.05[2, N-2] approximation (~3.0 for large N)
    cov_matrix = np.cov(ml_centered, ap_centered)
    var_ml = cov_matrix[0, 0]
    var_ap = cov_matrix[1, 1]
    cov_ml_ap = cov_matrix[0, 1]
    
    # Ensure non-negative argument for sqrt
    det = max(0, var_ml * var_ap - cov_ml_ap**2)
    sway_area = float(3.0 * np.pi * np.sqrt(det))

    return {
        "sway_path_length": path_length,
        "sway_area": sway_area,
        "sway_mean_velocity": mean_velocity
    }


def extract_signal_features(
    signals: NDArray[np.float64],
    fs: int = 100,
    signal_names: list[str] | None = None,
) -> dict[str, float]:
    """Extract time- and frequency-domain features from multi-channel signals.

    Parameters
    ----------
    signals : NDArray
        Shape ``(n_samples, n_signals)`` array of physical-unit signals.
    fs : int
        Sample rate in Hz.
    signal_names : list[str] | None
        Names for each signal channel.

    Returns
    -------
    dict[str, float]
        Feature name -> value dictionary.
    """
    if signal_names is None:
        # Default to LTMM signal names if none provided
        signal_names = ["v_acc", "ml_acc", "ap_acc", "yaw_vel", "pitch_vel", "roll_vel"]
        signal_names = signal_names[: signals.shape[1]]

    features: dict[str, float] = {}

    for i, name in enumerate(signal_names):
        x = signals[:, i]

        # Time-domain features
        features[f"{name}_mean"] = float(np.mean(x))
        features[f"{name}_var"] = float(np.var(x))
        features[f"{name}_std"] = float(np.std(x))
        features[f"{name}_rms"] = _rms(x)
        features[f"{name}_min"] = float(np.min(x))
        features[f"{name}_max"] = float(np.max(x))
        features[f"{name}_range"] = float(np.ptp(x))
        features[f"{name}_skew"] = float(skew(x))
        features[f"{name}_kurtosis"] = float(kurtosis(x))
        features[f"{name}_iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))

        # Jerk (derivative of acceleration) -> proxy using signal diff
        jerk = np.diff(x) * fs
        features[f"{name}_jerk_mean"] = float(np.mean(jerk))
        features[f"{name}_jerk_rms"] = _rms(jerk)

        # Frequency-domain features
        features[f"{name}_dom_freq"] = _dominant_freq(x, fs)
        features[f"{name}_spectral_entropy"] = _spectral_entropy(x, fs)
        
        # Power in frequency bands (low 0.1-3 Hz, mid 3-10 Hz, high 10-20 Hz)
        features[f"{name}_power_low"] = _bandpower(x, fs, 0.1, 3.0)
        features[f"{name}_power_mid"] = _bandpower(x, fs, 3.0, 10.0)
        features[f"{name}_power_high"] = _bandpower(x, fs, 10.0, 20.0)

    # Cross-channel / Postural Sway features
    if signals.shape[1] >= 3:
        # Total acceleration magnitude
        acc_mag = np.sqrt(np.sum(signals[:, IDX_V_ACC:IDX_AP_ACC+1] ** 2, axis=1))
        features["acc_magnitude_mean"] = float(np.mean(acc_mag))
        features["acc_magnitude_var"] = float(np.var(acc_mag))
        features["acc_magnitude_std"] = float(np.std(acc_mag))
        features["acc_magnitude_rms"] = _rms(acc_mag)
        
        # Postural Sway features derived from ML and AP
        ml = signals[:, IDX_ML_ACC]
        ap = signals[:, IDX_AP_ACC]
        sway_feats = calc_postural_sway_features(ml, ap, fs)
        features.update(sway_feats)

    return features
