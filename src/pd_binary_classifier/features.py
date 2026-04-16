from typing import List, Sequence, Tuple

import numpy as np


def _safe_float(x: float) -> float:
    if np.isnan(x) or np.isinf(x):
        return 0.0
    return float(x)


def _iqr(x: np.ndarray) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    return _safe_float(q3 - q1)


def _skew(x: np.ndarray) -> float:
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-12:
        return 0.0
    z = (x - m) / s
    return _safe_float(np.mean(z**3))


def _kurtosis_excess(x: np.ndarray) -> float:
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-12:
        return 0.0
    z = (x - m) / s
    return _safe_float(np.mean(z**4) - 3.0)


def _spectral_stats(x: np.ndarray, fs_hz: float) -> Tuple[float, float, float]:
    n = x.size
    if n == 0:
        return 0.0, 0.0, 0.0

    x_centered = x - np.mean(x)
    spec = np.fft.rfft(x_centered)
    power = (spec.real**2 + spec.imag**2) / max(n, 1)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    if power.size <= 1:
        return 0.0, 0.0, 0.0

    power_no_dc = power.copy()
    power_no_dc[0] = 0.0

    dom_freq = freqs[int(np.argmax(power_no_dc))]

    band_mask = (freqs >= 3.0) & (freqs <= 7.0)
    band_power = np.sum(power[band_mask])

    p = power_no_dc / (np.sum(power_no_dc) + 1e-12)
    spectral_entropy = -np.sum(p * np.log(p + 1e-12))

    return _safe_float(dom_freq), _safe_float(band_power), _safe_float(spectral_entropy)


def channel_feature_vector(x: np.ndarray, fs_hz: float) -> np.ndarray:
    mean_v = np.mean(x)
    std_v = np.std(x)
    median_v = np.median(x)
    iqr_v = _iqr(x)
    rms_v = np.sqrt(np.mean(x**2))
    energy_v = np.sum(x**2)
    abs_mean_v = np.mean(np.abs(x))
    skew_v = _skew(x)
    kurt_v = _kurtosis_excess(x)
    dom_freq, band_power, spectral_entropy = _spectral_stats(x, fs_hz)

    feats = np.array(
        [
            mean_v,
            std_v,
            median_v,
            iqr_v,
            rms_v,
            energy_v,
            abs_mean_v,
            skew_v,
            kurt_v,
            dom_freq,
            band_power,
            spectral_entropy,
        ],
        dtype=np.float32,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def movement_channel_names() -> List[str]:
    tasks = [
        "Relaxed1",
        "Relaxed2",
        "RelaxedTask1",
        "RelaxedTask2",
        "StretchHold",
        "HoldWeight",
        "DrinkGlas",
        "CrossArms",
        "TouchNose",
        "Entrainment1",
        "Entrainment2",
    ]
    wrists = ["LeftWrist", "RightWrist"]
    sensors_axes = [
        ("Accelerometer", "X"),
        ("Accelerometer", "Y"),
        ("Accelerometer", "Z"),
        ("Gyroscope", "X"),
        ("Gyroscope", "Y"),
        ("Gyroscope", "Z"),
    ]

    names: List[str] = []
    for task in tasks:
        for wrist in wrists:
            for sensor, axis in sensors_axes:
                names.append(f"{task}_{wrist}_{sensor}_{axis}")
    return names


def movement_feature_names(channel_names: Sequence[str]) -> List[str]:
    suffixes = [
        "mean",
        "std",
        "median",
        "iqr",
        "rms",
        "energy",
        "abs_mean",
        "skew",
        "kurtosis_excess",
        "dom_freq",
        "bandpower_3_7hz",
        "spectral_entropy",
    ]
    names: List[str] = []
    for ch in channel_names:
        for sfx in suffixes:
            names.append(f"mv_{ch}_{sfx}")
    return names


def extract_movement_features(movement_2d: np.ndarray, fs_hz: float) -> np.ndarray:
    feats = [channel_feature_vector(ch, fs_hz) for ch in movement_2d]
    return np.concatenate(feats).astype(np.float32)


def questionnaire_feature_names(n_items: int) -> List[str]:
    return [f"q_nms_{i+1:02d}" for i in range(n_items)]
