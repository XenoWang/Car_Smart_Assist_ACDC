"""Opt-in low-light perturbation for diagnostic RGB training views."""
from numbers import Real

import numpy as np


def lowlight_rgb(image, exposure, *, shot_scale=0.0, read_std=0.0, rng=None):
    """Return independent uint8 RGB pixels using an approximate linear sensor model.

    shot_scale defines Poisson variance = exposed signal * shot_scale.
    read_std is Gaussian standard deviation in normalized linear RGB.
    The uncalibrated perturbation does not estimate a real camera's photon count.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("Low-light input must be a nonempty uint8 RGB array")
    if image.shape[2] != 3 or not image.shape[0] or not image.shape[1]:
        raise ValueError("Low-light input must be a nonempty uint8 RGB array")
    for name, value in (("exposure", exposure), ("shot_scale", shot_scale), ("read_std", read_std)):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(name + " must be a finite number")
    if not 0 < exposure <= 1 or not 0 <= read_std <= .1:
        raise ValueError("Exposure must be in (0,1] and read_std in [0,0.1]")
    if shot_scale != 0 and not 1e-6 <= shot_scale <= 1:
        raise ValueError("shot_scale must be zero or in [0.000001,1]")
    if (shot_scale or read_std) and rng is None:
        raise ValueError("Noise requires an explicit random generator")

    encoded = image.astype(np.float32) / 255
    linear = np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055) / 1.055) ** 2.4)
    signal = linear * exposure
    if shot_scale:
        signal = rng.poisson(signal / shot_scale).astype(np.float32) * shot_scale
    if read_std:
        signal = signal + rng.normal(0, read_std, signal.shape).astype(np.float32)
    signal = np.clip(signal, 0, 1)
    encoded = np.where(signal <= .0031308, 12.92 * signal, 1.055 * signal ** (1 / 2.4) - .055)
    return np.ascontiguousarray(np.rint(encoded * 255).clip(0, 255).astype(np.uint8))
