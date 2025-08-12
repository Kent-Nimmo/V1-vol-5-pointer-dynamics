"""Translation functions for the geometry baseline.

This module implements the "gradient‑as‑mass" translator used in the
geometry baseline simulation.  Given a 2‑D electric field snapshot
``E0`` on a lattice of size ``L x L``, it applies Gaussian smoothing,
computes the gradient magnitude, and normalises the result to produce
an envelope field.  No background removal is performed.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

def gaussian_smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth ``field`` using a Gaussian filter.

    Parameters
    ----------
    field : np.ndarray
        Input 2‑D array.
    sigma : float
        Standard deviation of the Gaussian kernel in lattice units.

    Returns
    -------
    np.ndarray
        Smoothed field.
    """
    # ``gaussian_filter`` applies reflection boundary conditions by default,
    # which is acceptable for smoothing.  Ensure output is float64.
    return gaussian_filter(field.astype(np.float64), sigma=sigma, mode="reflect")


def gradient_magnitude(field: np.ndarray) -> np.ndarray:
    """Compute the magnitude of the gradient of ``field``.

    Parameters
    ----------
    field : np.ndarray
        Input 2‑D array.

    Returns
    -------
    np.ndarray
        Gradient magnitude at each lattice site.
    """
    dy, dx = np.gradient(field)  # returns derivatives along y (rows) and x (cols)
    return np.hypot(dx, dy)


def envelope(field: np.ndarray, sigma: float) -> tuple[np.ndarray, float]:
    """Apply smoothing and compute a normalised envelope for the FPHS field.

    The translator baseline proceeds as follows:

    1. Smooth the input field with a Gaussian of width ``sigma``.
    2. Compute the gradient magnitude of the smoothed field.
    3. Normalise the gradient magnitude by its root‑mean‑square (RMS)
       to produce a dimensionless envelope.

    Parameters
    ----------
    field : np.ndarray
        Raw electric field snapshot ``E0``.
    sigma : float
        Standard deviation for Gaussian smoothing.

    Returns
    -------
    env : np.ndarray
        The normalised envelope field ``N(|∇E0|)``.
    rms : float
        The RMS of the gradient magnitude used for normalisation.
    """
    smoothed = gaussian_smooth(field, sigma)
    grad_mag = gradient_magnitude(smoothed)
    # Compute RMS normalisation factor.  Avoid divide by zero by adding eps.
    rms = float(np.sqrt(np.mean(grad_mag**2)) + 1e-12)
    env = grad_mag / rms
    return env, rms