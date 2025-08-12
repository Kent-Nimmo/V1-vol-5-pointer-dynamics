"""Radial profiles and fitting for the geometry baseline.

This module provides functions to bin 2‑D fields into radial shells
(logarithmic rings) and compute statistics per shell.  It also offers a
weighted least‑squares fit in log–log space to extract power‑law
behaviour.  Rings with too few pixels can be excluded from the fit.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List

def radialize(field: np.ndarray, ell: float, num_bins: int, r_min_factor: float, r_max_fraction: float, min_pixels: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute radial profiles of a 2‑D field using logarithmic bins.

    Parameters
    ----------
    field : np.ndarray
        2‑D array ``Φ`` or ``|∇Φ|`` of shape ``(L, L)``.
    ell : float
        Smoothing length used in the translator.  Determines the inner
        cutoff ``r_min = r_min_factor * ell``.
    num_bins : int
        Number of radial rings (logarithmic bins) to use.
    r_min_factor : float
        Factor multiplied by ``ell`` to obtain the minimum radius ``r_min``.
    r_max_fraction : float
        Fraction of the lattice size used to set ``r_max = r_max_fraction * L``.
    min_pixels : int, optional
        Minimum number of samples per ring required to include it in the
        returned profiles.  Default is 50.

    Returns
    -------
    r_centres : np.ndarray
        Centre radius of each retained ring.
    means : np.ndarray
        Mean field value in each retained ring.
    variances : np.ndarray
        Variance of the field values in each retained ring.
    """
    L = field.shape[0]
    assert field.shape[0] == field.shape[1], "field must be square"
    # Coordinates relative to centre (assume centre at (L/2, L/2)).
    y_indices, x_indices = np.indices((L, L))
    cy = (L - 1) / 2.0
    cx = (L - 1) / 2.0
    # Euclidean distance from centre.
    radii = np.sqrt((y_indices - cy) ** 2 + (x_indices - cx) ** 2)
    # Define radial range.
    r_min = r_min_factor * float(ell)
    r_max = r_max_fraction * float(L)
    if r_min <= 0:
        r_min = 1e-6
    # Logarithmic bin edges.
    # Avoid including r=0 by starting at r_min.
    bin_edges = np.logspace(np.log10(r_min), np.log10(r_max), num_bins + 1)
    # Assign each pixel to a bin.
    # Use np.digitize; indices from 1 to num_bins inclusive, 0 for values < first edge.
    bin_indices = np.digitize(radii.flatten(), bin_edges) - 1
    # Prepare arrays to accumulate sums and counts.
    sums = np.zeros(num_bins, dtype=np.float64)
    sq_sums = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    # Flatten field for easier indexing.
    flat_field = field.flatten().astype(np.float64)
    for idx, val in zip(bin_indices, flat_field):
        if idx < 0 or idx >= num_bins:
            continue
        sums[idx] += val
        sq_sums[idx] += val * val
        counts[idx] += 1
    # Compute means, variances, and bin centres.
    r_centres = []
    means = []
    variances = []
    for i in range(num_bins):
        if counts[i] < min_pixels:
            continue
        mean = sums[i] / counts[i]
        var = max(sq_sums[i] / counts[i] - mean * mean, 0.0)
        # Bin centre as geometric mean of edges.
        r_c = np.sqrt(bin_edges[i] * bin_edges[i + 1])
        r_centres.append(r_c)
        means.append(mean)
        variances.append(var)
    return np.array(r_centres), np.array(means), np.array(variances)


def fit_power_law(r: np.ndarray, y: np.ndarray, variances: np.ndarray) -> Tuple[float, float, float]:
    """Fit a power law ``y ∝ r^s`` using weighted least squares in log–log space.

    The regression is performed on ``log10(y) = s log10(r) + c`` with weights
    ``w = 1/variance``.  Rings with non‑positive values of ``y`` or
    non‑positive variances are excluded from the fit.

    Parameters
    ----------
    r : np.ndarray
        Radii of the rings.
    y : np.ndarray
        Mean field values for each ring.
    variances : np.ndarray
        Variance of field values for each ring.

    Returns
    -------
    slope : float
        Estimated power‑law exponent ``s``.
    intercept : float
        Intercept ``c`` in log–log space.
    r2 : float
        Coefficient of determination ``R^2`` measuring goodness of fit.
    """
    # Filter out non‑positive y or variances.
    mask = (r > 0) & (y > 0) & (variances > 0)
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan, np.nan
    r_log = np.log10(r[mask])
    y_log = np.log10(y[mask])
    w = 1.0 / variances[mask]
    # Weighted least squares: slope and intercept in log space.
    # Compute weighted averages.
    W = np.sum(w)
    xw_mean = np.sum(w * r_log) / W
    yw_mean = np.sum(w * y_log) / W
    # Weighted covariance and variance.
    cov = np.sum(w * (r_log - xw_mean) * (y_log - yw_mean)) / W
    var = np.sum(w * (r_log - xw_mean) ** 2) / W
    slope = cov / var if var > 0 else np.nan
    intercept = yw_mean - slope * xw_mean
    # Compute R^2.
    y_pred = slope * r_log + intercept
    ss_tot = np.sum(w * (y_log - yw_mean) ** 2)
    ss_res = np.sum(w * (y_log - y_pred) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2