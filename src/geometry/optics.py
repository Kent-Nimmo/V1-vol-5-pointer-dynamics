"""Optics (thin‑lens) analysis for the geometry baseline.

This module computes deflection angles for a set of impact parameters
based on the geometry potential ``Φ``.  In the small‑angle (thin lens)
approximation, the deflection ``α(b)`` is proportional to the gradient
of ``Φ`` integrated along the line of sight.  We then perform a
weighted linear regression of ``α(b)`` versus ``1/b`` to extract a
power‑law behaviour.

The implementation here is deliberately simple and is intended for
relative comparisons rather than absolute lensing predictions.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple


def compute_deflection(phi: np.ndarray, b_values: np.ndarray) -> np.ndarray:
    """Compute deflection angles ``α(b)`` for a set of impact parameters.

    Parameters
    ----------
    phi : np.ndarray
        Geometry potential ``Φ`` on an ``L x L`` grid.
    b_values : np.ndarray
        Array of impact parameters (in lattice units) for which to compute
        the deflection.  Values should be within the range of the grid.

    Returns
    -------
    alpha : np.ndarray
        Deflection angles corresponding to each impact parameter.  The
        units are arbitrary but consistent across conditions.
    """
    L = phi.shape[0]
    assert phi.shape[0] == phi.shape[1], "phi must be square"
    # Compute gradient of phi.  gradient returns [dy, dx], i.e., y‑component and x‑component of gradient.
    dy, dx = np.gradient(phi)
    # Centre coordinates for mapping physical impact parameter to array indices.
    cy = (L - 1) / 2.0
    alpha = []
    # Integrate derivative along x direction for each impact parameter.
    for b in b_values:
        # Convert continuous impact parameter b to a floating row index.  Positive b corresponds to y > cy; negative b to y < cy.
        row_float = cy + b
        # Determine integer indices for interpolation.  Use nearest neighbour if out of bounds.
        if row_float < 0 or row_float > L - 1:
            alpha.append(np.nan)
            continue
        row_low = int(np.floor(row_float))
        row_high = min(row_low + 1, L - 1)
        t = row_float - row_low
        # Interpolate the y‑gradient between row_low and row_high.
        grad_row = (1 - t) * dy[row_low, :] + t * dy[row_high, :]
        # Integrate along x across the entire width.  For a thin lens at z=0, the deflection is twice the integral over positive x.  However, since the grid is finite, we integrate over all x and multiply by dx.
        dx_pix = 1.0
        # Use Simpson's rule or simple sum.  Here we do simple sum for speed.
        integral = np.sum(grad_row) * dx_pix
        # Multiply by 2 for two symmetric halves.
        alpha.append(2.0 * integral)
    return np.array(alpha, dtype=np.float64)


def fit_lensing(b_values: np.ndarray, alpha: np.ndarray, window_fraction: float = 0.6) -> Tuple[float, float, float]:
    """Fit the relationship ``α(b) ≈ slope * (1/b) + intercept`` using weighted LS.

    Parameters
    ----------
    b_values : np.ndarray
        Impact parameters used to compute ``α(b)``.
    alpha : np.ndarray
        Deflection angles corresponding to ``b_values``.
    window_fraction : float, optional
        Fraction of the central region of the data to include in the fit.
        Must be between 0 and 1.  Default is 0.6 (central 60 %).

    Returns
    -------
    slope : float
        Slope of the linear fit ``α(b)`` versus ``1/b``.
    intercept : float
        Intercept of the fit.
    r2 : float
        Coefficient of determination ``R^2`` for the fit.
    """
    # Filter out NaN values.
    mask = np.isfinite(alpha) & np.isfinite(b_values) & (b_values > 0)
    b_filtered = b_values[mask]
    alpha_filtered = alpha[mask]
    if b_filtered.size < 2:
        return np.nan, np.nan, np.nan
    # Sort by b to apply window.
    order = np.argsort(b_filtered)
    b_sorted = b_filtered[order]
    alpha_sorted = alpha_filtered[order]
    n = b_sorted.size
    # Determine window indices.
    m = max(int(n * window_fraction), 2)
    start = (n - m) // 2
    end = start + m
    b_win = b_sorted[start:end]
    alpha_win = alpha_sorted[start:end]
    # Independent variable x = 1/b.
    x = 1.0 / b_win
    y = alpha_win
    # Perform unweighted linear regression y = slope * x + intercept.
    # Compute means.
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # Compute slope and intercept.
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    slope = ss_xy / ss_xx if ss_xx > 0 else np.nan
    intercept = y_mean - slope * x_mean
    # Compute R^2.
    y_pred = slope * x + intercept
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2