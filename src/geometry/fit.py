"""Simple wrappers around radial profile fitting.

This module exposes convenience functions that call into
``geometry.radial`` to perform power‑law fits on radial profiles.  It
exists primarily to maintain a clear separation of responsibilities in
``run_geometry.py``.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any
from .radial import radialize, fit_power_law


def fit_field(field: np.ndarray, ell: float, radial_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute radial profiles and power‑law fits for a field.

    Parameters
    ----------
    field : np.ndarray
        Input 2‑D array (either ``Φ`` or ``|∇Φ|``).
    ell : float
        Smoothing length used to define the inner radius.
    radial_cfg : dict
        Configuration dictionary with keys:
        ``num_bins``, ``r_min_factor``, ``r_max_fraction``, and ``min_pixels``.

    Returns
    -------
    result : dict
        Contains arrays ``r_centres``, ``means``, ``variances`` and fit
        parameters ``slope``, ``intercept``, ``r2``.  If the fit failed,
        slopes and r2 may be NaN.
    """
    r_centres, means, variances = radialize(
        field,
        ell=ell,
        num_bins=radial_cfg.get("num_bins", 36),
        r_min_factor=radial_cfg.get("r_min_factor", 3.0),
        r_max_fraction=radial_cfg.get("r_max_fraction", 0.3),
        min_pixels=radial_cfg.get("min_pixels", 50),
    )
    slope, intercept, r2 = fit_power_law(r_centres, means, variances)
    return {
        "r_centres": r_centres,
        "means": means,
        "variances": variances,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
    }