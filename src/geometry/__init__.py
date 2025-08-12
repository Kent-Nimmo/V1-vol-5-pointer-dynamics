"""Top-level package for the geometry baseline pipeline.

This package implements the baseline translator (gradient‑as‑mass) and the
analysis routines for computing radial and lensing fits on FPHS geometry
snapshots.  See the README for high‑level usage.
"""

__all__ = [
    "translate",
    "poisson",
    "radial",
    "fit",
    "optics",
    "io",
    "run_geometry",
]