"""I/O utilities for the geometry baseline pipeline.

This module centralises the logic for loading FPHS field snapshots,
constructing synthetic placeholders when needed, computing hashes of
input arrays, and writing results to disk.  It also provides helpers
for reading the YAML configuration file.
"""

from __future__ import annotations

import os
import json
import hashlib
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration as a nested dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hash_array(arr: np.ndarray) -> str:
    """Compute a SHA‑256 hash for an array, including shape and dtype.

    The hash is computed over the raw bytes of the array concatenated
    with its shape and dtype to ensure uniqueness across arrays of
    different dimensions or types.

    Parameters
    ----------
    arr : np.ndarray
        Array to hash.

    Returns
    -------
    str
        Hex digest of the SHA‑256 hash.
    """
    hasher = hashlib.sha256()
    hasher.update(arr.tobytes())
    # Include shape and dtype to avoid collisions.
    hasher.update(str(arr.shape).encode())
    hasher.update(str(arr.dtype).encode())
    return hasher.hexdigest()


def load_E0(gauge: str, L: int, b: float, kappa: float, f: float, seed: int, data_dir: str, use_placeholder: bool = False) -> np.ndarray:
    """Load an ``E0`` snapshot from disk or generate a placeholder.

    Parameters
    ----------
    gauge : str
        Gauge group, e.g., ``SU2`` or ``SU3``.
    L : int
        Lattice size.
    b : float
        Coupling constant.
    kappa : float
        Momentum scale.
    f : float
        Measurement fraction.
    seed : int
        Random seed for reproducibility.
    data_dir : str
        Directory where snapshot files are stored.
    use_placeholder : bool, optional
        If ``True`` and no snapshot file is found, generate a Gaussian
        random field as a placeholder.  Otherwise, raise a
        ``FileNotFoundError`` if the file is missing.  Default is
        ``False``.

    Returns
    -------
    np.ndarray
        2‑D electric field array ``E0`` of shape ``(L, L)``.
    """
    # Construct filename pattern.  Use underscores to separate metadata.
    # Example: E0_SU3_L256_b3.5_k1.0_f0.10_seed2.npz
    f_str = f"{f:.2f}"
    fname = f"E0_{gauge}_L{L}_b{b}_k{kappa}_f{f_str}_seed{seed}.npz"
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        data = np.load(path)
        if "E0" not in data:
            raise KeyError(f"File {path} does not contain 'E0' array")
        arr = data["E0"].astype(np.float64)
        # Verify shape.
        if arr.shape != (L, L):
            raise ValueError(f"E0 array shape mismatch: expected {(L, L)}, found {arr.shape}")
        return arr
    # If file not found.
    if use_placeholder or os.environ.get("FPHS_GEOM_USE_PLACEHOLDER") == "1":
        rng = np.random.default_rng(seed)
        # Generate a Gaussian random field with zero mean and unit variance.
        arr = rng.standard_normal((L, L), dtype=np.float64)
        # Optionally apply a smoothing to make it less noisy.
        return arr
    # Otherwise raise.
    raise FileNotFoundError(f"Snapshot {fname} not found in {data_dir}")


def save_json(obj: Dict[str, Any], path: str) -> None:
    """Write a dictionary to a JSON file with indentation and sorted keys."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save a Pandas DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_plot(fig: plt.Figure, path: str) -> None:
    """Save a Matplotlib figure to a PNG file and close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)