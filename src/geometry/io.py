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
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hash_array(arr: np.ndarray) -> str:
    """Compute a SHA-256 hash for an array, including shape and dtype."""
    hasher = hashlib.sha256()
    hasher.update(arr.tobytes())
    hasher.update(str(arr.shape).encode())
    hasher.update(str(arr.dtype).encode())
    return hasher.hexdigest()


def load_E0(
    gauge: str, L: int, b: float, kappa: float, f: float, seed: int,
    data_dir: str, use_placeholder: bool = False
) -> np.ndarray:
    """Load an E0 snapshot from disk or generate a placeholder.

    If the flat filename is not found, also try the nested layout:
    data/inputs/<gauge>/L<L>/b<b>/k<kappa>/f<f>/seed<seed>/E0.npz
    """
    # Construct flat filename pattern, e.g.:
    # E0_SU3_L256_b3.5_k1.0_f0.10_seed2.npz
    f_str = f"{f:.2f}"
    fname = f"E0_{gauge}_L{L}_b{b}_k{kappa}_f{f_str}_seed{seed}.npz"
    path = os.path.join(data_dir, fname)

    # If the flat filename isn't present, also try the nested layout shipped in this repo
    if not os.path.exists(path):
        nested = os.path.join(
            data_dir,
            gauge,
            f"L{L}",
            f"b{b}",
            f"k{kappa:.2f}",
            f"f{f_str}",
            f"seed{seed}",
            "E0.npz",
        )
        if os.path.exists(nested):
            path = nested

    if os.path.exists(path):
        data = np.load(path)
        if "E0" not in data:
            raise KeyError(f"File {path} does not contain 'E0' array")
        arr = data["E0"].astype(np.float64)
        if arr.shape != (L, L):
            raise ValueError(f"E0 array shape mismatch: expected {(L, L)}, found {arr.shape}")
        return arr

    # If file not found, optionally generate a placeholder
    if use_placeholder or os.environ.get("FPHS_GEOM_USE_PLACEHOLDER") == "1":
        rng = np.random.default_rng(seed)
        return rng.standard_normal((L, L), dtype=np.float64)

    # Otherwise error out
    raise FileNotFoundError(
        f"Snapshot {fname} not found in {data_dir} or nested layout under "
        f"{os.path.join(data_dir, gauge, f'L{L}', f'b{b}', f'k{kappa:.2f}', f'f{f_str}', f'seed{seed}', 'E0.npz')}"
    )


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_plot(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
