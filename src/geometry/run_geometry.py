"""Run the geometry baseline pipeline over a parameter grid.

This script reads a configuration file specifying the FPHS anchors (gauge,
lattice size ``L``, coupling ``b``, momentum scale ``κ``), measurement
fractions ``f``, seeds and translator parameters.  For each combination
of parameters it loads the corresponding ``E0`` snapshot, applies the
translator baseline (smooth, gradient, Poisson), computes radial
profiles and lensing fits, and writes per‑condition outputs.  At the
end it aggregates all per‑condition results into a summary CSV.

The code can optionally use synthetic placeholders for missing data by
setting the environment variable ``FPHS_GEOM_USE_PLACEHOLDER=1``.  This
is useful for verifying code correctness without real data, but the
resulting slopes have no physical meaning.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

try:
    # When imported as part of the geometry package, use relative imports.
    from .io import load_config, load_E0, hash_array, save_json, save_csv, save_plot
    from .translate import envelope
    from .poisson import compute_phi
    from .fit import fit_field
    from .optics import compute_deflection, fit_lensing
except ImportError:
    # Support running this file as a script.  Adjust sys.path to include the
    # parent of ``src/geometry`` and import modules using absolute names.
    import os as _os
    import sys as _sys
    _BASE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _BASE not in _sys.path:
        _sys.path.insert(0, _BASE)
    # Now import using the package name 'geometry'.
    from geometry.io import load_config, load_E0, hash_array, save_json, save_csv, save_plot
    from geometry.translate import envelope
    from geometry.poisson import compute_phi
    from geometry.fit import fit_field
    from geometry.optics import compute_deflection, fit_lensing


def run_condition(cfg: Dict[str, Any], anchor: Dict[str, Any], f: float, seed: int, data_dir: str, output_dir: str) -> Dict[str, Any]:
    """Run the geometry baseline for a single parameter combination.

    Parameters
    ----------
    cfg : dict
        Top‑level configuration loaded from YAML.
    anchor : dict
        Dictionary with keys ``gauge``, ``L``, ``b`` and ``kappa``.
    f : float
        Measurement fraction.
    seed : int
        Seed for random number generation.
    data_dir : str
        Directory containing ``E0`` snapshots.
    output_dir : str
        Root directory for saving results.

    Returns
    -------
    dict
        Summary dictionary with slopes, R² and metadata for the condition.
    """
    gauge = anchor["gauge"]
    L = int(anchor["L"])
    b_val = float(anchor["b"])
    kappa = float(anchor["kappa"])
    ell = cfg.get("ell", 4)
    # Load E0 snapshot (or placeholder).
    use_placeholder = bool(os.environ.get("FPHS_GEOM_USE_PLACEHOLDER"))
    E0 = load_E0(gauge, L, b_val, kappa, f, seed, data_dir, use_placeholder=use_placeholder)
    E0_hash = hash_array(E0)
    # Apply translator: smoothing + gradient + RMS normalisation.
    env, env_rms = envelope(E0, sigma=ell)
    # Solve Poisson to get phi and gradient magnitude.
    poisson_cfg = cfg.get("poisson", {})
    pad_factor = int(poisson_cfg.get("pad_factor", 4))
    eps_factor = float(poisson_cfg.get("epsilon_factor", 0.5))
    phi, grad_phi_mag = compute_phi(env, ell=ell, pad_factor=pad_factor, epsilon_factor=eps_factor)
    # Compute median normalisation for |∇Φ|.
    med_grad = float(np.median(grad_phi_mag)) + 1e-12
    grad_phi_norm = grad_phi_mag / med_grad
    # Build radial configuration.
    radial_cfg = cfg.get("radial", {})
    # Fit raw phi.
    phi_fit = fit_field(phi, ell=ell, radial_cfg=radial_cfg)
    # Fit raw |∇Φ|.
    grad_fit = fit_field(grad_phi_mag, ell=ell, radial_cfg=radial_cfg)
    # Fit normalised |∇Φ|.
    grad_norm_fit = fit_field(grad_phi_norm, ell=ell, radial_cfg=radial_cfg)
    # Lensing analysis for each lambda.
    optics_cfg = cfg.get("optics", {})
    lambda_list = optics_cfg.get("lambda_list", [])
    b_start = float(optics_cfg.get("impact_b_start", 8))
    b_end = float(optics_cfg.get("impact_b_end", 96))
    n_points = int(optics_cfg.get("impact_points", 32))
    window_fraction = float(optics_cfg.get("impact_window_fraction", 0.6))
    # Build impact parameter grid relative to lattice centre.  Use positive b values only.
    b_values = np.geomspace(b_start, b_end, num=n_points)
    # Precompute deflections for each lambda and store tables.
    lensing_results = []
    lensing_table_rows: List[Dict[str, Any]] = []
    for lam in lambda_list:
        # Index field n(x) = 1 + lam * env (env already normalised).  In the thin lens approximation the deflection scales linearly with lam, so we can compute deflection once using grad_phi_mag and scale by lam.
        # Here we compute deflection from phi directly to capture the potential's shape; scaling lam is equivalent to scaling env.
        alpha = compute_deflection(phi, b_values)
        # Scale alpha by lam (linearity of index perturbation).  This yields deflection angles for index n(x) = 1 + lam * env.
        alpha_scaled = lam * alpha
        slope, intercept, r2 = fit_lensing(b_values, alpha_scaled, window_fraction=window_fraction)
        lensing_results.append({"lambda": lam, "slope": slope, "intercept": intercept, "r2": r2})
        # Record table rows for CSV: one per b.
        for bval, aval in zip(b_values, alpha_scaled):
            lensing_table_rows.append({"lambda": lam, "b": bval, "alpha": aval})
    # Save radial profiles as CSV.  Use separate files for raw and normalised |∇Φ|.
    radial_dir = os.path.join(output_dir, "radial")
    cond_prefix = f"{gauge}_L{L}_b{b_val}_k{kappa}_f{f}_seed{seed}"
    # Raw phi radial table.
    phi_df = pd.DataFrame({
        "r": phi_fit["r_centres"],
        "phi": phi_fit["means"],
        "phi_var": phi_fit["variances"],
    })
    save_csv(phi_df, os.path.join(radial_dir, f"radial_phi_{cond_prefix}.csv"))
    # Raw grad radial table.
    grad_df = pd.DataFrame({
        "r": grad_fit["r_centres"],
        "grad_phi": grad_fit["means"],
        "grad_phi_var": grad_fit["variances"],
    })
    save_csv(grad_df, os.path.join(radial_dir, f"radial_grad_{cond_prefix}.csv"))
    # Normalised grad radial table.
    grad_norm_df = pd.DataFrame({
        "r": grad_norm_fit["r_centres"],
        "grad_phi_norm": grad_norm_fit["means"],
        "grad_phi_norm_var": grad_norm_fit["variances"],
    })
    save_csv(grad_norm_df, os.path.join(radial_dir, f"radial_grad_norm_{cond_prefix}.csv"))
    # Save lensing table.
    lens_dir = os.path.join(output_dir, "lensing")
    lens_df = pd.DataFrame(lensing_table_rows)
    save_csv(lens_df, os.path.join(lens_dir, f"lensing_{cond_prefix}.csv"))
    # Save a simple radial plot (phi and grad).  Create figure.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(phi_df["r"], phi_df["phi"], marker="o")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_title("Φ(r) radial profile")
    ax[0].set_xlabel("r")
    ax[0].set_ylabel("Φ(r)")
    ax[1].plot(grad_df["r"], grad_df["grad_phi"], marker="o")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_title("|∇Φ|(r) radial profile")
    ax[1].set_xlabel("r")
    ax[1].set_ylabel("|∇Φ|(r)")
    save_plot(fig, os.path.join(output_dir, "plots", f"radial_{cond_prefix}.png"))
    # Save lensing plot: alpha vs 1/b for each lambda.
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x_inv_b = 1.0 / b_values
    for lam in lambda_list:
        mask = lens_df["lambda"] == lam
        alpha_vals = lens_df.loc[mask, "alpha"].values
        ax2.plot(x_inv_b, alpha_vals, marker=".", label=f"λ={lam}")
    ax2.set_xlabel("1/b")
    ax2.set_ylabel("α(b)")
    ax2.set_title("Lensing deflection vs 1/b")
    ax2.legend()
    save_plot(fig2, os.path.join(output_dir, "plots", f"lensing_{cond_prefix}.png"))
    # Assemble summary dictionary.
    summary = {
        "gauge": gauge,
        "L": L,
        "b": b_val,
        "kappa": kappa,
        "f": f,
        "seed": seed,
        "ell": ell,
        "env_rms": env_rms,
        "E0_hash": E0_hash,
        "phi_slopes": {
            "s_phi": phi_fit["slope"],
            "r2_phi": phi_fit["r2"],
        },
        "grad_slopes": {
            "s_grad": grad_fit["slope"],
            "r2_grad": grad_fit["r2"],
        },
        "grad_norm_slopes": {
            "s_grad_norm": grad_norm_fit["slope"],
            "r2_grad_norm": grad_norm_fit["r2"],
        },
        "median_grad": med_grad,
        "pad_factor": pad_factor,
        "epsilon_factor": eps_factor,
        "lensing": lensing_results,
    }
    # Save per‑condition JSON.
    runs_dir = os.path.join(output_dir, "runs")
    save_json(summary, os.path.join(runs_dir, f"summary_{cond_prefix}.json"))
    return summary


def run_all(config_path: str = None, data_dir: str = "data/inputs", output_dir: str = "runs") -> None:
    """Run geometry baseline for all anchors, f values and seeds.

    This function is suitable for a command‑line entry point.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file.  Defaults to
        ``configs/anchors.yaml`` relative to the repository root.
    data_dir : str, optional
        Directory containing input snapshots ``E0``.  Defaults to
        ``data/inputs``.
    output_dir : str, optional
        Directory to store outputs.  Defaults to ``runs``.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "anchors.yaml")
    cfg = load_config(config_path)
    anchors: List[Dict[str, Any]] = cfg.get("anchors", [])
    f_values: List[float] = cfg.get("f_values", [])
    seeds: List[int] = cfg.get("seeds", [])
    # Ensure output subdirectories exist.
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "radial"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lensing"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "runs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    # List to accumulate all summaries.
    summaries: List[Dict[str, Any]] = []
    # Iterate over anchors, f and seeds.
    for anchor in anchors:
        for f in f_values:
            for seed in seeds:
                summary = run_condition(cfg, anchor, f, seed, data_dir, output_dir)
                summaries.append(summary)
    # Aggregate summaries into a single CSV.
    rows = []
    for summ in summaries:
        row = {
            "gauge": summ["gauge"],
            "L": summ["L"],
            "b": summ["b"],
            "kappa": summ["kappa"],
            "f": summ["f"],
            "seed": summ["seed"],
            "ell": summ["ell"],
            "s_phi": summ["phi_slopes"]["s_phi"],
            "r2_phi": summ["phi_slopes"]["r2_phi"],
            "s_grad": summ["grad_slopes"]["s_grad"],
            "r2_grad": summ["grad_slopes"]["r2_grad"],
            "s_grad_norm": summ["grad_norm_slopes"]["s_grad_norm"],
            "r2_grad_norm": summ["grad_norm_slopes"]["r2_grad_norm"],
        }
        # For each lambda in lensing results, record slope and r2 with key suffix.
        for lens in summ["lensing"]:
            lam = lens["lambda"]
            row[f"alpha_slope_lambda{lam}"] = lens["slope"]
            row[f"alpha_r2_lambda{lam}"] = lens["r2"]
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    save_csv(summary_df, os.path.join(output_dir, "summary.csv"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run geometry baseline analysis on FPHS data.")
    parser.add_argument("--config", type=str, default=None, help="Path to anchors configuration YAML.")
    parser.add_argument("--data-dir", type=str, default="data/inputs", help="Directory containing input E0 snapshots.")
    parser.add_argument("--out-dir", type=str, default="runs", help="Directory to store outputs.")
    args = parser.parse_args()
    run_all(config_path=args.config, data_dir=args.data_dir, output_dir=args.out_dir)