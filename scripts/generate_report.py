#!/usr/bin/env python
"""
Generate a Markdown report summarising geometry baseline results.

This script reads the aggregated summary CSV produced by
``aggregate.py`` and writes a simple Markdown report highlighting
statistics of the radial and lensing fits.  The report is intended as
a human‐readable overview rather than an exhaustive record.
"""

import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate report for geometry baseline results.")
    parser.add_argument("--summary", type=str, required=True, help="Path to summary CSV file.")
    parser.add_argument("--out-report", type=str, required=True, help="Path to write Markdown report.")
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    # Compute mean and standard deviation of slopes grouped by gauge and f.
    group_cols = ["gauge", "b", "kappa", "f"]
    agg = df.groupby(group_cols).agg({
        "s_phi": ["mean", "std"],
        "r2_phi": ["mean"],
        "s_grad": ["mean", "std"],
        "r2_grad": ["mean"],
        "s_grad_norm": ["mean", "std"],
        "r2_grad_norm": ["mean"],
    })
    # Flatten multiindex columns.
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]
    # Start writing report.
    lines = []
    lines.append("# Geometry Baseline Report")
    lines.append("")
    lines.append("This report summarises the results of the geometry baseline translator applied to the available FPHS snapshots.")
    lines.append("")
    lines.append("## Radial fit statistics")
    lines.append("")
    lines.append("Mean and standard deviation of radial slopes (Φ, |∇Φ|, and normalised |∇Φ|) grouped by gauge, b, κ and f.")
    lines.append("")
    # Convert agg to markdown table.
    lines.append(agg.reset_index().to_string(index=False))
    lines.append("")
    # Lensing slopes summary per lambda.
    lam_cols = [col for col in df.columns if col.startswith("alpha_slope_lambda")]
    if lam_cols:
        lines.append("## Lensing fit statistics")
        lines.append("")
        lam_data = []
        for lam_col in lam_cols:
            lam_val = lam_col.split("lambda")[-1]
            mean_slope = df[lam_col].mean()
            std_slope = df[lam_col].std()
            r2_col = lam_col.replace("slope", "r2")
            mean_r2 = df[r2_col].mean() if r2_col in df.columns else np.nan
            lam_data.append({"lambda": lam_val, "slope_mean": mean_slope, "slope_std": std_slope, "r2_mean": mean_r2})
        lam_df = pd.DataFrame(lam_data)
        lines.append(lam_df.to_string(index=False))
        lines.append("")
    # Write to file.
    # Ensure destination directory exists.
    import os
    out_dir = os.path.dirname(args.out_report)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {args.out_report}")

if __name__ == "__main__":
    main()