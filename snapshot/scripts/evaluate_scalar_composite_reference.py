"""Replay the bounded scalar control. Emits numerical evidence, not a fit.

Run with PYTHONPATH=src; --output creates a NEW JSON file and never overwrites.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from deuteron_wigner.quantum.scalar_composite_reference import (
    charge_radius_squared, longitudinal_form_factor,
    overlap_form_factor, triangle_form_factor,
)


def evaluate():
    start = time.perf_counter()
    rows = []
    for mass in (1.0, 1.5, 1.9):
        for q2 in (0.0, 0.25, 1.0, 4.0):
            for order in (32, 64, 128):
                cov = triangle_form_factor(mass, q2, order)
                overlap = overlap_form_factor(mass, q2, order)
                longitudinal = longitudinal_form_factor(mass, q2, order)
                rows.append({
                    "M_over_m": mass, "Q_squared_over_m_squared": q2, "order": order,
                    "covariant_F": cov, "DY_overlap_F": overlap,
                    "beta": longitudinal.beta,
                    "longitudinal_valence_F": longitudinal.valence,
                    "longitudinal_pair_F": longitudinal.pair,
                    "longitudinal_total_F": longitudinal.total,
                    "overlap_absolute_error": abs(overlap - cov),
                    "longitudinal_absolute_error": abs(longitudinal.total - cov),
                    "omitted_pair_relative_error": abs(longitudinal.valence / cov - 1),
                })
    fine = [r for r in rows if r["order"] == 128]
    radial_error = 0.0
    radius_checks = []
    for mass in (1.0, 1.5, 1.9):
        for q2 in (0.0, 0.25, 1.0, 4.0):
            cov = triangle_form_factor(mass, q2, 128)
            for scale in (0.5, 2.0):
                radial_error = max(radial_error,
                    abs(overlap_form_factor(mass, q2, 128, scale) - cov),
                    abs(longitudinal_form_factor(mass, q2, 128, scale).total - cov))
        h = 1e-5
        f0, f1, f2 = [triangle_form_factor(mass, q2, 128) for q2 in (0, h, 2*h)]
        derivative_radius = -6 * (-3*f0 + 4*f1 - f2) / (2*h)
        radius_checks.append({"M_over_m": mass,
            "radius_squared_times_m_squared": charge_radius_squared(mass),
            "finite_difference_radius": derivative_radius,
            "absolute_error": abs(derivative_radius - charge_radius_squared(mass))})
    refinement = max(abs(rows[i][key] - rows[i+1][key])
        for i in range(len(rows)) if rows[i]["order"] == 64
        for key in ("covariant_F", "DY_overlap_F", "longitudinal_total_F"))
    summary = {
        "max_fine_overlap_absolute_error": max(r["overlap_absolute_error"] for r in fine),
        "max_fine_longitudinal_absolute_error": max(r["longitudinal_absolute_error"] for r in fine),
        "max_64_to_128_change": refinement,
        "max_radial_map_absolute_error": radial_error,
        "max_radius_derivative_absolute_error": max(r["absolute_error"] for r in radius_checks),
        "max_nonzero_transfer_omitted_pair_relative_error": max(r["omitted_pair_relative_error"] for r in fine),
    }
    summary["numerical_targets_met"] = (
        max(summary[k] for k in (
            "max_fine_overlap_absolute_error", "max_fine_longitudinal_absolute_error",
            "max_64_to_128_change", "max_radial_map_absolute_error")) < 1e-8
        and summary["max_radius_derivative_absolute_error"] < 2e-7)
    return {
        "purpose": "scalar reference feasibility and missing-pair negative control; no fit",
        "source": "https://arxiv.org/abs/hep-ph/9906225",
        "scope": "equal-mass constant-vertex scalar triangle, J+ in two spacelike frames",
        "normalization": "one continuous I0; no separate frame or numerical renormalization",
        "physical_nucleon_claim": False, "all_component_covariance_test": False,
        "network_or_learned_dynamics_implemented": False,
        "numerical_absolute_F_target": 1e-8,
        "runtime_seconds_excluding_imports": time.perf_counter() - start,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "summary": summary, "radius_checks": radius_checks, "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate()
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(json.dumps(report["summary"], indent=2))
        print(f"Numerical evaluation seconds: {report['runtime_seconds_excluding_imports']:.3f}")
    else:
        print(encoded, end="")
    if not report["summary"]["numerical_targets_met"]:
        raise SystemExit("Numerical target failed: reassess before using the reference.")


if __name__ == "__main__":
    main()
