"""Bounded joint state/current rank study, with untouched kinematic holdouts.

Rank is selected on the development grid only. No observable is fitted and
no component is renormalized. --output creates a new evidence file.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from deuteron_wigner.quantum.scalar_composite_reference import (
    longitudinal_form_factor, triangle_form_factor,
)
from deuteron_wigner.quantum.scalar_joint_representation import (
    A_MIN, A_MAX, ExactCore, build_core, dy_current, longitudinal_current, state_samples,
)


DEVELOPMENT = [(m,q) for m in (1.0,1.5,1.9) for q in (0.0,.25,1.,4.)]
HOLDOUT = [(m,q) for m in (1.,1.5,1.9) for q in (.0625,.5625,2.25)] + [
    (1.7,q) for q in (0.,.0625,.25,.5625,1.,2.25,4.)]
DEGREE = 56
RANKS = (1,2,3,4,5,6,8,12)


def compare(core, grid, order=64):
    rows = []
    states = {}
    for mass in sorted(set(m for m,_ in grid)):
        b = state_samples(core,mass,order)
        reference = state_samples(ExactCore(),mass,order)
        states[mass] = {"state_L2_error": float(np.linalg.norm(b-reference)),
                        "norm_error": float(abs(np.linalg.norm(b)**2-1))}
    for mass,q2 in grid:
        f = triangle_form_factor(mass,q2,128)
        lp = longitudinal_form_factor(mass,q2,128)
        dy = dy_current(core,mass,q2,order)
        represented = longitudinal_current(core,mass,q2,order)
        rows.append({"M_over_m": mass, "Q2_over_m2": q2,
            **states[mass], "reference_F": f, "DY_F": float(dy.real),
            "longitudinal_valence": float(represented.valence.real),
            "longitudinal_pair": float(represented.pair.real),
            "longitudinal_total": float(represented.total.real),
            "DY_error": float(abs(dy-f)),
            "longitudinal_error": float(abs(represented.total-f)),
            "valence_error": float(abs(represented.valence-lp.valence)),
            "pair_error": float(abs(represented.pair-lp.pair)),
            "frame_difference": float(abs(dy-represented.total)),
            "imaginary_residual": float(max(abs(dy.imag),abs(represented.total.imag)))})
    maxima = {key: max(row[key] for row in rows) for key in (
        "state_L2_error","norm_error","DY_error","longitudinal_error",
        "valence_error","pair_error","frame_difference","imaginary_residual")}
    # Include separately retained terms and the frame difference, not just totals.
    maxima["targets_met"] = maxima["state_L2_error"] <= 1e-3 and max(
        maxima[key] for key in ("norm_error","DY_error","longitudinal_error",
                               "valence_error","pair_error","frame_difference")) <= 1e-4
    return {"maxima":maxima,"rows":rows}


def _vectors(core, grid, order):
    values = []
    for m,q in grid:
        lp = longitudinal_current(core,m,q,order)
        values.extend((dy_current(core,m,q,order),lp.valence,lp.pair))
    return np.asarray(values)


def evaluate():
    start = time.perf_counter()
    interpolation = []
    for degree in (24,40,56):
        report = compare(build_core(degree),DEVELOPMENT)
        interpolation.append({"degree":degree,"maxima":report["maxima"]})
    curves = []
    selected = None
    for rank in RANKS:
        core = build_core(DEGREE,rank)
        report = compare(core,DEVELOPMENT)
        curves.append({"rank":rank,"stored_coefficients":core.stored_coefficients,
                       "stored_bytes":core.stored_bytes,**report})
        if selected is None and report["maxima"]["targets_met"]:
            selected = rank
    if selected is None:
        raise ArithmeticError("No tested rank meets development targets; reassess, do not tune holdouts.")
    # Freeze rank BEFORE evaluating any held-out mass or transfer values.
    compressed = build_core(DEGREE,selected)
    dense = build_core(DEGREE)
    holdout = compare(compressed,HOLDOUT)
    dense_holdout = compare(dense,HOLDOUT)
    grid = DEVELOPMENT + HOLDOUT
    refinement = {}
    for name,core in (("exact",ExactCore()),("dense",dense),("selected_rank",compressed)):
        low = _vectors(core,grid,64)
        high = _vectors(core,grid,96)
        refinement[name] = float(np.max(np.abs(low-high)))
    phase = lambda a,s: 0.8*a + 0.6*s/(1+s)
    phase_checks = {}
    for name,core in (("dense",dense),("selected_rank",compressed)):
        base = _vectors(core,[(1.5,1.)],64)
        matching,wrong = [],[]
        for transport,target in ((True,matching),(False,wrong)):
            lp = longitudinal_current(core,1.5,1.,64,phase,transport)
            target.extend((dy_current(core,1.5,1.,64,phase,transport),lp.valence,lp.pair))
        phase_checks[name] = {
            "matched_state_and_operator_error":float(np.max(np.abs(np.asarray(matching)-base))),
            "state_only_rephase_error":float(np.max(np.abs(np.asarray(wrong)-base)))}
    timings = {}
    for name,core in (("exact_formula",ExactCore()),("dense_interpolant",dense),
                      ("selected_rank",compressed)):
        samples = []
        for _ in range(3):
            tick = time.perf_counter()
            _vectors(core,DEVELOPMENT,64)
            samples.append(time.perf_counter()-tick)
        timings[name] = {"seconds_for_12_points":samples,
            "median_seconds":float(np.median(samples)),
            "stored_coefficients":core.stored_coefficients,
            "stored_bytes":core.stored_bytes}
    neg = longitudinal_current(compressed,1.5,1.,64)
    neg_reference = triangle_form_factor(1.5,1.,128)
    # Post-result scope check: retaining 1/s falloff does not impose its exact
    # coefficient. This does NOT retune the rank on the held-out observables.
    virtualities = np.linspace(A_MIN,A_MAX,121)
    tail_check = {"a_samples":121,"exact_s_times_h_limit":1.0,
        "dense_max_coefficient_error":float(max(abs(dense.bind(a).g(1.)-1) for a in virtualities)),
        "selected_rank_max_coefficient_error":float(max(abs(compressed.bind(a).g(1.)-1) for a in virtualities)),
        "exact_boundary_constraint_imposed":False}
    return {
        "purpose":"joint scalar state/operator compression; no learned dynamics or physical topology claim",
        "representation":"common tail-factored propagator in virtuality/radial Chebyshev coordinates",
        "domain":{"M_over_m_max":1.9,"Q2_over_m2_max":4.,"a_interval":[1-1.9**2/4,2.]},
        "targets":{"reference_absolute_F":1e-8,"compressed_absolute_current":1e-4,"state_L2":1e-3},
        "degree":DEGREE,"selected_rank":selected,"rank_selected_on_development_only":True,
        "holdout_is_kinematic_validation_not_new_dynamics":True,
        "exact_pair_form_factor_appended":False,"separate_curve_renormalization":False,
        "interpolation_refinement":interpolation,"development_rank_curve":curves,
        "holdout":holdout,"dense_holdout":dense_holdout,
        "quadrature_64_to_96":refinement,"basis_phase_checks":phase_checks,
        "ultraviolet_tail_check":tail_check,
        "omitted_pair_relative_error_at_M1p5_Q2_1":float(abs(neg.valence/neg_reference-1)),
        "timings":timings,
        "storage_reduction_vs_dense":dense.stored_bytes/compressed.stored_bytes,
        "runtime_seconds_excluding_imports":time.perf_counter()-start,
        "environment":{"python":platform.python_version(),"numpy":np.__version__},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    report = evaluate()
    encoded = json.dumps(report,indent=2,allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x",encoding="utf-8") as stream:
            stream.write(encoded)
        summary = {key:report[key] for key in ("selected_rank","storage_reduction_vs_dense",
            "quadrature_64_to_96","basis_phase_checks","ultraviolet_tail_check",
            "timings","runtime_seconds_excluding_imports")}
        summary["holdout_maxima"] = report["holdout"]["maxima"]
        summary["development_curve"] = [{"rank":r["rank"],**r["maxima"]}
                                        for r in report["development_rank_curve"]]
        print(json.dumps(summary,indent=2))
    else:
        print(encoded,end="")
    if not report["holdout"]["maxima"]["targets_met"]:
        raise SystemExit("Held-out representation target failed; do not retune on the holdouts.")
    if max(report["quadrature_64_to_96"].values()) > 1e-8:
        raise SystemExit("Quadrature refinement is unresolved; reassess the representation evidence.")


if __name__ == "__main__":
    main()
