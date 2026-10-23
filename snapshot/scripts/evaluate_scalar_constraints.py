"""Compare exact-boundary scalar representations with a fresh validation set.

Development-only selection: first passing rank for each mode, then minimum
stored coefficient count; ties use the maximum development current error.
Kinematic holdouts are opened only after this decision. No observable fit.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from scripts.evaluate_scalar_joint_representation import DEVELOPMENT, HOLDOUT as SEEN_REGRESSION, compare, _vectors
from deuteron_wigner.quantum.scalar_joint_representation import (
    A_MIN,A_MAX,ExactCore,build_core,dy_current,longitudinal_current,
)
from deuteron_wigner.quantum.scalar_constrained_representation import ConstrainedCore


DEGREE = 56
RANKS = (1,2,3,4,5,6,8,12)
NEW_VALIDATION = [(m,q) for m in (1.2,1.4,1.6,1.8) for q in (.16,.81,1.69,3.24)]


def boundaries(core):
    a_grid = np.linspace(A_MIN,A_MAX,121)
    t_grid = np.linspace(0,1,121)
    return {
        "UV_coefficient_error":float(max(abs(core.bind(a).g(1)-1) for a in a_grid)),
        "zero_momentum_error":float(max(abs(core.bind(a).g(0)-1/a) for a in a_grid)),
        "unit_virtuality_error":float(np.max(np.abs(core.bind(1).g(t_grid)-1))),
    }


def development_study():
    curves,choices = [],[]
    for mode in ("lift","projection"):
        first = None
        for rank in RANKS:
            core = ConstrainedCore(build_core(DEGREE,rank),mode)
            report = compare(core,DEVELOPMENT)
            constraint = boundaries(core)
            row = {"mode":mode,"rank":rank,"stored_coefficients":core.stored_coefficients,
                   "stored_bytes":core.stored_bytes,"boundary_errors":constraint,**report}
            curves.append(row)
            if first is None and report["maxima"]["targets_met"] and max(constraint.values()) < 1e-12:
                first = {"mode":mode,"rank":rank,"stored_coefficients":core.stored_coefficients,
                         "worst_development_current_error":max(report["maxima"][key] for key in
                             ("DY_error","longitudinal_error","valence_error","pair_error","frame_difference"))}
        if first is not None:
            choices.append(first)
    if not choices:
        raise ArithmeticError("No passing construction; reassess without opening validation.")
    selected = min(choices,key=lambda r:(r["stored_coefficients"],r["worst_development_current_error"]))
    return {"curves":curves,"first_passing_choices":choices,"selected":selected}


def evaluate():
    start = time.perf_counter()
    study = development_study()
    selected = study["selected"]
    # No NEW_VALIDATION evaluation occurs before the frozen selection above.
    core = ConstrainedCore(build_core(DEGREE,selected["rank"]),selected["mode"])
    validation = compare(core,NEW_VALIDATION)
    regression = compare(core,SEEN_REGRESSION)
    grid = DEVELOPMENT + SEEN_REGRESSION + NEW_VALIDATION
    dense = ConstrainedCore(build_core(DEGREE),selected["mode"])
    dense_reference = compare(dense,DEVELOPMENT+NEW_VALIDATION)
    interpolation = []
    for degree in (24,40,56):
        c = ConstrainedCore(build_core(degree),selected["mode"])
        interpolation.append({"degree":degree,"maxima":compare(c,DEVELOPMENT)["maxima"]})
    refinement = {"selected_64_to_96":float(np.max(np.abs(_vectors(core,grid,64)-_vectors(core,grid,96))))}
    raw = build_core(DEGREE,5)
    base_comparison = compare(raw,NEW_VALIDATION)
    phase = lambda a,s: .8*a+.6*s/(1+s)
    phase_base = _vectors(core,[(1.5,1.)],64)
    phase_results = {}
    for transport in (True,False):
        lp = longitudinal_current(core,1.5,1.,phase=phase,transport_operator=transport)
        v = np.array([dy_current(core,1.5,1.,phase=phase,transport_operator=transport),lp.valence,lp.pair])
        phase_results["matched" if transport else "unmatched"] = float(np.max(np.abs(v-phase_base)))
    timings = {}
    for name,c in (("exact",ExactCore()),("unconstrained_rank5",raw),("selected",core),("constrained_dense",dense)):
        values = []
        for _ in range(3):
            tick = time.perf_counter()
            _vectors(c,DEVELOPMENT,64)
            values.append(time.perf_counter()-tick)
        timings[name] = {"seconds_for_12_points":values,"median_seconds":float(np.median(values)),
                         "stored_coefficients":c.stored_coefficients,"stored_bytes":c.stored_bytes}
    w_max = (1-A_MIN)**2/(4*A_MIN)
    return {
        "purpose":"constraint-preserving approximation of the fixed scalar core; not new conserved dynamics",
        "preedit_checkpoint":"952d6cb6","degree":DEGREE,"development":study,
        "new_validation_grid":NEW_VALIDATION,"selection_precedes_validation":True,
        "new_validation":validation,"seen_regression":regression,
        "unconstrained_rank5_new_grid":base_comparison,"dense_reference":dense_reference,
        "interpolation":interpolation,"quadrature_refinement":refinement,
        "selected_boundary_errors":boundaries(core),"old_rank5_boundary_errors":boundaries(raw),
        "absolute_residual_error_bounds":{"lift_multiplier_max":w_max,"projection_sup_norm_bound":4.0,
                                          "exact_lift_cancellation_ratio_max":1+2*w_max},
        "phase_transport_errors":phase_results,"timings":timings,
        "one_continuous_normalization":True,"exact_pair_form_factor_appended":False,
        "all_physical_constraints_proved":False,"quantum_advantage":False,
        "runtime_seconds_excluding_imports":time.perf_counter()-start,
        "environment":{"python":platform.python_version(),"numpy":np.__version__},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-only",action="store_true")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    report = development_study() if args.development_only else evaluate()
    encoded = json.dumps(report,indent=2,allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x",encoding="utf-8") as stream:
            stream.write(encoded)
    if args.development_only:
        print(json.dumps({"selected":report["selected"],"first_passing_choices":report["first_passing_choices"],
                         "curve":[{"mode":r["mode"],"rank":r["rank"],**r["maxima"],**r["boundary_errors"]}
                                  for r in report["curves"]]},indent=2))
    else:
        print(json.dumps({"selected":report["development"]["selected"],
            "validation_maxima":report["new_validation"]["maxima"],
            "boundary_errors":report["selected_boundary_errors"],
            "refinement":report["quadrature_refinement"],"timings":report["timings"],
            "runtime_seconds":report["runtime_seconds_excluding_imports"]},indent=2))
        if not report["new_validation"]["maxima"]["targets_met"]:
            raise SystemExit("New validation failed; do not retune on it.")
        if report["quadrature_refinement"]["selected_64_to_96"] >= 1e-8:
            raise SystemExit("Numerical refinement unresolved; reassess before advancing.")


if __name__ == "__main__":
    main()
