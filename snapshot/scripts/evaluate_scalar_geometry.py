"""Bounded tangent and state/current information audit; no optimizer or fit."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from deuteron_wigner.quantum.scalar_joint_representation import (
    build_core,dy_current,longitudinal_current,state_samples,
)
from deuteron_wigner.quantum.scalar_constrained_representation import ConstrainedCore
from deuteron_wigner.quantum.scalar_representation_geometry import (
    TangentChart,tangent_chart,state_jacobian,current_gradients,coefficient_matrix,
    displaced_core,horizontal_tangent,weak_response_direction,
)


def currents(core,mass,q2,order=96):
    lp = longitudinal_current(core,mass,q2,order)
    return np.array([dy_current(core,mass,q2,order),lp.valence,lp.pair])


def spectral_summary(a):
    s = np.linalg.svd(a,compute_uv=False)
    return {"maximum":float(s[0]),"minimum":float(s[-1]),
            "ranks_by_relative_cutoff":{str(c):int(np.count_nonzero(s>c*s[0])) for c in (1e-6,1e-8,1e-10)},
            "singular_values":s.tolist()}


def direction_row(core,chart,v,mass,q2,order=128,current_order=96):
    b,j,a = state_jacobian(core,chart,mass,order)
    g = current_gradients(core,chart,mass,q2,current_order)
    response = g["real"] @ v.real + g["imaginary"] @ v.imag
    return {"mass":mass,"q2":q2,"state_order":order,"current_order":current_order,
            "raw_state_tangent":float(np.linalg.norm(j@v)),
            "horizontal_state_tangent":float(np.linalg.norm(a@v)),
            "norm_derivative":float(2*np.vdot(b,j@v).real),
            "current_derivative_real":response.real.tolist(),
            "current_derivative_imaginary":response.imag.tolist(),
            "joint_diagnostic_norm_unweighted":float(np.sqrt(np.linalg.norm(a@v)**2+np.linalg.norm(response)**2))}


def evaluate():
    tick = time.perf_counter()
    core = ConstrainedCore(build_core(56,5))
    full = tangent_chart(core.core)
    n = full.fixed_right_count
    left = TangentChart(full.left[:,:n],full.right[:,:n],n)
    spectra = []
    for order in (32,48,64,96,128):
        b,j,a = state_jacobian(core,full,1.5,order)
        spectra.append({"order":order,**spectral_summary(a)})
    # Selection here constructs a diagnostic witness, not a model/accuracy fit.
    b,j,a = state_jacobian(core,full,1.5,128)
    g = current_gradients(core,full,1.5,1.,96)
    rng = np.random.default_rng(20260909)
    v = rng.normal(size=full.dimension)+1j*rng.normal(size=full.dimension)
    v /= np.linalg.norm(v)
    derivative = g["real"]@v.real+g["imaginary"]@v.imag
    finite_differences = []
    for step in (1e-3,1e-4,1e-5):
        plus,minus = displaced_core(core,full,v,step),displaced_core(core,full,v,-step)
        bp,bm = state_samples(plus,1.5,128).ravel(),state_samples(minus,1.5,128).ravel()
        state_fd = (bp-bm)/(2*step)
        normalized_fd = (bp/np.linalg.norm(bp)-bm/np.linalg.norm(bm))/(2*step)
        psi = b/np.linalg.norm(b)
        horizontal_fd = normalized_fd-psi*np.vdot(psi,normalized_fd)
        current_fd = (currents(plus,1.5,1)-currents(minus,1.5,1))/(2*step)
        finite_differences.append({"step":step,"raw_state_error":float(np.linalg.norm(state_fd-j@v)),
            "horizontal_error":float(np.linalg.norm(horizontal_fd-a@v)),
            "current_derivative_error":float(np.max(abs(current_fd-derivative)))})
    # Invert only an explicitly identified small, well-resolved test chart.
    directions,_ = np.linalg.qr(rng.normal(size=(full.dimension,8)))
    tangent = a@directions
    metric = tangent.T@tangent
    change = np.diag(np.linspace(.5,2.,8))+.1*rng.normal(size=(8,8))
    covector = g["real"][2].real@directions
    velocity = np.linalg.solve(metric,covector)
    transformed_metric = (tangent@change).T@(tangent@change)
    transformed_velocity = np.linalg.solve(transformed_metric,change.T@covector)
    geometric_error = np.linalg.norm(tangent@velocity-tangent@change@transformed_velocity)
    geometric_scale = np.linalg.norm(tangent@velocity)
    coordinate_checks = {
        "restricted_directions":8,"restricted_metric_condition":float(np.linalg.cond(metric)),
        "transformed_metric_error":float(np.linalg.norm(transformed_metric-change.T@metric@change)),
        "natural_direction_relative_error":float(geometric_error/geometric_scale),
        "euclidean_direction_relative_change":float(np.linalg.norm(tangent@covector-tangent@change@change.T@covector)/np.linalg.norm(tangent@covector)),
        "full_metric_inverted":False,"optimizer_executed":False,
    }
    x = rng.normal(size=(5,5))+1j*rng.normal(size=(5,5))
    l,r = core.core.left,core.core.right
    factor_null = (l@x)@r+l@(-x@r)
    finite_change = np.diag(np.linspace(.4,2,5))+.05*rng.normal(size=(5,5))
    gauge = {"raw_complex_factor_coordinates":570,"complex_gauge_dimension":25,
        "coefficient_tangent_complex_dimension":full.dimension,
        "infinitesimal_coefficient_residual":float(np.linalg.norm(factor_null)),
        "finite_factor_gauge_error":float(np.linalg.norm((l@finite_change)@np.linalg.solve(finite_change,r)-l@r)),
        "external_global_phase_horizontal_error":float(np.linalg.norm(horizontal_tangent(b,1j*b))),
        "radial_horizontal_error":float(np.linalg.norm(horizontal_tangent(b,b))),
        "global_phase_not_a_free_constrained_core_parameter":True}
    norm_grad = 2*b@j
    nv = norm_grad/np.linalg.norm(norm_grad)
    norm_prime = norm_grad@nv
    f0 = currents(core,1.5,0)
    g0 = current_gradients(core,full,1.5,0,96)["real"]
    squared_norm = np.linalg.norm(b)**2
    quotient_prime = (g0[0]@nv)/squared_norm-f0[0]*norm_prime/squared_norm**2
    normalization = {"squared_norm":float(squared_norm),"raw_charge_norm_derivative_error":float(abs(g0[0]@nv-norm_prime)),
        "raw_charge_derivative":float((g0[0]@nv).real),"normalized_charge_derivative":float(abs(quotient_prime)),
        "omitting_norm_derivative_error":float(abs((g0[0]@nv)/squared_norm)),
        "reference_current_renormalized":False}
    # Compare the coarse and refined single-state witnesses without hiding aliasing.
    coarse = state_jacobian(core,left,1.5,48)[2]
    single = a[:,:n]
    pair = g["real"][2,:n].real
    coarse_v,_,_ = weak_response_direction(coarse,pair,1e-8)
    single_v,_,single_count = weak_response_direction(single,pair,1e-8)
    coarse_check = {"order48_norm":float(np.linalg.norm(coarse@coarse_v)),
                    "order128_norm":float(np.linalg.norm(single@coarse_v))}
    single_rows = [direction_row(core,left,single_v,m,q) for m,q in ((1.5,1.),(1.2,.81),(1.8,3.24))]
    ensemble = np.vstack([state_jacobian(core,left,m,128)[2]/np.sqrt(3) for m in (1.,1.5,1.9)])
    ensemble_choices = []
    for cutoff in (1e-6,1e-8,1e-10):
        ev,s,count = weak_response_direction(ensemble,pair,cutoff)
        ensemble_choices.append({"cutoff":cutoff,"weak_directions":count,
            "state_ensemble_norm":float(np.linalg.norm(ensemble@ev)),"pair_response":float(abs(pair@ev))})
    ev,_,count = weak_response_direction(ensemble,pair,1e-8)
    # Freeze this explicit witness before refinement/transfer/current probes.
    witness_rows = [direction_row(core,left,ev,m,q,160) for m,q in
                    ((1.,.25),(1.5,0.),(1.5,.25),(1.5,1.),(1.5,4.),(1.9,4.),(1.2,.81),(1.8,3.24))]
    refinement = [direction_row(core,left,ev,1.5,1.,o,c) for o,c in ((96,64),(128,96),(160,128))]
    finite_probes = []
    base_f = currents(core,1.5,1)
    base_b = state_samples(core,1.5,160)
    for step in (1e-3,.1,.25):
        perturbed = displaced_core(core,left,ev,step)
        delta_f = currents(perturbed,1.5,1)-base_f
        delta_b = state_samples(perturbed,1.5,160)-base_b
        singular = np.linalg.svd(coefficient_matrix(perturbed.core),compute_uv=False)
        finite_probes.append({"step":step,"raw_state_distance":float(np.linalg.norm(delta_b)),
            "current_change_real":delta_f.real.tolist(),"current_change_imaginary":delta_f.imag.tolist(),
            "sixth_singular_value":float(singular[5]),
            "max_boundary_error":float(max(abs(perturbed.bind(.2).g(0)-5),abs(perturbed.bind(.2).g(1)-1),abs(perturbed.bind(1).g(.4)-1)))})
    return {"purpose":"fixed scalar geometry and joint-response audit; no physical dynamics or optimizer",
        "preedit_checkpoint":"3fba8ce2a9eacfe50b9e75540b53cdadcb09fd1a",
        "degree":56,"residual_rank":5,"factor_gauge":gauge,"single_state_spectra":spectra,
        "complex_direction_finite_differences":finite_differences,"coordinate_covariance":coordinate_checks,
        "normalization":normalization,"coarse_sampling_control":coarse_check,
        "single_state_witness":{"weak_count":single_count,"transfer_checks":single_rows},
        "ensemble_witness":{"state_masses":[1.,1.5,1.9],"cutoff_comparison":ensemble_choices,
            "selected_diagnostic_cutoff":1e-8,"fixed_right_coordinates":ev.tolist(),
            "refinement":refinement,"transfer_and_current_checks":witness_rows,"finite_probes":finite_probes},
        "state_only_metric_qualifies_all_current_directions":False,
        "joint_diagnostic_is_physical_QGT":False,"scalar_stage_research_boundary":"close after this audit; carry response-domain lesson into spinful shared-dynamics pilot pre-evaluation",
        "runtime_seconds":time.perf_counter()-tick,"environment":{"python":platform.python_version(),"numpy":np.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    report = evaluate()
    if args.output:
        with args.output.open("x",encoding="utf-8") as stream:
            stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("factor_gauge","complex_direction_finite_differences",
        "coordinate_covariance","normalization","coarse_sampling_control","runtime_seconds")},indent=2))
    print(json.dumps({k:report["ensemble_witness"][k] for k in ("cutoff_comparison","refinement","finite_probes")},indent=2))
    if max(r["current_derivative_error"] for r in report["complex_direction_finite_differences"])>1e-7:
        raise SystemExit("Current derivative validation failed; do not advance.")


if __name__ == "__main__":
    main()
