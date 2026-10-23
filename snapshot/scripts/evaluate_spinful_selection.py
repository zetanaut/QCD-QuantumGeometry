"""Run the bounded high-level spinful architecture-selection controls."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from deuteron_wigner.quantum.spinful_selection_controls import (
    ETA,GAMMA,IDENTITY,dot,slash,momenta,loop_vertex,bilinear,extract,
    parameter_form_factors,lightfront_form_factors,finite_cloud_probability,
    inverse_propagator,ward_vertex,
)


def preferred_frame_vertex(pout,pin,strength=.02):
    """Deliberate conserved but noncovariant additive negative control."""
    q = pout-pin
    qsq = dot(q,q)
    n = np.array([1.,0.,0.,0.])
    nt = n-q*dot(n,q)/qsq
    return strength*(-qsq)/(1-qsq)*nt[:,None,None]*IDENTITY


def template(masses,qvalues):
    # Shape q, form-factor channel, species; includes no Born 1.
    return np.array([[parameter_form_factors(m,q,1.)-[1.,0.] for m in masses]
                     for q in qvalues]).transpose(0,2,1)


def inference_experiment():
    masses = [.35,.8,1.6]
    truth = np.array([.006,.004,.003])
    development = [.1,.3,.8,1.6]
    withheld = [.18,.55,1.2,2.7]
    d,h = template(masses,development),template(masses,withheld)
    designs = {"F1_only":d[:,0,:],"joint_F1_F2":d.reshape(-1,3)}
    sigma = 2e-5
    rng = np.random.default_rng(20260910)
    # Common F1 noise between comparisons; joint case has additional F2 data.
    noise = rng.normal(size=(100,len(development),2))*sigma
    reports = {}
    truth_predictions = h@truth
    for name,a in designs.items():
        exact,_ = nnls(a,a@truth)
        singular = np.linalg.svd(a,compute_uv=False)
        parameters,predictions,residuals = [],[],[]
        for realization in noise:
            error = realization[:,0] if name=="F1_only" else realization.ravel()
            fit,residual = nnls(a,a@truth+error)
            parameters.append(fit)
            predictions.append(h@fit)
            residuals.append(residual)
        parameters,predictions = np.array(parameters),np.array(predictions)
        errors = predictions-truth_predictions
        reports[name] = {"singular_values":singular.tolist(),"condition":float(singular[0]/singular[-1]),
            "noiseless_parameter_error":float(np.linalg.norm(exact-truth)),
            "parameter_RMSE":np.sqrt(np.mean((parameters-truth)**2,axis=0)).tolist(),
            "withheld_F1_RMSE":float(np.sqrt(np.mean(errors[:,:,0]**2))),
            "withheld_F2_RMSE":float(np.sqrt(np.mean(errors[:,:,1]**2))),
            "withheld_F2_max_absolute_error":float(np.max(abs(errors[:,:,1]))),
            "noise_fit_residual_RMS":float(np.sqrt(np.mean(np.array(residuals)**2)))}
    return {"scalar_species_masses":masses,"true_g2_over_16pi2":truth.tolist(),
            "development_Q2":development,"reserved_Q2":withheld,"noise_sigma":sigma,
            "noise_realizations":100,"same_F1_noise_in_both_comparisons":True,
            "charge_only_response_rank":0,"closure_within_fixed_known_family_only":True,
            "physical_data_used":False,"results":reports}


def evaluate():
    start = time.perf_counter()
    kappa = .01
    rows,refinement = [],[]
    boosts = [(.23,-.12,.17),(-.15,.22,-.1),(.31,.11,.27)]
    for mu in (.35,.8,1.6):
        for q2 in (.05,.25,1.,4.):
            invariant = parameter_form_factors(mu,q2,kappa,96)
            lf = lightfront_form_factors(mu,q2,kappa,64)
            high = lightfront_form_factors(mu,q2,kappa,96)
            refinement.append(float(np.max(abs(lf-high))))
            for index,boost in enumerate(boosts):
                po,pi = momenta(q2,boost,angle=.4+.37*index)
                vertex = loop_vertex(po,pi,mu,kappa,48)
                current = bilinear(vertex,po,pi)
                estimates,conditions,residuals = {},{},{}
                for name,components in (("plus",(0,)),("minus",(1,)),("transverse",(2,3)),("all",(0,1,2,3))):
                    value,residual,condition = extract(current,po,pi,components)
                    estimates[name] = value.real.tolist()
                    conditions[name],residuals[name] = condition,residual
                refined = bilinear(loop_vertex(po,pi,mu,kappa,80),po,pi)
                ward = np.einsum("i,ijk->jk",(po-pi)*ETA,current)
                rows.append({"mu":mu,"Q2":q2,"boost":list(boost),"F_reference":invariant.tolist(),
                    "F_lightfront":lf.real.tolist(),"component_extractions":estimates,
                    "component_conditions":conditions,"component_fit_residuals":residuals,
                    "max_component_F_error":float(max(np.max(abs(np.array(v)-invariant)) for v in estimates.values())),
                    "LF_reference_error":float(np.max(abs(lf-invariant))),
                    "current_conservation_residual":float(np.linalg.norm(ward)),
                    "loop_48_to_80_current_change":float(np.linalg.norm(current-refined))})
    negative = []
    for boost in boosts:
        po,pi = momenta(1.,boost)
        clean = loop_vertex(po,pi,.5,kappa)
        bad = bilinear(clean+preferred_frame_vertex(po,pi),po,pi)
        plus,_,_ = extract(bad,po,pi,(0,))
        transverse,_,_ = extract(bad,po,pi,(2,3))
        _,residual,_ = extract(bad,po,pi)
        negative.append({"boost":list(boost),"conservation_residual":float(np.linalg.norm(np.einsum("i,ijk->jk",(po-pi)*ETA,bad))),
                         "plus_F":plus.real.tolist(),"transverse_F":transverse.real.tolist(),
                         "component_disagreement":float(np.max(abs(plus-transverse))),"two_structure_residual":residual})
    rng = np.random.default_rng(93)
    ward_rows = []
    for _ in range(16):
        po,pi = rng.normal(size=(2,4))*.7
        target = inverse_propagator(po)-inverse_propagator(pi)
        for transverse in (-.03,0.,.03):
            vertex = ward_vertex(po,pi,transverse=transverse)
            residual = np.einsum("i,ijk->jk",(po-pi)*ETA,vertex)-target
            bare = slash(po-pi)-target
            ward_rows.append({"transverse_coefficient":transverse,"WTI_residual":float(np.linalg.norm(residual)),
                              "bare_vertex_residual":float(np.linalg.norm(bare))})
    on_shell = []
    po,pi = momenta(1.,boosts[0])
    for transverse in (-.03,0.,.03):
        f,residual,_ = extract(bilinear(ward_vertex(po,pi,transverse=transverse),po,pi),po,pi)
        on_shell.append({"transverse_coefficient":transverse,"F":f.real.tolist(),"fit_residual":residual})
    orbital = [{"Q2":q,"full":lightfront_form_factors(.5,q,kappa).real.tolist(),
                "orbital_removed":lightfront_form_factors(.5,q,kappa,orbital=False).real.tolist()} for q in (0.,.25,1.,4.)]
    norm = [{"kT_cutoff":cut,"two_body_probability":finite_cloud_probability(.5,cut,kappa)} for cut in (2.,8.,32.,128.)]
    inference = inference_experiment()
    return {"purpose":"high-level comparison of learnable spinful construction layers; no selected nucleon model",
        "preedit_checkpoint":"6f3b5bc4108fbbec35654c965b4e56ad6cfc6502",
        "control_action":"charged mass-one Dirac fermion plus neutral massive scalar(s), local Yukawa g phi psibar psi, through g^2",
        "kappa_g2_over_16pi2":kappa,"current_charge_subtracted_before_integration":True,
        "same_internal_external_fermion_mass":True,"normalized_continuum_state_supplied":False,
        "covariant_loop_and_LF_overlap":rows,"LF_64_to_96_max_change":max(refinement),
        "conserved_preferred_frame_negative":negative,"off_shell_WTI":ward_rows,
        "same_propagator_transverse_freedom":on_shell,"orbital_interference_control":orbital,
        "unsubtracted_cloud_cutoff_dependence":norm,
        "equal_scalar_fermion_mass_F2_zero":{"exact":kappa,"parameter":float(parameter_form_factors(1.,0.,kappa)[1]),
                                          "direct_LF":float(lightfront_form_factors(1.,0.,kappa)[1].real)},
        "synthetic_shared_interaction_inference":inference,
        "full_LF_four_current_derived":False,"nonperturbative_binding_solved":False,
        "physical_topology_demonstrated":False,"quantum_advantage":False,
        "runtime_seconds":time.perf_counter()-start,"environment":{"python":platform.python_version(),"numpy":np.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    report = evaluate()
    rows = report["covariant_loop_and_LF_overlap"]
    maxima = {key:max(row[key] for row in rows) for key in ("max_component_F_error","LF_reference_error","current_conservation_residual","loop_48_to_80_current_change")}
    report["positive_control_maxima"] = maxima
    if args.output:
        with args.output.open("x",encoding="utf-8") as stream:
            stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"maxima":maxima,"LF_refinement":report["LF_64_to_96_max_change"],
        "negative":report["conserved_preferred_frame_negative"],"transverse_freedom":report["same_propagator_transverse_freedom"],
        "inference":report["synthetic_shared_interaction_inference"],"runtime_seconds":report["runtime_seconds"]},indent=2))
    if max(maxima.values())>1e-7 or report["LF_64_to_96_max_change"]>1e-7:
        raise SystemExit("Reference/refinement unresolved; reassess before a model recommendation.")


if __name__ == "__main__":
    main()
