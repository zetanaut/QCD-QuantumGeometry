"""Print candidate-analyticity evidence; no fit, production changes or file writes."""
import json
import numpy as np
from deuteron_wigner.quantum.nonlocal_binding_analyticity import (
    CRITICAL_RATIO, GaussianMass, contour_count, lf_momentum, lf_pole, lf_residue,
)


def pair(value):
    value = complex(value)
    return [value.real, value.imag]


def evaluate():
    scale = .8; cases = []
    for ratio in (.3,.45,.6):
        model = GaussianMass(scale*ratio, scale)
        poles = [model.chiral_pole(branch) for branch in (0,-1)]
        bounds = tuple(scale**2*x for x in (-.2,1.8,-1.1,1.1))
        wbounds = tuple(scale*x for x in (.1,1.1,.02,.75))
        cases.append(dict(M0_over_Lambda=ratio, invariant_poles_GeV2=[pair(s) for s in poles],
            scalar_invariant_residues=[pair(1/model.derivative(s)) for s in poles],
            positive_real_part_frequency_poles_at_rest_GeV=[pair(np.sqrt(s)) for s in poles],
            lf_poles_at_plus_08_transverse_02_minus01_GeV=[pair(lf_pole(s,.8,(.2,-.1))) for s in poles],
            invariant_rectangle_GeV2=list(bounds), invariant_count=[pair(contour_count(model.denominator,model.derivative,bounds,n)) for n in (64,96)],
            upper_frequency_rectangle_GeV=list(wbounds), upper_frequency_count=[pair(contour_count(lambda w:model.denominator(w*w),lambda w:2*w*model.derivative(w*w),wbounds,n)) for n in (96,144)],
            max_relative_equation_error_ten_branches=max(float(abs(model.denominator(model.chiral_pole(k)))/(abs(model.chiral_pole(k))+model.m0**2)) for k in range(-5,5))))
    model = GaussianMass(.36,.8,.004)
    s = model.numerical_pole(.8**2*(.4+.3j)); pp=.8;pt=(.2,-.1);pm=lf_pole(s,pp,pt)
    singular = np.linalg.svd(model.inverse_dirac(lf_momentum(pp,pm,pt)),compute_uv=False)
    residue = lf_residue(model,s,pp,pt);errors=[]
    for step in (1e-5,1e-6):
        delta=step*(1+.4j)
        errors.append(float(np.linalg.norm(delta*np.linalg.inv(model.inverse_dirac(lf_momentum(pp,pm+delta,pt)))-residue)/np.linalg.norm(residue)))
    return dict(date='2026-09-11',preedit_commit='1cb997e21a80a8dd858e6cfc850af8ec19b6789a',
        scope='Conditional Gaussian-smeared chiral mass analytic structure; not a gap/nucleon solve or a selected confinement model',
        Lambda_GeV=scale,critical_chiral_M0_over_Lambda=CRITICAL_RATIO,
        complex_numbers_encoded_as_real_imaginary_pairs=True,cases=cases,
        finite_current_mass_check=dict(current_mass_GeV=.004,M0_GeV=.36,invariant_pole_GeV2=pair(s),
            lf_pole_GeV=pair(pm),Dirac_singular_values_GeV=singular.tolist(),
            lf_residue_norm=float(np.linalg.norm(residue)),relative_residue_limit_errors=errors),
        naive_global_ansatz_is_direct_self_adjoint_retained_mode_kinetic_input=False,
        complex_colored_poles_prove_loss_of_locality=False,
        positive_colored_spectral_density_required=False,physical_sector_projection_or_continuation_supplied=False,
        physical_binding_or_chiral_gap_solution=False,canonical_kernels_changed=False,
        production_model_selected=False,public_push=False)


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2))
