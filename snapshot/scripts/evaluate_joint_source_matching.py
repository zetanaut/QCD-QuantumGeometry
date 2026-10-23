"""Replay shared source structure and conditional inference, not a data fit."""
import json
from pathlib import Path
import numpy as np
from deuteron_wigner.quantum.joint_source_matching import (
    JointSourceAction,photon_contact,action_contact,stress_contact,
    projected_observables,match_projected_observables,match_conditional_observables,
    compass_dipole_response,conditional_projection,HBARC,ALPHA_EM,
)


def evaluate():
    mp=.13957039;mode=np.array([.8,.6])
    source=JointSourceAction(np.array([[.7,.1],[.1,.4]]),np.array([[-.2,.07],[.07,.3]]),np.array([[.5,.15],[.15,.8]]))
    zero=np.zeros((2,2));stress=JointSourceAction(zero,zero,source.e)
    p0=np.array([.2,.02,.04,.03]);q1=np.array([.04,.02,.05,-.01]);q2=np.array([.03,.06,-.02,.01])
    q3=np.array([.02,-.01,.04,.03]);q4=np.array([.02,.04,-.03,.01])
    e1=np.array([.1,.3,-.2,.4]);e2=np.array([.15,-.1,.4,.2]);e3=np.array([.2,.1,.05,-.2])
    p1=p0+q1+q2;projected=source.projected(mode)
    actual=action_contact(stress,p1,p0,q1,q2,e1,e2,mode,mode)
    reduced=stress_contact(p1,p0,q1,q2,e1,e2,projected[2])
    p1+=q3
    three=photon_contact(stress,p1,p0,[q1,q2,q3],[e1,e2,q3],mode,mode)
    ward3=action_contact(stress,p1,p0+q3,q1,q2,e1,e2,mode,mode)-action_contact(stress,p1-q3,p0,q1,q2,e1,e2,mode,mode)
    p1+=q4
    four=photon_contact(stress,p1,p0,[q1,q2,q3,q4],[e1,e2,e3,q4],mode,mode)
    ward4=photon_contact(stress,p1,p0+q4,[q1,q2,q3],[e1,e2,e3],mode,mode)-photon_contact(stress,p1-q4,p0,[q1,q2,q3],[e1,e2,e3],mode,mode)
    root=Path(__file__).resolve().parents[1]
    current=json.loads((root/'docs/next_level/quantum_collective_current_results.json').read_text())
    compton=json.loads((root/'docs/next_level/quantum_collective_compton_results.json').read_text())
    # Explicit replay provenance; not recomputation or a change of old physics.
    radius=next(v['loop_radius_squared_fm2'] for v in current.values() if isinstance(v,dict) and 'loop_radius_squared_fm2' in v)
    dipoles=compton['combined_reference_loop_fm3']
    loop=np.array([radius,dipoles['alpha'],dipoles['beta']])
    params=np.array([.4,-.12,.8]);target=projected_observables(*params,mp,loop=loop)
    inferred=match_projected_observables(target,mp,loop=loop)
    wrong=projected_observables(*match_projected_observables(target,mp),mp,loop=loop)
    cosines=np.array([-.9,-.7,-.5,-.3,-.1])
    response=np.array([compass_dipole_response(mp,4*mp*mp,z) for z in cosines])
    response/=max(abs(response[:,0]))
    examples=[]
    for name,nuisance in [('fixed synthetic normalization',None),('free synthetic constant offset',np.ones((5,1)))]:
        result=conditional_projection(response,np.eye(5),nuisance=nuisance)
        result['information_matrix']=result['information_matrix'].tolist()
        examples.append(dict(name=name,**result))
    family=[]
    for kappa in [-.5,0.,.8]:
        for e in [-1.2,0.,2.4]:
            coefficients=match_conditional_observables(.45,.00023,mp,sum_weight=kappa,e=e,loop=loop)
            values=projected_observables(*coefficients,mp,loop=loop)
            family.append(dict(kappa=kappa,renormalized_projected_C_D_E=coefficients.tolist(),
                radius_squared_fm2=float(values[0]),alpha_fm3=float(values[1]),beta_fm3=float(values[2]),
                alpha_plus_beta_fm3=float(values[1]+values[2]),
                conditional_alpha_fm3=float(.5*(values[1]-values[2])+kappa*(values[1]+values[2]))))
    weights=np.array([mode[0]**2,2*mode[0]*mode[1],mode[1]**2]);k=2*ALPHA_EM*HBARC**3/mp
    design=np.array([np.r_[6*HBARC**2*weights,0*weights,0*weights],np.r_[k*weights,2*k*weights,k*mp*mp*weights],np.r_[-k*weights,-2*k*weights,0*weights]])
    return dict(scope='Unfitted local source-action and synthetic matching controls; no physical pion, nucleon or detector likelihood.',
        operator='-E_ij (D_mu phi_i)^T F^{mu rho} F_rho^nu D_nu phi_j, E GeV^-4; full O4 sources; arbitrary node count',
        fixture=dict(external_mass_GeV=mp,mode=mode.tolist(),C=source.c.tolist(),D=source.d.tolist(),E=source.e.tolist(),projected_C_D_E=list(projected)),
        actual_action_derivatives=dict(two_photon=actual.real,independent_two_photon=reduced,
            two_photon_difference=abs(actual-reduced),three_photon_Ward=three.real,three_lower_difference=ward3.real,
            three_photon_Ward_error=abs(three-ward3),four_photon_Ward=four.real,four_lower_difference=ward4.real,
            four_photon_Ward_error=abs(four-ward4)),
        source_only_dipoles_fm3=projected_observables(0,0,projected[2],mp)[1:].tolist(),
        known_reference_loop=dict(radius_squared_fm2=radius,alpha_fm3=loop[1],beta_fm3=loop[2],
            provenance=['quantum_collective_current_results.json','quantum_collective_compton_results.json'],
            physical_prediction=False,already_normalized_reference_observables_not_bare_coefficients=True),
        synthetic_three_condition_matching=dict(parameters=params.tolist(),target=target.tolist(),inferred=inferred.tolist(),
            inverse_error=float(np.max(abs(params-inferred))),double_count_error_equals_loop=float(np.max(abs(wrong-target-loop)))),
        matrix_information=dict(independent_symmetric_entries=9,three_projected_condition_rank=int(np.linalg.matrix_rank(design)),
            unresolved_matrix_directions=6,conditional_plus_radius_projected_rank=2,unresolved_projected_directions=1),
        synthetic_response_examples=examples,synthetic_conditional_families=family,
        source_audit=dict(paper='https://arxiv.org/abs/1405.6377',official_public_sources='https://wwwcompass.cern.ch/compass/publications/papers/',
            fit='conditional alpha=-beta, normalization floated; corrected pointlike denominator includes radiative, chiral-loop and form-factor terms',
            actual_folded_two_template_response_or_likelihood_recovered=False,
            archive_scope='bounded paper/public source inspection, not a proof no further data exist',
            synthetic_responses_are_not_COMPASS_templates=True),
        reassessment=dict(physical_coefficient_refit=False,production_model_selected=False,source_free_dynamics_changed=False,
            known_loop_erased=False,complete_higher_derivative_basis=False,public_push=False,
            next='pre-evaluate source-preserving transport into the shared finite-momentum quark/quantum network; qualify accessible and hidden mode/source directions and coupled derivatives. Keep conditional physical matching parameterized; no reference-model completion agenda.'))


if __name__=='__main__': print(json.dumps(evaluate(),indent=2,allow_nan=False))
