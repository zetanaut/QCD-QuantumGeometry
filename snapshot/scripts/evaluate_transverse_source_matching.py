"""Bounded common-action radius / independent two-photon matching replay."""
import json
import numpy as np

from deuteron_wigner.quantum.common_action_pilot import calibrated_action
from deuteron_wigner.quantum.transverse_source_matching import (
    SourceCurvature, action_vertex, action_contact, transverse_vertex,
    match_projected, polarizability, radius_coefficient, dot,
)


def evaluate():
    mass = .13957039
    c, d = match_projected(.659, 2e-4, mass)
    cases = []
    for angle, ratio, coupling in [(.35, .8, .1), (.6, 1.4, .12), (1., .8, .2)]:
        # Charged-mass kinematic variant, not an update to old isospin results.
        # The Nf=2+1 isospin scalar shape is an approximate benchmark here.
        base = calibrated_action(mass, .0922, .515, .004, angle, ratio, coupling, [3., 1.])
        s = base.solve(.004); e = s['pion_modes'][:, 0]; h = s['pion_modes'][:, 1]
        ee, hh, eh = np.outer(e, e), np.outer(h, h), np.outer(e, h)+np.outer(h, e)
        simple = SourceCurvature(c*ee, d*ee)
        alternative = SourceCurvature(c*ee+.6*eh+1.2*hh, d*ee-.4*eh+.8*hh)
        currents = []; error = 0.
        for q2 in [.02, .05, .1]:
            q = np.sqrt(q2); energy = np.sqrt(mass**2+q2/4)
            p0, p1 = np.array([energy,-q/2,0,0]), np.array([energy,q/2,0,0])
            j = action_vertex(simple,p1,p0,e,e)
            error = max(error,float(np.max(abs(j-transverse_vertex(p1,p0,c)))))
            currents.append(float((1+j[0]/(2*energy)).real))
        w=.03;theta=.8;k=np.array([w,0,0,w]);wp=w/(1+w/mass*(1-np.cos(theta)))
        kp=wp*np.array([1,np.sin(theta),0,np.cos(theta)])
        p0=np.array([mass,0,0,0]);p1=p0+k-kp;eps=np.array([0,0,1,0])
        invariant=dot(k,kp)*dot(eps,eps)-dot(k,eps)*dot(kp,eps)
        amplitude=action_contact(simple,p1,p0,k,-kp,eps,eps,e,e)
        radius_only=SourceCurvature(simple.c,np.zeros((2,2)))
        restricted=action_contact(radius_only,p1,p0,k,-kp,eps,eps,e,e)
        # Off-shell heavy/light transition is a coefficient-sensitivity witness,
        # not a physical decay prediction for the unqualified heavy mode.
        tp0=np.array([.3,.1,.02,.04]);tp1=np.array([.5,-.2,.03,.06])
        transition_delta=action_vertex(alternative,tp1,tp0,h,e)-action_vertex(simple,tp1,tp0,h,e)
        cases.append(dict(angle=angle,quartic_ratio=ratio,coupling_GeV2=coupling,
            pion_mass_squared_GeV2=float(s['pion_mass_squared'][0]),
            scalar_radius_squared_fm2=base.scalar_radius_squared(s['vacuum'],e),
            projected_C_GeV_minus2=simple.projected(e)[0],projected_D_GeV_minus2=simple.projected(e)[1],
            Q2_GeV2=[.02,.05,.1],tree_truncated_vector_F=currents,
            action_vertex_vs_analytic_max_error_GeV=error,
            real_Compton_structure_contact_charge_stripped=float(amplitude.real),
            real_Compton_action_vs_reduced_max_error=float(abs(amplitude+4*(c+2*d)*invariant)),
            radius_only_Compton_structure_contact_charge_stripped=float(restricted.real),
            alternative_same_light_projection_max_error=float(np.max(abs(np.array(alternative.projected(e))-[c,d]))),
            heavy_light_offshell_current_change_norm_GeV=float(np.linalg.norm(transition_delta))))
    sensitivity=[]
    sigma_alpha=np.hypot(.6,.7)*1e-4
    for radius,alpha in [(.655,2e-4),(.663,2e-4),(.659,2e-4-sigma_alpha),(.659,2e-4+sigma_alpha)]:
        cc,dd=match_projected(radius,alpha,mass)
        sensitivity.append(dict(radius_fm=radius,alpha_fm3=alpha,C_GeV_minus2=cc,D_GeV_minus2=dd))
    weights=np.array([.8**2,2*.8*.6,.6**2])
    response=np.array([np.r_[weights,0*weights],np.r_[weights,2*weights]])
    return dict(
        scope='Weak-source local tree derivative expansion. Not QCD-matched coefficients, production sigma model, new quantum states, or a precision likelihood.',
        matching_inputs=dict(radius_fm=.659,radius_error_fm=.004,radius_source='https://pdglive.lbl.gov/DataBlock.action?node=S008CR',
            charged_pion_mass_GeV=mass,mass_source='https://pdglive.lbl.gov/DataBlock.action?node=S008M',
            alpha_fm3=2e-4,alpha_stat_fm3=.6e-4,alpha_sys_fm3=.7e-4,
            polarizability_source='https://arxiv.org/abs/1405.6377',
            polarizability_qualification='COMPASS assumes alpha=-beta; extracted summary includes radiative/chiral-loop/nuclear corrections. No raw events, joint likelihood or QCD coefficient matching imported.',
            scalar_shape_qualification='Prior Nf=2+1 isospin scalar slope benchmark retained approximately in a separately retuned charged-mass variant; original isospin results unchanged.'),
        radius_only=dict(C_GeV_minus2=c,D_GeV_minus2=0.,alpha_fm3=polarizability(c,0,mass),
            alpha_with_original_isospin_mass_fm3=polarizability(c,0,.1348),
            outcome='Radius calibrated; independent two-photon comparison fails for D=0 at this tree order. No significance or all-orders exclusion claimed.'),
        joint_matching=dict(C_GeV_minus2=c,D_GeV_minus2=d,alpha_fm3=polarizability(c,d,mass),
            status='Two independently fitted projected coefficients, NOT two predictions.',
            radius_was_withheld_in_preceding_experiment_now_calibration=True),
        matching_rank=int(np.linalg.matrix_rank(response)),fixed_basis_symmetric_coefficients=6,unidentified_directions=4,
        cases=cases,one_input_at_a_time_sensitivity=sensitivity,
        expansion_boundary='0<=Q2<=0.1 GeV2 diagnostic window; linear slope is not a full form factor, no controlled loop/higher-order error or timelike/UV completion.',
        decision='Retain the two shared curvature/source operator classes and linked derivatives. Reject one-coefficient radius-only inference and default closure of matrix null directions. Next qualify their full-minus-retained matching map into the quark-resolved state/current network; a pion calibration must not become an independently fitted nucleon head or a sigma-model repair agenda.')


if __name__ == '__main__':
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
