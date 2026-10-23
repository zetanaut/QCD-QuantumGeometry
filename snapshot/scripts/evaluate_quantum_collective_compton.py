"""Replay two-photon matching/absorption controls; prints JSON only."""
import json
import numpy as np
from scipy.linalg import expm
from deuteron_wigner.quantum.quantum_collective_compton import (
    box_unit,compton_unit,compton_kinematics,scalar_photon_coefficient,
    scalar_photon_trace,photon_vacuum_kernel,scalar_exchange,
    scalar_exchange_matrix,dipole_estimate,absorption_dipole_sum,
    absorption_cross_section,dot,triangle_unit,ALPHA_EM,HBARC,
)
from deuteron_wigner.quantum.common_action_pilot import calibrated_action


def evaluate():
    mp=.13957039
    model=calibrated_action(mp,.0922,.515,.004,.6,1.4,.12,[3.,1.])
    state=model.solve(.004);v=state['vacuum'];e=state['pion_modes'][:,0]
    mass=state['quark_mass'];g2=float((e@model.yukawa)**2)
    scan=[]
    for energy in [.02,.01,.005,.0025,.00125]:
        estimate=dipole_estimate(mass,mp,energy,yukawa_squared=g2)
        scan.append(dict(photon_energy_GeV=energy,**estimate))
    # Analytic crossing variable: forward x=omega^2, backward x=omega*omega'.
    # Fit the last three energies to a quadratic in x, not in omega; this
    # is numerical low-energy extrapolation, not fitting physical data.
    energies=np.array([r['photon_energy_GeV'] for r in scan[-3:]])
    intercepts=[]
    for name,angle in [('alpha_plus_beta',0.),('alpha_minus_beta',np.pi)]:
        xp=energies**2/(1+energies/mp*(1-np.cos(angle)))
        values=np.array([r[name] for r in scan[-3:]])
        intercepts.append(float(np.polynomial.polynomial.polyfit(xp,values,2)[0]))
    asum,adiff=intercepts
    box_alpha=(asum+adiff)/2;box_beta=(asum-adiff)/2
    bmatrix=scalar_exchange_matrix(model,v,mass,0.);bscalar=float(e@bmatrix@e)
    scalar_alpha=2*ALPHA_EM*HBARC**3/mp*bscalar
    _,hs=model.matrices(v);poles,modes=np.linalg.eigh(hs)
    potential_vertex=2*model.quartic*v*e*e
    mode_pieces=scalar_photon_coefficient(mass,0.)*(model.yukawa@modes)*(potential_vertex@modes)/poles
    dispersion=[dict(radial_order=n,angular_order=2*n,alpha_plus_beta_fm3=absorption_dipole_sum(
        mass,mp,order=n,angular_order=2*n,yukawa_squared=g2)) for n in [48,96,192]]
    p1,p0,k,kp=compton_kinematics(.025,1.1,mp)
    e1=np.array([0.,1,0,0]);e2=np.array([0.,np.cos(1.1),0.,-np.sin(1.1)])
    original=compton_unit(mass,p1,p0,k,-kp,e1,e2)
    shifted=compton_unit(mass,p1,p0,k,-kp,e1+.7*k,e2-.9*kp)
    gen=np.zeros((4,4));gen[0,1:]=gen[1:,0]=[.3,-.25,.15];boost=expm(gen)
    moved=compton_unit(mass,boost@p1,boost@p0,boost@k,-boost@kp,boost@e1,boost@e2)
    refined=compton_unit(mass,p1,p0,k,-kp,e1,e2,order=18,triangle_order=36)
    t=dot(k-kp,k-kp);inv=dot(k,kp)*dot(e1,e2)-dot(k,e2)*dot(kp,e1)
    scalar_actual=scalar_photon_trace(mass,k,-kp,e1,e2)
    scalar_reduced=4*scalar_photon_coefficient(mass,t)*inv
    scalar_amplitude=-4*scalar_exchange(model,v,e,e,mass,t)*inv
    offp=np.array([.15,.01,.02,.03]);q1=np.array([.04,.06,.01,.02]);q2=np.array([.03,-.02,.04,.01]);off1=offp+q1+q2
    polarization=np.array([.1,.3,-.2,.4])
    boxward=box_unit(mass,off1,offp,q1,q2,q1,polarization)
    vertexward=dot(polarization,triangle_unit(mass,off1,offp+q1)-triangle_unit(mass,off1-q1,offp))
    q=np.array([.04,.1,.03,.07]);step=1e-6
    massderivative=(photon_vacuum_kernel(mass+step,mass,q,e1,e2)-photon_vacuum_kernel(mass-step,mass,q,e1,e2))/(2*step)
    mass_source=scalar_photon_trace(mass,q,-q,e1,e2)
    return dict(inputs=dict(quark_mass_GeV=mass,external_mass_GeV=mp,yukawa_squared=g2,colors=3,
            flavor_charges=[2/3,-1/3],reference='same two-multiplet local action, old tree kinematics/potential/couplings; not a solved quantum pion',
            loop_order='one free-quark loop plus tree potential-linked scalar exchange; no gluon/collective loops or resummation',
            subtraction='same zero-invariant wave subtraction; common prescribed external pole and order-by-order external normalization; scalar source holds photon subtraction scale fixed',
            scope='electric e^2 stripped from amplitudes, restored in polarizabilities and absorption; no finite local C/D fit'),
        low_energy_scan_fm3=scan,
        extrapolated_quark_box_Born_fm3=dict(alpha_plus_beta=asum,alpha_minus_beta=adiff,alpha=box_alpha,beta=box_beta,
            method='last three energies, quadratic in forward omega^2 or backward omega*omega_prime; not physical-data fitting'),
        scalar_partner=dict(coefficient_matrix_GeV_minus2=bmatrix.tolist(),matrix_rank=int(np.linalg.matrix_rank(bmatrix)),
            b_sigma_GeV_minus2=bscalar,scalar_mass_squared_GeV2=poles.tolist(),scalar_mode_b_GeV_minus2=mode_pieces.tolist(),
            alpha_fm3=scalar_alpha,beta_fm3=-scalar_alpha,alpha_plus_beta_fm3=0.,
            unit_sigma_photon_coefficient_GeV_minus1=scalar_photon_coefficient(mass,0.)),
        combined_reference_loop_fm3=dict(alpha=box_alpha+scalar_alpha,beta=box_beta-scalar_alpha,
            alpha_plus_beta=asum,alpha_minus_beta=adiff+2*scalar_alpha,
            NOT_a_physical_prediction=True,local_source_terms_not_included=True),
        independent_absorption=dict(threshold_photon_energy_GeV=(4*mass*mass-mp*mp)/(2*mp),
            convergence=dispersion,forward_loop_difference_fm3=abs(dispersion[-1]['alpha_plus_beta_fm3']-asum),
            cross_sections=[dict(s_GeV2=s,sigma_GeV_minus2=absorption_cross_section(mass,mp,s,yukawa_squared=g2)) for s in [.35,.5,1.,3.,10.]],
            scope='gamma plus reference pion to unconfined u/dbar; canonical final spinors and three tree source attachments. Not physical hadron photoabsorption or a QCD lower bound.'),
        amplitude_controls=dict(unit_original=original,unit_gauge_shifted=shifted,
            gauge_difference=abs(original['normalized_loop']-shifted['normalized_loop']),
            omitted_Born_gauge_difference=abs(original['box']-shifted['box']),
            Lorentz_difference=abs(original['normalized_loop']-moved['normalized_loop']),
            quadrature_difference=abs(original['normalized_loop']-refined['normalized_loop']),
            scalar_trace_difference_GeV=abs(scalar_actual-scalar_reduced),
            scalar_mass_source_difference_GeV=abs(massderivative-mass_source),
            offshell_box_Ward=boxward,offshell_triangle_difference=vertexward,
            offshell_Ward_difference=abs(boxward-vertexward),
            projected_box_Born=g2*original['normalized_loop'],scalar_amplitude=scalar_amplitude,
            combined_reference_loop=g2*original['normalized_loop']+scalar_amplitude),
        reassessment=dict(local_tree_C_D_cannot_independently_change_alpha_plus_beta=True,explicit_finite_mass_loop_already_supplies_nonzero_sum=True,
            local_C_D_changed=False,physical_polarizability_refit=False,
            prior_COMPASS_input_assumed_alpha_equals_minus_beta=True,
            next='qualify an independent higher-derivative two-source structure and parameterized joint radius/alpha/beta matching; audit conditional data interpretation before any fit. Do not erase the verified positive reference absorption sum or select a production quark-meson model.',
            unresolved=['full QCD/chiral/source finite matching','all gluon/collective loops','independent quantum LF minus/transverse Compton','quantum state/vacuum matching','generated proton/neutron and modular nuclear states']))


if __name__=='__main__': print(json.dumps(evaluate(),indent=2,allow_nan=False))
