"""Replay resolved qq one-photon matching controls; prints JSON, writes nothing."""
import json
import numpy as np
from scipy.linalg import expm

from deuteron_wigner.quantum.nonzero_momentum_current import (
    benchmark_reference, benchmark_momenta, radiative_current,
    canonical_radiative_current, contract, CollectiveChannel,
    QuarkCollectiveReference, lf_coordinates, dot, external_states, GAMMA,
)


def norm(a):
    return float(np.linalg.norm(a))


def evaluate():
    model=benchmark_reference(); rows=[]
    for variation in [0.,.5,2.,-1.]:
        out,inc=benchmark_momenta(model.mass,variation)
        soft=radiative_current(model,out,inc)
        slf=radiative_current(model,out,inc,propagation='light_front')
        sp=radiative_current(model,out,inc,propagation='propagating')
        hard=canonical_radiative_current(model.mass,out,inc,1.2)
        hlf=canonical_radiative_current(model.mass,out,inc,1.2,
                quark_propagation='light_front',gluon_representation='light_front')
        hpg=canonical_radiative_current(model.mass,out,inc,1.2,gluon_representation='propagating')
        hpq=canonical_radiative_current(model.mass,out,inc,1.2,quark_propagation='propagating')
        full=soft['total']+hard['total'];lf=slf['total']+hlf['total'];q=soft['photon']
        rows.append(dict(variation=variation,outgoing_GeV=out.tolist(),incoming_GeV=inc.tolist(),
            Q2_exchanges_GeV2=[float(-dot(p,p)) for p in soft['exchange']],photon_q2_GeV2=float(dot(q,q)),
            soft_current_norm_GeV_inverse=norm(soft['total']),canonical_current_norm_GeV_inverse=norm(hard['total']),
            total_current_norm_GeV_inverse=norm(full),LF_component_relative=norm(full-lf)/norm(full),
            Ward_norm=norm(contract(full,q)),Ward_relative=norm(contract(full,q))/(norm(q)*norm(full)),
            omit_collective_current_Ward_norm=norm(contract(soft['endpoint'],q)),
            omit_quark_instantaneous_soft_relative=norm(sp['total']-soft['total'])/norm(soft['total']),
            omit_quark_instantaneous_canonical_relative=norm(hpq['total']-hard['total'])/norm(hard['total']),
            omit_gluon_instantaneous_canonical_relative=norm(hpg['total']-hard['total'])/norm(hard['total']),
            canonical_radiating_gluon_divergence=hard['max_radiating_gluon_divergence']))
    out,inc=benchmark_momenta(model.mass);q=np.sum(out-inc,axis=0)
    soft=radiative_current(model,out,inc);retained=radiative_current(model.retain_lightest(),out,inc)
    zero_c=CollectiveChannel(model.pion.hessian,model.pion.couplings,np.zeros_like(model.pion.c))
    minimal=radiative_current(QuarkCollectiveReference(model.scalar,zero_c,model.mass),out,inc)
    r,s=out-inc;evals,e=np.linalg.eigh(model.pion.hessian)
    heavy=CollectiveChannel(evals[1:,None],e[:,1:].T@model.pion.couplings,e[:,1:].T@model.pion.c@e[:,1:])
    cross=model.pion.current(r,-s)-model.pion.retain_lightest().current(r,-s)-heavy.current(r,-s)
    matches=dict(retained_scalar_mass_GeV=float(np.sqrt(np.linalg.eigvalsh(model.scalar.hessian)[0])),
        retained_pion_mass_GeV=float(np.sqrt(evals[0])),
        residual_current_norm_GeV_inverse=norm(soft['total']-retained['total']),
        residual_Ward_norm=norm(contract(soft['total']-retained['total'],q)),
        lost_heavy_light_source_coefficient_norm_GeV_minus3=norm(cross),
        zero_C_change_norm_GeV_inverse=norm(soft['total']-minimal['total']),
        zero_C_change_Ward_norm=norm(contract(soft['total']-minimal['total'],q)))
    # Covariant component transformation with actual momentum-dependent LF spins.
    base=soft['total']+canonical_radiative_current(model.mass,out,inc,1.2)['total']; boosts=[]
    for direction in [[.4,0,0],[0,0,.7],[-.3,.2,.15]]:
        generator=np.zeros((4,4));generator[0,1:]=generator[1:,0]=direction
        lorentz=expm(generator)
        spin=np.kron(expm(sum(direction[i]*GAMMA[0]@GAMMA[i+1]/2 for i in range(3))),np.eye(6))
        def w(p):
            return external_states(lorentz@p,model.mass).conj().T@np.kron(GAMMA[0],np.eye(6))@spin@external_states(p,model.mass)/(2*model.mass)
        left,right=np.kron(w(out[0]),w(out[1])),np.kron(w(inc[0]),w(inc[1]))
        expected=np.array([left@j@right.conj().T for j in np.einsum('mn,nij->mij',lorentz,base)])
        bo,bi=out@lorentz.T,inc@lorentz.T
        actual=radiative_current(model,bo,bi,propagation='light_front')['total']
        actual+=canonical_radiative_current(model.mass,bo,bi,1.2,
            quark_propagation='light_front',gluon_representation='light_front')['total']
        boosts.append(dict(rapidity_vector=direction,full_component_relative=norm(actual-expected)/norm(expected)))
    # Positive external energies but negative internal plus: explicit anti-quark term.
    spatial=np.array([[.9,-.2,-.8],[-.05,.09,-.2],[.04,.02,.1],[.2,-.1,1.2]])
    momenta=np.c_[np.sqrt(model.mass**2+np.sum(spatial**2,axis=1)),spatial]
    po,pi=momenta[:2],momenta[2:];pq=np.sum(po-pi,axis=0)
    pa=radiative_current(model,po,pi);pb=radiative_current(model,po,pi,propagation='light_front')
    pair=dict(outgoing_GeV=po.tolist(),incoming_GeV=pi.tolist(),
        minimum_internal_plus_GeV=min(float(lf_coordinates(p)[0]) for p in [*(pi+pq),*(po-pq)]),
        LF_component_relative=norm(pa['total']-pb['total'])/norm(pa['total']),Ward_norm=norm(contract(pb['total'],pq)))
    return dict(inputs=dict(reference='explicit nonzero-momentum TREE action, not quantum-cell continuation',
                    order='e*g_i*g_j and e*g_s^2; electric e and overall i stripped',
                    quark_mass_GeV=model.mass,gs_squared=1.2,
                    scalar_H_GeV2=model.scalar.hessian.tolist(),pion_H_GeV2=model.pion.hessian.tolist(),
                    yukawa=model.pion.couplings.tolist(),C_GeV_minus2=model.pion.c.tolist(),
                    retained_content='both full multiplets; comparison retains lowest scalar and pion eigenmodes',
                    excluded_content='no loop modes integrated; no physical UV split or QCD coefficients matched',
                    source_subtraction='difference of full and explicitly retained induced amplitudes and currents',
                    external_order='spin(-,+) x flavor(u,d) x color(0,1,2), 12 per quark',
                    domain='listed nonzero spacelike exchanges, off-pole quark lines and k+ != 0; not a fitted EFT window'),
                cases=rows,source_matching=matches,boosts=boosts,signed_pair_route=pair,
                maxima=dict(LF_component_relative=max(r['LF_component_relative'] for r in rows),
                    Ward_relative=max(r['Ward_relative'] for r in rows),
                    boost_component_relative=max(r['full_component_relative'] for r in boosts)),
                claims=dict(tree_amplitude_matching=True,independent_LF_quark_and_gluon=True,
                    physical_nucleon=False,quantum_rematched_coefficients=False,
                    quantum_spatial_field_solve=False,full_QCD_matching=False,old_covariance_repaired=False,
                    regulator_or_BLFQ_H2_selected=False))


if __name__ == '__main__':
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
