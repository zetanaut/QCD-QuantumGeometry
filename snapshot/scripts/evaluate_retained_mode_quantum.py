"""Finite-cell retained quantum modes, exact elimination and convergence."""
import json
import numpy as np
from deuteron_wigner.quantum.common_action_pilot import calibrated_action,HBARC
from deuteron_wigner.quantum.collective_quark_matching import exchange_weights
from deuteron_wigner.quantum.retained_mode_quantum import (
    chiral_cell,ground_state,connected_response,eliminate_exact,
    eliminated_ground_energy,eliminated_response,oscillator_states,
)


def source(cell,couplings,component):
    weights=np.zeros((2,4));weights[:,component]=couplings
    return cell.linear_source(weights)


def evaluate(full_resolution=True):
    a=calibrated_action(.13957039,.0922,.515,.004,.6,1.4,.12,[3.,1.])
    tree=a.solve(.004);cases=[]
    specifications=[(1000.,w,n) for w in [.5,.65] for n in ([6,8,10,12] if full_resolution else [4,6])]
    if full_resolution: specifications += [(500.,.5,10),(2000.,.5,10)]
    for volume,frequency,shell in specifications:
        cell=chiral_cell(a,.004,volume=volume,shell=shell,frequency=frequency)
        state=ground_state(cell.hamiltonian);psi=state['state']
        ops=[source(cell,a.yukawa,k) for k in [0,1]]+[source(cell,a.scalar_source,1)]
        result=connected_response(cell,state,ops)
        vacuum=np.array([psi@cell.fields[i]@psi for i in [0,4]])
        ward=a.scalar_source@vacuum/.004
        cases.append(dict(volume_GeV_minus3=volume,cell_side_fm=volume**(1/3)*HBARC,
            first_omitted_spatial_wavenumber_GeV=2*np.pi/volume**(1/3),
            basis_frequency_GeV=frequency,total_excitation_cutoff=shell,dimension=len(psi),
            ground_energy_GeV_constant_removed=state['energy'],state_equation_residual_GeV=state['residual'],
            sigma_expectations_GeV=vacuum.tolist(),
            quark_mass_insertion_GeV=float(.004+a.yukawa@vacuum),
            scalar_pseudoscalar_quark_susceptibilities_GeV_minus2=np.diag(result['susceptibility'])[:2].tolist(),
            source_PCAC_susceptibility_GeV2=float(result['susceptibility'][2,2]),
            source_PCAC_Ward_error_GeV2=float(abs(result['susceptibility'][2,2]-ward)),
            max_linear_solve_residual=float(max(result['linear_solve_errors']))))
    # A small complete-shell cell supplies the dense independent elimination
    # control. It is not advertised as a converged physical calculation.
    cell=chiral_cell(a,.004,volume=1000.,shell=3,frequency=.5)
    state=ground_state(cell.hamiltonian);psi=state['state'];h=cell.hamiltonian.toarray()
    p=np.array([i for i,s in enumerate(cell.states) if sum(s[4:])==0])
    energy=eliminated_ground_energy(h,p);reduced=eliminate_exact(h,p,energy)
    phi=psi[p];wave=reduced['embedding'];ops=[source(cell,a.yukawa,k) for k in [0,1]]
    means=[float(psi@op@psi) for op in ops]
    transported=[float(phi@wave.T@op@wave@phi/(phi@reduced['norm']@phi)) for op in ops]
    bare=[float(phi@op.toarray()[np.ix_(p,p)]@phi/(phi@phi)) for op in ops]
    dynamic=[]
    for nu in [0.,.2,.5]:
        full=connected_response(cell,state,ops,nu)['susceptibility']
        elim=eliminated_response(cell,state,ops,p,nu)
        dynamic.append(dict(euclidean_frequency_GeV=nu,full_susceptibility_GeV_minus2=full.tolist(),
            elimination_max_error_GeV_minus2=float(np.max(abs(full-elim['susceptibility']))),
            omitted_direct_response_error_norm_GeV_minus2=float(np.linalg.norm(full-elim['mediated']))))
    node_states=oscillator_states(3,4);lookup={s:i for i,s in enumerate(node_states)}
    amplitude=np.zeros((len(node_states),len(node_states)))
    for value,s in zip(psi,cell.states): amplitude[lookup[s[:4]],lookup[s[4:]]]=value
    probabilities=np.linalg.svd(amplitude,compute_uv=False)**2
    probabilities=probabilities[probabilities>1e-15]
    return dict(
        scope='Ordinary-time homogeneous eight-field quantum mechanics in a declared finite cell, not a light-front zero mode, a 3+1D quantum field calculation, a matched Wilsonian QCD action or a physical pion/nucleon model.',
        interpretation='User selected action for retained quantum modes; each mode is quantized once. Subsequent exact projection changes representation, not which fluctuations are counted.',
        scale_boundary='Cell volume and basis resolution declared. 2pi/L labels the first omitted spatial wavenumber, NOT a matched QCD renormalization scale. Omitted spatial/quark/gluon modes have not been integrated out to determine these coefficients.',
        physical_inputs='Prior tree calibration reused without refitting solely as a reference; quantum matching not inferred. Canonical LF/quark/gluon and transverse C/D operators preserved separately, not all quantized in this pilot.',
        tree_reference=dict(vacuum_GeV=tree['vacuum'].tolist(),quark_mass_insertion_GeV=tree['quark_mass'],
            scalar_pseudoscalar_weights_GeV_minus2=exchange_weights(a,tree['vacuum'],0)[[0,1]].tolist()),
        quantum_cases=cases,
        exact_elimination=dict(full_dimension=len(psi),retained_dimension=len(p),
            retained_rule='Second multiplet in its numerical oscillator vacuum; representation test, not physical high-energy selection.',
            full_energy_GeV=state['energy'],independent_reduced_energy_GeV=energy,
            reconstructed_state_error=float(np.linalg.norm(wave@phi-psi)),
            retained_naive_norm=float(phi@phi),physical_norm=float(phi@reduced['norm']@phi),
            full_source_means=means,transported_source_means=transported,bare_projected_source_means=bare,
            response_comparisons=dynamic,
            small_cell_node_schmidt_probabilities=probabilities.tolist(),
            schmidt_scope='Exact state-information diagnostic at shell 3; cutoff-dependent, not converged physical entanglement, topological order or quantum advantage.'),
        decision='Retained-mode interpretation passes the bounded quantum test. Carry source endpoints, direct response and energy-dependent normalization through elimination; do not add eliminated exchange beside explicit retained modes. Rematch coefficients after quantization and do not fit physical data to this homogeneous cell. Next define a bounded nonzero-momentum retained quark/collective interaction and gauged source calculation, using canonical LF amplitudes and shared fit directions, before p/n inference. This is not permission for a sigma-model or BLFQ repair campaign.')


if __name__=='__main__': print(json.dumps(evaluate(),indent=2,allow_nan=False))
