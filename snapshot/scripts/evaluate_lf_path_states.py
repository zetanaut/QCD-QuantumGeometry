"""Replay qualified LF path-state coefficients; no fit or output-file writes."""
import json
import numpy as np

from deuteron_wigner.quantum.lf_path_states import PathCarrier,reference,pullback


def complex_parts(x):
    x=np.asarray(x)
    return dict(real=x.real.tolist(),imag=x.imag.tolist())


def evaluate():
    c,paths=reference();waves={name:c.wave(p) for name,p in paths.items()}
    energies=c.free_energies();names=list(paths)
    gram=np.array([[pullback(waves[a],waves[b]) for b in names] for a in names])
    free=np.array([[pullback(waves[a],waves[b],lambda x:energies[:,None]*x)
                    for b in names] for a in names])
    independent,n=c.ode_wave(paths['LF_detour'])
    coarse,nc=c.ode_wave(paths['LF_detour'],rtol=2e-8,atol=2e-10)
    v=waves['LF_detour'];ops=c.oscillator_creators()
    f=c.integrals(paths['LF_detour'])[0]
    unordered=-.5*c.field_action(f,c.field_action(f,v[0],ops),ops)
    no_two=v.copy();no_two[2,33:]=0
    probe=c.integrals(paths['different_loop'])[0]
    quadratic=lambda x:c.field_action(probe,c.field_action(probe,x,ops),ops)
    extended=PathCarrier(c.cell,[*c.momenta[::2],(.25,.2,.2)],c.quarks,c.mass)
    extra=extended.wave(paths['LF_detour'])
    columns=np.column_stack([sum(.6**j*waves[name][j] for j in range(3)).ravel() for name in names])
    return dict(
        scope='Order-gs^2 path-dressed canonical LF carrier; no physical bound state, local Gauss-law proof or production action',
        date='2026-09-11',parent_quarks_GeV=c.quarks.tolist(),quark_mass_GeV=c.mass,
        cell_lengths_GeV_inverse=list(c.cell.lengths),
        gluon_spatial_momenta_GeV=c.momenta[::2].tolist(),helicities=[-1,1],colors=8,
        boson_configurations=len(c.configs),raw_color_dimension=27,
        raw_wave_dimension=27*len(c.configs),all_paths_start_and_end_at_origin=True,
        paths={name:[list(x) for x in p.vertices] for name,p in paths.items()},
        paths_order=names,
        coefficients={name:dict(one_gluon_norm_squared=float(np.linalg.norm(w[1])**2),
            second_order_vacuum_amplitude=complex_parts(np.vdot(c.seed,w[2,0])),
            second_order_two_gluon_norm=float(np.linalg.norm(w[2,33:])),
            diagonal_norm_coefficients=complex_parts(pullback(w,w)),
            free_energy_shift_gs2_GeV=float((pullback(w,w,lambda x:energies[:,None]*x)[2]
                -energies[0]*pullback(w,w)[2]).real)) for name,w in waves.items()},
        Gram_gs2=complex_parts(gram[:,:,2]),free_Pminus_gs2_GeV=complex_parts(free[:,:,2]),
        bare_Pminus_GeV=float(energies[0]),
        polynomial_column_singular_values_at_gs_point6=np.linalg.svd(columns,compute_uv=False).tolist(),
        polynomial_higher_order_terms_are_not_physical_fourth_order=True,
        independent_oscillator_wave_error=float(np.linalg.norm(independent-v)),ODE_evaluations=n,
        coarse_oscillator_wave_error=float(np.linalg.norm(coarse-v)),coarse_ODE_evaluations=nc,
        quadrature_16_to_32_wave_error=float(np.linalg.norm(c.wave(paths['LF_detour'],16)-c.wave(paths['LF_detour'],32))),
        unordered_full_second_order_wave_error=float(np.linalg.norm(v[2]-unordered)),
        unordered_two_gluon_wave_error=float(np.linalg.norm((v[2]-unordered)[33:])),
        omitted_two_gluon_norm_coefficient_error=float(np.linalg.norm(pullback(v,v)-pullback(no_two,no_two))),
        omitted_two_gluon_quadratic_probe_gs2_error=float(abs(pullback(v,v,quadratic)[2]-pullback(no_two,no_two,quadratic)[2])),
        enlarged_mode_domain=dict(spatial_modes=3,boson_configurations=len(extended.configs),
            old_one_gluon_amplitude_error=float(np.linalg.norm(extra[1,1:33]-v[1,1:33])),
            one_gluon_norm_squared=float(np.linalg.norm(extra[1])**2),
            vacuum_coefficient_change=complex_parts(np.vdot(c.seed,extra[2,0]-v[2,0])),
            continuum_convergence_established=False),
        additional_orthogonal_flux_register=False,local_Gauss_law=False,
        production_binding_action=False,GTMD_process_links=False,
        physical_nucleon_prediction=False,old_BLFQ_H2_rebuilt=False,public_push=False)


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2))
