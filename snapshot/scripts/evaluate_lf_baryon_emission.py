"""Replay first-emission baryon controls, not physical p/n predictions."""
import json
from math import sqrt

import numpy as np

from deuteron_wigner.quantum.finite_lf_coupling import reference_inputs
from deuteron_wigner.quantum.lf_baryon_emission import (
    EmissionLayer, norm2, difference, total_generator, spin_map, emission_leaf, emission_contact,
)


def evaluate():
    cell,m,g,_=reference_inputs(); p=(.375,.2,0); mq=.27; gs=.43
    obj=EmissionLayer(cell,[p],mq,g,gs,1)
    cases=[]
    for species,q in (('p',1),('n',0)):
        for spin in (-1,1):
            state=obj.interpolator(p,species,spin)
            out=obj.apply(state); wave=obj.pulse(state,m,12)
            single=obj.tensor_reference(state,slots=[0])
            cases.append(dict(species=species,twice_canonical_input_spin=spin,
                seed_norm_squared=norm2(state),output_mask_groups=len(out),
                emission_coefficient_norm_squared_GeV2=norm2(out),
                tensor_wave_error_GeV=difference(out,obj.tensor_reference(state)),
                total_charge_error_GeV=difference(total_generator(obj,out,charge=True),{k:q*v for k,v in out.items()}),
                max_total_color_error_GeV=max(sqrt(norm2(total_generator(obj,out,color=a))) for a in range(8)),
                missing_two_emitters_error_GeV=difference(out,single),
                spin_relabelling_error_GeV=difference(out,obj.apply(obj.interpolator(p,species,spin,canonical=False))),
                pulse_phi_norm_squared_coefficient=norm2({k:v for k,v in wave.items() if k[0]=='phi'}),
                pulse_gluon_norm_squared_coefficient=norm2({k:v for k,v in wave.items() if k[0]=='g'})))
    scans=[]
    for shell in range(4):
        # Collinear seed exists also at shell zero; scan common fixed input.
        pilot=EmissionLayer(cell,[(.375,0,0)],mq,g,gs,shell)
        wave=pilot.pulse(pilot.interpolator((.375,0,0),'p',1),m,12)
        scans.append(dict(shell=shell,channel_routes=len(pilot.routes),
            output_mask_groups=len(wave),phi_norm_squared_coefficient=norm2({k:v for k,v in wave.items() if k[0]=='phi'}),
            gluon_norm_squared_coefficient=norm2({k:v for k,v in wave.items() if k[0]=='g'})))
    tiny=EmissionLayer(cell,[(.375,0,0)],mq,g,gs,0);seed=tiny.interpolator((.375,0,0),'p',1)
    a=(.125,.2,-.2);k=(.25,-.2,.2);incoming=(.375,.2,.2)
    contacts=[]
    for e in np.eye(4):
        direct=emission_contact(a,k,incoming,mq,e,cell.volume)
        contacts.append(dict(H_derivative_wrt_source_GeV_dimensionless_norm=float(np.linalg.norm(direct)),
            independent_instantaneous_error=float(np.linalg.norm(direct-emission_contact(a,k,incoming,mq,e,cell.volume,covariant=True)))))
    r=spin_map(p,mq)
    return dict(scope='first-order free-reference baryon emission, not physical nucleons or a production Hamiltonian',
        inputs=dict(cell_lengths_GeV_minus1=cell.lengths,quark_mass_GeV=mq,
            collective_mass_squared_GeV2=m.tolist(),Yukawas=g.tolist(),gs=gs,
            initial_each_quark_momentum_GeV=p,duration_GeV_minus1=12),
        derived_spin_map_real=r.real.tolist(),derived_spin_map_imag=r.imag.tolist(),
        leaf_constraint_error=float(np.linalg.norm(emission_leaf((.125,0,-.2),(.25,.2,.2),p,mq)
            -emission_leaf((.125,0,-.2),(.25,.2,.2),p,mq,constraint=True))),
        cases=cases,window_scan=scans,
        direct_time_wave_error=difference(tiny.pulse(seed,m,12),tiny.pulse(seed,m,12,direct=True)),
        crossed_fermion_source_contact_components=contacts,
        qualifications=dict(complete_emitter_interference=True,independent_exterior_particle_and_graded_controls=True,
            unequal_momenta_and_variable_quark_number_tested=True,
            source_contact_is_not_complete_baryon_EM_response=True,
            full_quantum_bound_nucleon=False,composite_covariance_repaired=False,
            physical_matching_or_fit=False,global_Fock_or_continuum_completion=False,
            old_BLFQ_H2_model_rebuilt=False,public_push=False))


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2))
