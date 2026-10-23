"""Replay canonical-spin correspondence; no fit of physical parameters or writes."""
import json
import numpy as np

from deuteron_wigner.quantum.path_dirac_matching import (
    canonical_map,direct_dirac_map,constrained_good_map,gap,spin_components,
    path_dictionary,coordinate_image,fit_tangent,energy_coefficient,
    spin_loss,spin_retention_loss,
)
from deuteron_wigner.quantum.lf_path_states import PathCarrier


def evaluate():
    c,p=path_dictionary();cases=[]
    for mass in (0.,.27,.54):
        carrier=PathCarrier(c.cell,c.momenta[::2],c.quarks,mass)
        v=canonical_map(carrier);d=gap(carrier);exact=-v/d[:,None,None]
        results={}
        for label,indices in (('spin_blind',(0,)),('spin_diagonal',(0,3)),('spin_complete',(0,1,2,3))):
            result=fit_tangent(v,d,coordinate_image(carrier,p,indices))
            results[label]={name:result[name] for name in ('rank','threshold','wave_error','relative_wave_error','energy_deficit_GeV')}
            results[label]['singular_values']=result['singular_values'].tolist()
            results[label]['max_coefficient_magnitude']=float(np.max(abs(result['coefficients'])))
            results[label]['spin_retention_loss_xyz']=spin_retention_loss(result['wave']).tolist()
        cases.append(dict(mass_GeV=mass,full_Dirac_map_error=float(np.linalg.norm(v-direct_dirac_map(carrier))),
            independent_good_constraint_error=float(np.linalg.norm(v-constrained_good_map(carrier))),
            pauli_component_squared_norms_GeV2=np.sum(abs(spin_components(v))**2,axis=0).tolist(),
            free_gaps_GeV=d.tolist(),exact_E2_minus_common_C0_GeV=energy_coefficient(v,d,exact),
            exact_spin_retention_loss_xyz=spin_retention_loss(exact).tolist(),
            exact_parent_helicity_alignment_loss=spin_loss(exact).tolist(),images=results))
    old=fit_tangent(canonical_map(c),gap(c),coordinate_image(c,dict(list(p.items())[:3]),(0,1,2,3)))
    extended=PathCarrier(c.cell,[*c.momenta[::2],(.25,.2,.2)],c.quarks,c.mass)
    bigger=fit_tangent(canonical_map(extended),gap(extended),coordinate_image(extended,p,(0,1,2,3)))
    return dict(scope='Canonical one-emission map and gap-weighted tangent correspondence, not binding or a physical self-energy',
        date='2026-09-11',parent_quarks_GeV=c.quarks.tolist(),
        gluon_spatial_modes_GeV=c.momenta[::2].tolist(),
        cell_lengths_GeV_inverse=list(c.cell.lengths),
        paths={name:[list(r) for r in path.vertices] for name,path in p.items()},
        cases=cases,preceding_three_path_spin_complete_rank=old['rank'],
        preceding_three_path_relative_error=old['relative_wave_error'],
        enlarged_three_spatial_mode_rank=bigger['rank'],enlarged_map_dimension=24,
        enlarged_three_spatial_mode_relative_error=bigger['relative_wave_error'],
        enlarged_three_spatial_mode_energy_deficit_GeV=bigger['energy_deficit_GeV'],
        canonical_node_retains_full_spin_tensor=True,
        pauli_coefficients_are_new_independent_physical_couplings=False,
        decorated_Wilson_state_qualified_beyond_first_field_order=False,
        complete_self_energy_matching_or_physical_binding=False,
        nucleon_axial_or_EM_observable=False,production_model_selected=False,
        local_Gauss_law_or_dynamical_covariance_proved=False,
        new_flux_register_or_BLFQ_H2_repair=False,public_push=False)


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2))
