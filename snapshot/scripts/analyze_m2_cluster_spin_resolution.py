"""Attribute a same-coordinate Q-resolution change without fitting it away.

The three-term split is an explicit telescoping convention (new norms/states
on the numerator change, old current on the state change), not a unique
physical separation. Each family and raw coupling order remains resolved.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from scripts.analyze_m2_cluster_spin import read_block, analyze
from deuteron_wigner.quantum.m2_cluster_spin_diagnostic import (
    complete_elastic_spin_amplitudes, elastic_orientation_analysis)


def block_change(old, new):
    """Exact numerator / both-norm / normalized-state telescoping identity."""
    a, b = old, new
    c0, c1 = a['family_coefficients'], b['family_coefficients']
    wf0, wi0, wf1, wi1 = (x[k] for x,k in (
        (a,'out_W'),(a,'in_W'),(b,'out_W'),(b,'in_W')))
    def sandwich(f, c, i):
        return np.einsum('ab,knmbc,cd->knmad', f, c, i, optimize=True)
    cbar0 = sandwich(wf0,c0,wi0)
    dc = sandwich(wf1,c1-c0,wi1)
    dn = sandwich(wf1-wf0,c0,wi1)+sandwich(wf0,c0,wi1-wi0)
    vf0,vi0,vf1,vi1 = (x[k] for x,k in (
        (a,'out_state'),(a,'in_state'),(b,'out_state'),(b,'in_state')))
    def amplitude(vf, c, vi):
        return np.einsum('a,knmab,b->knm',vf.conj(),c,vi,optimize=True)
    pieces=np.array([amplitude(vf1,dc,vi1),amplitude(vf1,dn,vi1),
        amplitude(vf1-vf0,cbar0,vi1)+amplitude(vf0,cbar0,vi1-vi0)])
    difference=amplitude(vf1,sandwich(wf1,c1,wi1),vi1)-amplitude(vf0,cbar0,vi0)
    # Scale this algebraic roundoff check for arbitrary diagnostic inputs;
    # the saved physical-scale packet is separately checked absolutely.
    scale=max(1.,float(np.max(np.abs(pieces))),float(np.max(np.abs(difference))))
    np.testing.assert_allclose(pieces.sum(axis=0),difference,atol=4e-14*scale,rtol=0)
    return pieces


def compare(old_paths,new_paths):
    r0,s0=analyze(old_paths,symmetry_complete=True)
    r1,s1=analyze(new_paths,symmetry_complete=True)
    if r0['families']!=r1['families']:
        raise ValueError('same named current families required')
    for key in r0['parameters']:
        if key not in ('q5','q6') and r0['parameters'][key]!=r1['parameters'][key]:
            raise ValueError('only Q resolution may change: '+key)
    for key in ('target','total_k2','retained_cutoff','transfer_GeV',
                'mean_total_transverse_GeV','physical_leg_prescription',
                'smooth_prescription','field_envelope_prescription','isospin_prescription',
                'charge','loop_flavor_charges','quadrature_order'):
        if r0['metadata'][key]!=r1['metadata'][key]:
            raise ValueError('source/current comparison mismatch: '+key)
    old=[read_block(p) for p in old_paths];new=[read_block(p) for p in new_paths]
    by_spin=lambda blocks:{(i['metadata']['jz_out_twice'],i['metadata']['jz_in_twice']):a for i,a in blocks}
    old,new=by_spin(old),by_spin(new)
    # One common phase preserves the independently fixed relative spin phase.
    overlap=np.vdot(old[(1,1)]['in_state'],new[(1,1)]['in_state'])
    if abs(overlap)<1e-8:raise ValueError('isolated state branch cannot be aligned')
    phase=np.exp(-1j*np.angle(overlap))
    blocks=[]
    for h in ((1,1),(-1,1)):
        a,b=old[h],dict(new[h])
        np.testing.assert_allclose(a['parity_map'],b['parity_map'],atol=0,rtol=0)
        for k in ('out_state','in_state'):b[k]=phase*b[k]
        blocks.append(block_change(a,b))
    delta=r0['metadata']['transfer_GeV'];g=r0['parameters']['g']
    pieces=complete_elastic_spin_amplitudes(*blocks,delta)
    summed=np.einsum('n,sknmab->smab',g**np.arange(5),pieces)
    np.testing.assert_allclose(summed.sum(axis=0),s1['current']-s0['current'],atol=5e-14,rtol=0)
    args=dict(energy_GeV2=r0['energy_GeV2'],total_pplus_GeV=r0['parameters']['Pplus_GeV'])
    coeff=np.array([elastic_orientation_analysis(v,delta,**args)['five']['coefficients'] for v in summed])
    common=elastic_orientation_analysis(s1['current'],delta,**args)['five']['coefficients']
    mass_basis=s1['five_coefficients']-common
    np.testing.assert_allclose(coeff.sum(axis=0)+mass_basis,
        s1['five_coefficients']-s0['five_coefficients'],atol=5e-13,rtol=0)
    pair=lambda z:[float(z.real),float(z.imag)]
    labels=('raw_numerator_at_new_norms_and_states','both_norm_maps_at_new_states','normalized_state_change')
    report=dict(scope='same-P/K source coordinates, fixed-prescription Q sensitivity; no convergence or physical claim',
        old_blocks=[str(p) for p in old_paths],new_blocks=[str(p) for p in new_paths],
        old_energy_GeV2=r0['energy_GeV2'],new_energy_GeV2=r1['energy_GeV2'],
        energy_change_GeV2=r1['energy_GeV2']-r0['energy_GeV2'],
        common_phase=[float(phase.real),float(phase.imag)],
        coefficient_order=['F1_separated','F2_separated','B1','B2','B3'],
        total_amplitude_change_norm=float(np.linalg.norm(s1['current']-s0['current'])),
        coefficient_change=[pair(z) for z in s1['five_coefficients']-s0['five_coefficients']],
        external_mass_basis_change=[pair(z) for z in mass_basis],
        attribution={label:dict(amplitude_norm=float(np.linalg.norm(summed[k])),
            coefficients_in_old_mass_basis=[pair(z) for z in coeff[k]]) for k,label in enumerate(labels)},
        two_structure_residuals=[r0['two_structure_residual_norm'],r1['two_structure_residual_norm']],
        independent_blocks_per_resolution=2,kinematically_completed_blocks=['pm','mm'],
        split_is_convention_not_unique_physics=True,physical=False,covariance_repaired=False)
    return report,dict(family_order_change_pieces=pieces,total_change_pieces=summed,
        old_current=s0['current'],new_current=s1['current'],old_mass_coefficient_pieces=coeff,
        external_mass_basis_change=mass_basis)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',nargs=2,type=Path,required=True)
    p.add_argument('--new',nargs=2,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();dest=a.output
    if dest.suffix!='.json' or dest.exists() or dest.with_suffix('.npz').exists():
        p.error('fresh JSON/NPZ paths required')
    report,arrays=compare(a.old,a.new);report['full_matrix_file']=dest.with_suffix('.npz').name
    with dest.with_suffix('.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    with dest.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
