"""Reconstruct the four independent elastic spin blocks and retain every defect."""
import argparse
import json
from pathlib import Path
import numpy as np
from deuteron_wigner.quantum.m2_cluster_spin_diagnostic import elastic_orientation_analysis,complete_elastic_spin_amplitudes
from deuteron_wigner.quantum.m2_current_covariance_diagnostic import current_covariance_diagnostic


def read_block(path):
    path=Path(path);info=json.loads(path.read_text())
    name=Path(info['full_matrix_file'])
    if str(name)!=name.name or name.suffix!='.npz':raise ValueError('adjacent numeric NPZ required')
    with np.load(path.parent/name,allow_pickle=False) as data:a={k:data[k] for k in data.files}
    return info,a


def analyze(paths,*,symmetry_complete=False):
    blocks=[read_block(p) for p in paths]
    if len(blocks)!=(2 if symmetry_complete else 4):raise ValueError('explicit independent spin-block set required')
    first=blocks[0][0];families=tuple(first['families']);nf=len(families)
    amplitudes=np.zeros((nf,5,4,2,2),complex);seen=set();normalization_effect=np.zeros((4,2,2),complex)
    for info,a in blocks:
        for key in ('parameters','energy_GeV2','families','intrinsic_parity'):
            if info[key]!=first[key]:raise ValueError('spin packet mismatch: '+key)
        for key in ('target','energy_out_GeV2','energy_in_GeV2','transfer_GeV','mean_total_transverse_GeV',
                    'charge','loop_flavor_charges','total_k2','retained_cutoff'):
            if info['metadata'][key]!=first['metadata'][key]:raise ValueError('mixed current metadata: '+key)
        m=info['metadata'];f,i=[(1,-1).index(m[k]) for k in ('jz_out_twice','jz_in_twice')]
        if (f,i) in seen:raise ValueError('duplicate spin block')
        seen.add((f,i))
        uf,ui=a['out_raw_state'],a['in_raw_state']
        np.testing.assert_allclose(a['family_coefficients'].sum(axis=0),a['current_coefficients'],atol=4e-14,rtol=0)
        value=np.einsum('a,knmab,b->knm',uf.conj(),a['family_coefficients'],ui,optimize=True)
        np.testing.assert_allclose(value,a['family_amplitudes'],atol=3e-14,rtol=0)
        amplitudes[:,:,:,f,i]=value
        total=np.einsum('n,nm->m',first['parameters']['g']**np.arange(5),value.sum(axis=0))
        normalization_effect[:,f,i]=total*(1/(np.linalg.norm(uf)*np.linalg.norm(ui))-1)
    m=first['metadata'];delta=m['transfer_GeV'];energy=first['energy_GeV2'];P=first['parameters']['Pplus_GeV'];g=first['parameters']['g']
    if m['energy_out_GeV2']!=energy or m['energy_in_GeV2']!=energy or any(m['mean_total_transverse_GeV']):
        raise ValueError('elastic same-pole zero-mean packet required, not unequal-anchor quantum transitions')
    if symmetry_complete:
        if seen!={(0,0),(1,0)} or first['intrinsic_parity']!=1:
            raise ValueError('explicit positive-parity pp/mp blocks required for kinematic completion')
        amplitudes=complete_elastic_spin_amplitudes(amplitudes[...,0,0],amplitudes[...,1,0],delta)
        normalization_effect=complete_elastic_spin_amplitudes(normalization_effect[...,0,0],normalization_effect[...,1,0],delta)
    elif seen!={(0,0),(0,1),(1,0),(1,1)}:raise ValueError('incomplete spin packet')
    args=dict(energy_GeV2=energy,total_pplus_GeV=P)
    order_amplitudes=amplitudes.sum(axis=0)
    total=np.einsum('n,nmab->mab',g**np.arange(5),order_amplitudes)
    q=elastic_orientation_analysis(total,delta,**args)
    cross=current_covariance_diagnostic(total[0],total[2:],delta,mass_GeV=np.sqrt(energy),total_pplus_GeV=P)
    family_coeff=np.zeros((nf,5,5),complex)
    family_two_residual=np.zeros_like(amplitudes);family_five_residual=np.zeros_like(amplitudes)
    for k in range(nf):
        for n in range(5):
            fit=elastic_orientation_analysis(amplitudes[k,n],delta,**args)
            family_coeff[k,n]=fit['five']['coefficients']
            family_two_residual[k,n]=fit['two']['residual'];family_five_residual[k,n]=fit['five']['residual']
    np.testing.assert_allclose(np.einsum('n,knj->j',g**np.arange(5),family_coeff),q['five']['coefficients'],atol=3e-13,rtol=0)
    for name,arr in (('two',family_two_residual),('five',family_five_residual)):
        np.testing.assert_allclose(np.einsum('n,knmab->mab',g**np.arange(5),arr),q[name]['residual'],atol=4e-14,rtol=0)
    def pair(z):return [float(z.real),float(z.imag)]
    groups={}
    for label,indices in (
        ('free_one_body',[families.index('free')]),
        ('canonical_contacts',[j for j,s in enumerate(families) if s.endswith('_canonical')]),
        ('field_regulator_contacts',[j for j,s in enumerate(families) if s.endswith('_regulator')]),
        ('paired_Wick',[families.index('Wick_paired')]),
        ('oscillator_projectors',[j for j,s in enumerate(families) if s.startswith('projector_')])):
        coef=np.einsum('n,knj->j',g**np.arange(5),family_coeff[indices])
        contribution=np.einsum('n,knmab->mab',g**np.arange(5),amplitudes[indices])
        groups[label]=dict(coefficients=[pair(z) for z in coef],amplitude_norm=float(np.linalg.norm(contribution)))
    report=dict(scope='finite-elastic common-regulated H2 spin/rotation diagnostic; no covariance repair or physical calibration',
        target=m['target'],metadata=m,parameters=first['parameters'],energy_GeV2=energy,
        input_blocks=[str(p) for p in paths],families=families,
        independent_spin_blocks=len(blocks),kinematically_completed_blocks=['pm','mm'] if symmetry_complete else [],
        two_structure_residual_norm=q['two']['residual_norm'],five_structure_residual_norm=q['five']['residual_norm'],
        two_structure_relative_residual=q['two']['residual_norm']/np.linalg.norm(total),
        five_structure_coefficients=[pair(z) for z in q['five']['coefficients']],
        five_structure_condition=q['five']['condition'],GM_plus=pair(cross.magnetic_from_plus),
        GM_transverse=pair(cross.magnetic_from_transverse),
        GM_disagreement=pair(cross.magnetic_from_plus-cross.magnetic_from_transverse),
        continuity_norm=float(np.linalg.norm(q['continuity'])),groups=groups,
        omitted_norm_map_effect_norm=float(np.linalg.norm(normalization_effect)),
        maximum_pole_residual=max(max(b[0]['pole_residuals_GeV2']) for b in blocks),
        exact_elastic_state_not_old_anchor_fit=True,physical=False,covariance_repaired=False)
    arrays=dict(current=total,order_amplitudes=order_amplitudes,family_order_amplitudes=amplitudes,
        external_tensors=q['basis'],five_coefficients=q['five']['coefficients'],
        two_residual=q['two']['residual'],five_residual=q['five']['residual'],
        family_order_five_coefficients=family_coeff,family_order_two_residual=family_two_residual,
        family_order_five_residual=family_five_residual,omitted_norm_map_effect=normalization_effect)
    return report,arrays


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--blocks',nargs='+',type=Path,required=True)
    p.add_argument('--symmetry-complete',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();dest=a.output
    if dest.exists() or dest.with_suffix('.npz').exists():p.error('fresh output required')
    report,arrays=analyze(a.blocks,symmetry_complete=a.symmetry_complete);report['full_matrix_file']=dest.with_suffix('.npz').name
    with dest.with_suffix('.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    with dest.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
