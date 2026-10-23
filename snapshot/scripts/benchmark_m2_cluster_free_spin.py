"""Independent P/K resolution diagnosis in the exact zero-coupling limit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from deuteron_wigner.quantum.m2_cluster_spin_diagnostic import (
    free_spin_fiber,free_current_parts,elastic_orientation_analysis)
from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map
from deuteron_wigner.quantum.m2_smooth_longitudinal_projector import (
    SmoothMinusProjectorContext,SMOOTH_PRESCRIPTION)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--K2',type=int,required=True);p.add_argument('--cutoff',type=int,required=True)
    p.add_argument('--target',choices=('PROTON','NEUTRON'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();dest=a.output
    if dest.exists() or dest.with_suffix('.npz').exists():p.error('fresh output required')
    start=time.monotonic();delta=(.12,.16);P=1.3
    models=[free_spin_fiber(total_k2=a.K2,cutoff=a.cutoff,target=a.target,jz_twice=h,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.) for h in (1,-1)]
    energies,vectors=np.linalg.eigh(models[0].matrix)
    count=int(np.count_nonzero(abs(energies-energies[0])<1e-10))
    u=vectors[:,:count]
    down=1j*transverse_parity_map(models[1],models[0])@u
    states=[u,down];arrays=dict(ground_energies=energies,up_subspace=u,down_subspace=down)
    packet=np.zeros((2,4,2,2,count,count),complex)
    print(a.target,a.K2,a.cutoff,'dimension',len(energies),'ground multiplicity',count,flush=True)
    context=SmoothMinusProjectorContext(mean_total_transverse_GeV=(0.,0.),prescription=SMOOTH_PRESCRIPTION)
    for f in range(2):
        for i in range(2):
            physical,projector=free_current_parts(models[f],models[i],delta,total_pplus_GeV=P,
                charge='electromagnetic',quadrature_order=8,minus_context=context)
            for k,c in enumerate((physical,projector)):
                packet[k,:,f,i]=np.einsum('ar,mab,bs->mrs',states[f].conj(),c,states[i],optimize=True)
            if a.K2==9 and a.cutoff==0:
                arrays[f'raw_g0_{f}{i}']=physical+projector
    arrays['subspace_current_parts']=packet
    report=dict(target=a.target,total_k2=a.K2,retained_cutoff=a.cutoff,g=0.,b_GeV=.4,mq_GeV=.35,
        mg_GeV=0.,Pplus_GeV=P,transfer_GeV=delta,energy_GeV2=float(energies[0]),
        dimension=len(energies),ground_multiplicity=count,subspace_current_preserved=True,
        physical=False,covariance_repaired=False,operator_order='g0 only; all Q return words vanish exactly',
        regulator_variation_is_vacuous_at_g0=True,full_matrix_file=dest.with_suffix('.npz').name)
    if count==1:
        amplitudes=packet[...,0,0]
        arrays['spin_current_parts']=amplitudes
        report['orientation']={}
        for name,value in (('physical',amplitudes[0]),('projector',amplitudes[1]),('full',sum(amplitudes))):
            fit=elastic_orientation_analysis(value,delta,energy_GeV2=energies[0],total_pplus_GeV=P)
            report['orientation'][name]=dict(two_residual=fit['two']['residual_norm'],
                five_residual=fit['five']['residual_norm'],coefficients=[[z.real,z.imag] for z in fit['five']['coefficients']],
                continuity_norm=float(np.linalg.norm(fit['continuity'])))
    else:
        report['orientation']=None
        report['state_selection']='degenerate source subspace preserved; no arbitrary spin-half state selected'
    report['seconds']=time.monotonic()-start
    with dest.with_suffix('.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    with dest.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
