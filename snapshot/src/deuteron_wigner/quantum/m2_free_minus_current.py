"""Exact free minus current, including N+2 outside oscillator neighbors.

This supplies a canonical one-body operator, NOT the interaction-dependent
minus current, its Wick/counterterm completion, or a physical current.
"""
import numpy as np
from scipy import sparse

from .m2_canonical_good_current import parton_charge
from .m2_constrained_fermion_current import ALPHA,GOOD_AXES,ANTI_AXES,bad_field_kernel
from .m2_coupled_longitudinal_jacobi import oscillator_shell,jacobi_frame,longitudinal_basis
from .m2_coupled_pilot import _displacement
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


def displaced_intrinsic_moments(dimension,cutoff,shift,weights,*,b_GeV):
    """D, <D kappa_j>, <D kappa_j kappa_k>; kappa=sum_a weights[a] Q_a.

    Embed ket Q and Q² in N+2 BEFORE taking the continuum displaced overlap.
    This is not multiplication of truncated N-shell D and momentum matrices.
    """
    shell=oscillator_shell(dimension,cutoff);extended=oscillator_shell(dimension,cutoff+2)
    d=len(shell[0])//2;n=len(shell);ne=len(extended)
    shift=np.asarray(shift,float);weights=np.asarray(weights,float);b=float(b_GeV)
    if (shift.shape!=(d,2) or weights.shape!=(d,)
            or not np.all(np.isfinite(np.r_[shift.ravel(),weights,b])) or b<=0):
        raise ValueError("finite shift/weights and positive width required")
    occupation=np.array(shell);ex=np.array(extended);lookup={q:i for i,q in enumerate(extended)}
    a=(shift[:,0]-1j*shift[:,1])/(2*b);alpha=np.r_[a,a.conj()]
    tables=np.array([[[_displacement(o,i,z) for i in range(cutoff+3)]
                       for o in range(cutoff+1)] for z in alpha])
    displacement=np.ones((n,ne),complex)
    for k in range(2*d):
        displacement*=tables[k,occupation[:,k,None],ex[None,:,k]]
    rows=[[],[]];columns=[[],[]];values=[[],[]]
    for column,q in enumerate(extended):
        for k in range(2*d):
            for step in (-1,1):
                target=list(q);target[k]+=step
                if target[k]<0 or sum(target)>cutoff+2:continue
                row=lookup[tuple(target)]
                v=b*weights[k%d]*np.sqrt(q[k]+(step==1))/2
                for j,phase in enumerate((1.,(-1j if k<d else 1j)*step)):
                    rows[j].append(row);columns[j].append(column);values[j].append(v*phase)
    operators=[sparse.csr_matrix((v,(r,c)),shape=(ne,ne))
               for v,r,c in zip(values,rows,columns)]
    ket=sparse.csr_matrix((np.ones(n),(np.arange(n),np.arange(n))),shape=(ne,n))
    first=np.array([displacement@(q@ket) for q in operators])
    second=np.array([[displacement@(qj@qk@ket) for qk in operators] for qj in operators])
    return displacement[:,:n],first,second


def projected_free_minus(dimension,cutoff,shift,weights,*,b_GeV,pplus_GeV,
                         outgoing_offset,incoming_offset,mass_GeV,species):
    """Charge-stripped spin/orbital tensor, shape (h_out,h_in,orb_out,orb_in).

    Both physical momenta are kappa+offset at the same positive parton p+.
    Antiquark reversal/axes/CAR-charge conventions match the full Dirac kernel.
    """
    pp=float(pplus_GeV);pf=np.r_[pp,np.asarray(outgoing_offset,float)]
    pi=np.r_[pp,np.asarray(incoming_offset,float)]
    if not np.isfinite(pp) or pp<=0:raise ValueError("positive parton P+ required")
    if species=="q":
        bf=bad_field_kernel(pf,mass_GeV);bi=bad_field_kernel(pi,mass_GeV);axes=GOOD_AXES
    elif species=="qbar":
        bf=bad_field_kernel(-pi,mass_GeV);bi=bad_field_kernel(-pf,mass_GeV);axes=ANTI_AXES
    else:raise ValueError("explicit q or qbar species required")
    density,first,second=displaced_intrinsic_moments(dimension,cutoff,shift,weights,b_GeV=b_GeV)
    derivatives=[a/(np.sqrt(2.)*pp) for a in ALPHA]
    def spin(op):return axes.conj().T@op@axes
    result=spin(bf.conj().T@bi)[:,:,None,None]*density
    for j,a in enumerate(derivatives):
        result+=spin(a@bi+bf.conj().T@a)[:,:,None,None]*first[j]
        for k,c in enumerate(derivatives):
            result+=spin(a@c)[:,:,None,None]*second[j,k]
    return result if species=="q" else result.swapaxes(0,1)


def free_minus_current_between(outgoing,incoming,transfer,*,total_pplus_GeV,charge,
                               mean_total_transverse=(0.,0.)):
    """Retained qqq/qqqg free J-/(2P+), including quark spin flips.

    No interaction/contact, Wick or matching contribution is implied.
    Both target Jz endpoints remain explicit; quark flavor is not mixed.
    """
    for name in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    delta=np.asarray(transfer,float);mean=np.asarray(mean_total_transverse,float);pp=float(total_pplus_GeV)
    if (delta.shape!=(2,) or mean.shape!=(2,) or not np.all(np.isfinite(np.r_[delta,mean,pp])) or pp<=0):
        raise ValueError("finite transverse vectors and positive total P+ required")
    sf=SPIN_FLAVOR_BASIS;lookup={q:i for i,q in enumerate(sf)};result=[];p=incoming
    for sector,uf,ui in zip(("qqq","qqqg","qqqg"),outgoing.bases,incoming.bases):
        d=2 if sector=="qqq" else 3;nc=d-1;n=len(oscillator_shell(d,p.cutoff));blocks=[]
        for modes in longitudinal_basis(sector,p.total_k2):
            frame=jacobi_frame(modes);current=sparse.csr_matrix((n*nc*64,)*2,dtype=complex)
            for active in range(3):
                x=frame.fractions[active]
                kernel=projected_free_minus(d,p.cutoff,frame.forward[:-1,active,None]*delta,
                    frame.inverse[active,:-1],b_GeV=p.width_GeV,pplus_GeV=x*pp,
                    outgoing_offset=x*mean+(1-x/2)*delta,incoming_offset=x*mean-x*delta/2,
                    mass_GeV=p.quark_mass_GeV,species="q")
                for hf in (0,1):
                    for hi in (0,1):
                        rows=[];columns=[];values=[]
                        for column,state in enumerate(sf):
                            if state[active]%2!=hi:continue
                            target=list(state);target[active]=2*(state[active]//2)+hf
                            rows.append(lookup[tuple(target)]);columns.append(column)
                            values.append(float(parton_charge(state[active],"q",charge)))
                        spin=sparse.csr_matrix((values,(rows,columns)),shape=(64,64))
                        current+=sparse.kron(kernel[hf,hi],sparse.kron(sparse.eye(nc),spin),format="csr")
            blocks.append(current)
        result.append((uf.getH()@sparse.block_diag(blocks,format="csr")@ui).toarray())
    return sparse.block_diag(result).toarray()
