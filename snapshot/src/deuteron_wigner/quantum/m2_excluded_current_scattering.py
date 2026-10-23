"""Number-current qg/gg scattering inside qqqgg, per g_s².

Canonical j_g^+=f A_perp d_-A_perp has positive adjoint charge and
mode factor (k_out+k_in)/(2 sqrt(k_out k_in)). This supplies the mixed
and adjoint number-current scattering terms, not their pair-annihilation,
instantaneous-fermion, A^4 or cross-channel partners.
"""
from functools import lru_cache
from itertools import combinations
from math import pi, sqrt
import numpy as np
from scipy import sparse

from deuteron_wigner.microscopic.h0.color import representation_generators
from .m2_canonical_mixed_current import canonical_gluon_current_factor
from .m2_coupled_pair_action import _pair, _width, _contact_shell
from .m2_excluded_fermion_scattering import _pair_rotation
from .m2_excluded_fock_kinematics import SPECIES, color_basis, frame, _act


@lru_cache(maxsize=10, typed=True)
def current_pair_color(sector, left, right):
    species=SPECIES[sector]
    pair,_=_pair((left,right),len(species))
    dims=tuple(8 if s=="g" else 3 for s in species)
    u=color_basis(sector)
    tensor=u.reshape(dims+(u.shape[1],))
    reps=["8" if species[i]=="g" else "3bar" if species[i]=="qbar" else "3" for i in pair]
    raw=sum(_act(a,_act(b,tensor,right),left) for a,b in zip(
        representation_generators(reps[0]),representation_generators(reps[1]))).reshape(u.shape)
    reduced=u.conj().T@raw
    if np.linalg.norm(raw-u@reduced)>2e-12:
        raise ArithmeticError("number-current scattering leaves complete singlet space")
    reduced.setflags(write=False)
    return reduced


def five_body_contact(sector,incoming,outgoing,pair,b_GeV,cutoff):
    """Constituent-independent transverse coincidence on the fixed five-body shell."""
    fi,fo=frame(sector,tuple(incoming)),frame(sector,tuple(outgoing))
    pair,_=_pair(pair,5)
    if sum(incoming)!=sum(outgoing) or any(incoming[i]!=outgoing[i] for i in range(5) if i not in pair):
        raise ValueError("pair total and spectator modes must be conserved")
    b=_width(b_GeV);i,j=pair
    ai=fi.fractions[i]*fi.fractions[j]/(fi.fractions[i]+fi.fractions[j])
    ao=fo.fractions[i]*fo.fractions[j]/(fo.fractions[i]+fo.fractions[j])
    ri=_pair_rotation(sector,fi.modes,pair,cutoff)
    ro=_pair_rotation(sector,fo.modes,pair,cutoff)
    return b*b/pi*sqrt(ai*ao)*(ro.T@_contact_shell(4,cutoff,cutoff)@ri)


def current_scattering_per_gs2(space,*,b_GeV,channel):
    """Explicit qg or gg component, all distinct pairs, no zero-transfer inverse."""
    if space.sector!="qqqgg" or channel not in ("qg","gg"):
        raise ValueError("explicit qg/gg channel in the qqqgg sector required")
    _width(b_GeV)
    species=SPECIES[space.sector]
    pairs=[(i,j) for i,j in combinations(range(5),2)
           if "".join((species[i],species[j]))==channel]
    lookup={m:i for i,m in enumerate(space.longitudinal)}
    no,nc,ns=len(space.orbitals),space.ncolor,space.nspin
    spins=np.arange(ns);rows,cols,values=[],[],[]
    for li,incoming in enumerate(space.longitudinal):
        for i,j in pairs:
            color=current_pair_color(space.sector,i,j)
            for ki in range(2 if species[i]=="g" else 1,incoming[i]+incoming[j],2):
                transfer=ki-incoming[i]
                if transfer==0:
                    continue
                outgoing=list(incoming);outgoing[i],outgoing[j]=ki,incoming[i]+incoming[j]-ki
                outgoing=tuple(outgoing)
                factor=2*space.total_k2/(pi*transfer**2)
                for k in (i,j):
                    if species[k]=="g":
                        factor*=canonical_gluon_current_factor(outgoing[k],incoming[k])
                block=factor*np.kron(five_body_contact(space.sector,incoming,outgoing,(i,j),
                                                       b_GeV,space.cutoff),color)
                a,b=np.nonzero(block)
                ir=(lookup[outgoing]*no*nc+a[:,None])*ns+spins
                ic=(li*no*nc+b[:,None])*ns+spins
                pr,pc=space.lookup[ir],space.lookup[ic]
                keep=(pr>=0)&(pc>=0);outside=(pr<0)&(pc>=0)
                v=np.broadcast_to(block[a,b][:,None],keep.shape)
                if np.linalg.norm(v[outside])>2e-12:
                    raise ArithmeticError("number-current scattering violates Iz/Jz")
                rows.append(pr[keep]);cols.append(pc[keep]);values.append(v[keep])
    if not values:
        return sparse.csr_matrix((space.dimension,space.dimension),dtype=complex)
    return sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                             shape=(space.dimension,space.dimension)).tocsr()

