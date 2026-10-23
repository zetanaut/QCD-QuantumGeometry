"""Physical-leg retained cubic current with raw color/flavor/spin coherence.

This supplies only the number-changing order-g contact, not the free,
quartic, Wick, projector, loop or complete matched H2 current.
"""
from dataclasses import dataclass
from operator import index
import numpy as np
from scipy import sparse
from deuteron_wigner.microscopic.h0.color import emitted_gluon_color_amplitudes
from .m2_coupled_longitudinal_jacobi import emission_routes,longitudinal_basis,oscillator_shell
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS
from .m2_canonical_good_current import parton_charge
from .m2_gaussian_cluster_current import PRESCRIPTION,cluster_cubic_current_orbitals,ClusterCurrentIntegral
from .m2_gaussian_field_regulator import GaussianClusterRegulator,_array
from .m2_gaussian_cluster_h2 import require_same_cluster_definition


@dataclass(frozen=True)
class RawClusterCubicCurrent:
    canonical: tuple
    regulator: tuple
    prescription: str
    complete_H2_matched_current = False
    physical = False

    @property
    def total(self):
        return tuple(a+b for a,b in zip(self.canonical,self.regulator))


def cluster_emission_current(total_k2,in_cutoff,out_cutoff,transfer,*,b_GeV,mass_GeV,
        total_pplus_GeV,gluon_helicity,charge,regulator,mean_total_transverse_GeV,
        prescription,quadrature_order):
    """Raw qqq→qqqg four-current, J/(2P+) per gs, all three emitters.

    Flavor charge multiplies each helicity-resolved orbital linearly. The
    gluon field normalization is already inside that orbital. There is one
    emitted gluon, hence no extra Bose factor. Spin flips are retained.
    """
    if isinstance(gluon_helicity,(bool,np.bool_)) or gluon_helicity not in (-1,1):
        raise ValueError("explicit physical gluon helicity required")
    if not isinstance(regulator,GaussianClusterRegulator) or prescription!=PRESCRIPTION:
        raise ValueError("explicit common cluster and physical-leg endpoint prescription required")
    _array(transfer,(2,),"transverse transfer")
    _array(mean_total_transverse_GeV,(2,),"mean transverse momentum")
    pp,b,m = float(total_pplus_GeV),float(b_GeV),float(mass_GeV)
    if (not np.all(np.isfinite([pp,b,m])) or min(pp,b)<=0 or m<0
            or isinstance(quadrature_order,(bool,np.bool_)) or index(quadrature_order)<2):
        raise ValueError("positive P+/width, nonnegative mass and exact path order at least two required")
    q,g = (longitudinal_basis(s,total_k2) for s in ("qqq","qqqg"))
    nq,ng = len(oscillator_shell(2,in_cutoff)),len(oscillator_shell(3,out_cutoff))
    qi,gi = ({s:i for i,s in enumerate(states)} for states in (q,g))
    sf = SPIN_FLAVOR_BASIS
    lookup = {state:i for i,state in enumerate(sf)}
    parton_charge(np.arange(4),"q",charge)
    entries = [[([],[],[]) for _ in range(4)] for _ in range(2)]
    for route in emission_routes(total_k2):
        color = emitted_gluon_color_amplitudes(route.emitter)[:,None]
        for hf in (0,1):
            for hi in (0,1):
                orbital = cluster_cubic_current_orbitals(route,in_cutoff,out_cutoff,b_GeV,mass_GeV,
                    (2*hf-1,gluon_helicity,2*hi-1),regulator=regulator,total_pplus_GeV=total_pplus_GeV,
                    transfer_GeV=transfer,mean_total_transverse_GeV=mean_total_transverse_GeV,
                    flavor_charge=1.,prescription=prescription,quadrature_order=quadrature_order)
                rows,cols,weights = [],[],[]
                for col,state in enumerate(sf):
                    if state[route.emitter]%2!=hi:
                        continue
                    target = list(state);target[route.emitter] = 2*(state[route.emitter]//2)+hf
                    rows.append(lookup[tuple(target)]);cols.append(col)
                    weights.append(float(parton_charge(state[route.emitter],"q",charge)))
                spin = sparse.csr_matrix((weights,(rows,cols)),shape=(64,64))
                spin_color = sparse.kron(color,spin,format="csr")
                for kind,components in enumerate((orbital.canonical,orbital.regulator)):
                    for mu,matrix in enumerate(components):
                        block = sparse.kron(sparse.csr_matrix(matrix),spin_color,format="coo")
                        r,c,v = entries[kind][mu]
                        r.extend(gi[route.outgoing]*ng*128+block.row)
                        c.extend(qi[route.incoming]*nq*64+block.col);v.extend(block.data)
    shape = (len(g)*ng*128,len(q)*nq*64)
    results = [tuple(sparse.coo_matrix((v,(r,c)),shape=shape).tocsr() for r,c,v in group)
               for group in entries]
    return RawClusterCubicCurrent(*results,prescription)


def cluster_cubic_current_between(outgoing,incoming,transfer,*,charge,
        mean_total_transverse_GeV,prescription,quadrature_order):
    """Project coherent emission and reverse-transfer absorption separately.

    Requires the actual common-regulated pilots on both sides. Different
    spin and oscillator endpoints are allowed, but not different field/Wick
    definitions or physical parameters. No extra gs or M² factor is applied.
    """
    if not require_same_cluster_definition(outgoing,incoming):
        raise ValueError("actual common-cluster regulated current endpoints required")
    for key in ("total_k2","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(outgoing,key)!=getattr(incoming,key):
            raise ValueError("current endpoints disagree on "+key)
    delta = np.asarray(transfer,float)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)):
        raise ValueError("explicit finite transverse transfer required")
    nf,ni = [u.shape[1] for u in outgoing.bases],[u.shape[1] for u in incoming.bases]
    of,oi = np.cumsum([0]+nf),np.cumsum([0]+ni)
    values = [np.zeros((4,sum(nf),sum(ni)),complex) for _ in range(2)]
    p = incoming
    for sector,hg in ((1,-1),(2,1)):
        def raw(d,ci,co):
            return cluster_emission_current(p.total_k2,ci,co,d,b_GeV=p.width_GeV,
                mass_GeV=p.quark_mass_GeV,total_pplus_GeV=p.total_pplus_GeV,
                gluon_helicity=hg,charge=charge,regulator=p.field_regulator,
                mean_total_transverse_GeV=mean_total_transverse_GeV,
                prescription=prescription,quadrature_order=quadrature_order)
        forward = raw(delta,incoming.cutoff,outgoing.cutoff)
        backward = raw(-delta,outgoing.cutoff,incoming.cutoff)
        for kind,(a,b) in enumerate(((forward.canonical,backward.canonical),(forward.regulator,backward.regulator))):
            for mu in range(4):
                values[kind][mu,of[sector]:of[sector+1],:oi[1]] = (
                    outgoing.bases[sector].getH()@a[mu]@incoming.bases[0]).toarray()
                values[kind][mu,:of[1],oi[sector]:oi[sector+1]] = (
                    outgoing.bases[0].getH()@b[mu].getH()@incoming.bases[sector]).toarray()
    return ClusterCurrentIntegral(*values,prescription)
