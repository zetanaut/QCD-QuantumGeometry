"""Necessary binding/cluster controls, not a chosen nucleon Hamiltonian.

Six distinct static quark slots retain the full 3**6 color tensor product
and its five-dimensional global-singlet subspace. There is no dynamical
gauge-link Hilbert space, light-quark orbital solve or physical fit here.
"""
from functools import lru_cache
from itertools import combinations, permutations
from math import sqrt

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from .canonical_lf_network import four_vector, dot
from .modular_gauge_network import generators


def free_invariant_mass(masses, plus, transverse):
    """Independent Cartesian sum of the preserved symmetric-LF four-vectors."""
    m=np.asarray(masses,float);p=np.asarray(plus,float);k=np.asarray(transverse,float)
    if (m.ndim!=1 or len(m)==0 or p.shape!=m.shape or k.shape!=(len(m),2)
            or not np.isfinite(m).all() or np.any(m<0)):
        raise ValueError('Finite nonnegative masses with positive-plus momenta required.')
    total=sum((four_vector([pp,*kk],mm) for mm,pp,kk in zip(m,p,k)),np.zeros(4))
    return sqrt(max(0.,float(dot(total,total))))


def longitudinal_threshold(masses,k2):
    """Three unconfined free quarks, odd positive longitudinal mode integers.

    Returns a finite-grid kinetic minimum, NOT a physical dissociation edge
    in confining QCD. k2=sum(n_i), x_i=n_i/k2; all n_i odd.
    """
    m=np.asarray(masses,float)
    if (m.shape!=(3,) or np.any(m<=0) or not np.isfinite(m).all()
            or isinstance(k2,bool) or int(k2)!=k2 or k2<3 or k2%2!=1):
        raise ValueError('Three positive masses and odd integer k2>=3 required.')
    k2=int(k2)
    routes=[(a,b,k2-a-b) for a in range(1,k2,2) for b in range(1,k2-a,2)
            if k2-a-b>0]
    values=[np.sum(m*m/(np.array(n)/k2)) for n in routes]
    index=int(np.argmin(values))
    return sqrt(float(values[index])),routes[index]


def _sign(values):
    return (-1)**sum(values[i]>values[j] for i in range(len(values)) for j in range(i+1,len(values)))


@lru_cache(maxsize=1)
def singlet_carrier():
    """Ten epsilon-epsilon partitions, their actual Gram matrix and rank-5 map.

    Partitions are recouplings in ONE color space, not ten flux states.
    The same tensor coefficients embed isometrically in six occupied distinct
    fermion orbitals ordered by slot; no light-quark Pauli/orbital solve follows.
    """
    parts=[];columns=[]
    for pair in combinations(range(1,6),2):
        a=(0,*pair);b=tuple(i for i in range(6) if i not in a)
        v=np.zeros((3,)*6)
        for ca in permutations(range(3)):
            for cb in permutations(range(3)):
                indices=[0]*6
                for i,c in zip(a,ca): indices[i]=c
                for i,c in zip(b,cb): indices[i]=c
                v[tuple(indices)]=_sign(ca)*_sign(cb)/6
        parts.append((a,b));columns.append(v.ravel())
    e=np.array(columns).T
    u,s,_=np.linalg.svd(e,full_matrices=False)
    if not (np.max(abs(s[:5]-sqrt(2)))<1e-13 and np.max(s[5:])<1e-13):
        raise RuntimeError('Unexpected six-quark singlet rank.')
    # Fixing this exact rank is algebraic, not a near-small physical truncation.
    gram=e.T@e;basis=u[:,:5]
    for value in (e,gram,basis): value.setflags(write=False)
    return tuple(parts),e,gram,basis


def pair_action(values,i,j):
    """Fierz identity t_i.t_j = (P_ij-I/3)/2 on raw color tensors."""
    if not (0<=i<j<6): raise ValueError('Ordered distinct slots required.')
    v=np.asarray(values);shape=v.shape
    if shape[0]!=729: raise ValueError('Full six-quark color carrier required.')
    tensor=v.reshape((3,)*6+shape[1:])
    return (tensor.swapaxes(i,j).reshape(shape)-v/3)/2


def local_color_action(values,slot,generator):
    v=np.asarray(values);shape=v.shape
    if shape[0]!=729 or not 0<=slot<6: raise ValueError('Six color slots required.')
    tensor=v.reshape((3,)*6+shape[1:])
    return np.moveaxis(np.tensordot(generator,tensor,axes=(1,slot)),0,slot).reshape(shape)


def generator_pair_action(values,i,j):
    """Independent explicit Gell-Mann action, without Fierz/permutation reuse."""
    return sum((local_color_action(local_color_action(values,j,t),i,t)
                for t in generators()),np.zeros_like(values,dtype=complex))


def geometry(separation,side=1.):
    """Two equilateral static triangles in parallel transverse planes."""
    if not np.isfinite(separation) or not np.isfinite(side) or separation<=0 or side<=0:
        raise ValueError('Positive separation and side required.')
    a=np.array([[0,0,0],[0,side,0],[0,side/2,sqrt(3)*side/2]],float)
    a-=a.mean(axis=0)
    return np.concatenate((a,a+[separation,0,0]))


def pair_potential(points,*,coefficient=1.,derivative=None):
    """Illustrative static V=-3/2*kappa*sum(t_i.t_j*r_ij) in all five singlets.

    Units: kappa in energy/length. derivative supplies dr_slot/dR for the
    actual geometry derivative; it is NOT an electromagnetic four-current.
    """
    r=np.asarray(points,float)
    if r.shape!=(6,3) or not np.isfinite(r).all() or not np.isfinite(coefficient):
        raise ValueError('Six finite positions and finite coefficient required.')
    velocity=None if derivative is None else np.asarray(derivative,float)
    if velocity is not None and (velocity.shape!=r.shape or not np.isfinite(velocity).all()):
        raise ValueError('Six finite geometry velocities required.')
    _,_,_,u=singlet_carrier();h=np.zeros((5,5))
    for i,j in combinations(range(6),2):
        delta=r[i]-r[j];distance=np.linalg.norm(delta)
        if distance<=0: raise ValueError('Distinct static positions required.')
        weight=distance if velocity is None else delta@(velocity[i]-velocity[j])/distance
        h+=-1.5*coefficient*weight*(u.T@pair_action(u,i,j))
    return h


def partition_energies(points,*,coefficient=1.):
    """Tempting two-baryon pair-length labels, not eigenvalues of a common H."""
    parts,*_=singlet_carrier();r=np.asarray(points,float)
    return np.array([coefficient*sum(np.linalg.norm(r[i]-r[j])
                    for triple in part for i,j in combinations(triple,2)) for part in parts])


def connected_tree_length(points):
    r=np.asarray(points,float);d=np.linalg.norm(r[:,None,:]-r[None,:,:],axis=-1)
    return float(minimum_spanning_tree(d).sum())


def cluster_case(separation):
    """Dimensionless r/side and V/(kappa*side), kappa=side=1 reference."""
    parts,e,g,u=singlet_carrier();r=geometry(separation);h=pair_potential(r)
    w,v=np.linalg.eigh(h);ground=v[:,0]
    physical=parts.index(((0,1,2),(3,4,5)));singlet=u.T@e[:,physical]
    velocity=np.zeros_like(r);velocity[3:,0]=1
    dh=pair_potential(r,derivative=velocity)
    labels=partition_energies(r)
    return dict(separation=separation,pair_ground_energy=float(w[0]),
        pair_ground_shift_from_two_singlets=float(w[0]-6),
        product_singlet_diagonal=float(singlet@h@singlet),
        hidden_color_weight=float(1-abs(singlet@ground)**2),
        force=float(-ground@dh@ground),
        connected_tree_excess_length=connected_tree_length(r)-4,
        independent_partition_energy_Hermiticity_defect=float(np.linalg.norm(g*labels[None,:]-labels[:,None]*g)),
        excited_color_gaps=(w[1:]-w[0]).tolist())
