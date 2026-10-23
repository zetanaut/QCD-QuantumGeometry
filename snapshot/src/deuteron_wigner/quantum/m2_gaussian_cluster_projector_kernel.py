"""Exact Gaussian Hamiltonian/projector product before outer-shell truncation.

The finite-rank [Pi,T] is sandwiched between continuum displacements and
the full shifted Hamiltonian integral. No finite reported outer shell
stands in for Gaussian interaction tails. This derives the transverse
straight-path tangent only, not a minus current or physical matching.
"""
from dataclasses import dataclass
from operator import index
import numpy as np
from .m2_gaussian_field_regulator import gaussian_shell_integral,_array
from .m2_coupled_longitudinal_jacobi import oscillator_shell
from .m2_galerkin_projector_current import _rectangular_displacement
from .m2_gaussian_cluster_interactions import _pair_data


@dataclass(frozen=True,kw_only=True)
class GaussianHamiltonianOrbital:
    """Actual unshifted Hamiltonian integration data, with independent HO shifts.

    Polynomial and physical-field momenta remain those of H. Only the bra/ket
    wavefunctions are displaced in D H or H D; shifting fields together
    would instead translate the entire operator and lose the boundary.
    """
    out_map: np.ndarray
    in_map: np.ndarray
    field_maps: np.ndarray
    field_shifts: np.ndarray
    b_GeV: float
    field_precision: float
    measure: float
    constant: complex=1.
    linear: object=None
    quadratic: object=None
    reference_shape: object=None

    def integrate(self,out_cutoff,in_cutoff,*,out_shift=None,in_shift=None):
        integral=gaussian_shell_integral
        if self.reference_shape is not None:
            from .m2_gaussian_reference_shape import GaussianReferenceShape
            if not isinstance(self.reference_shape,GaussianReferenceShape):
                raise ValueError("explicit finite-reference shape required")
            integral=self.reference_shape.integrate
        return self.measure*integral(out_cutoff=out_cutoff,in_cutoff=in_cutoff,
            out_map=self.out_map,in_map=self.in_map,out_shift=out_shift,in_shift=in_shift,
            b_GeV=self.b_GeV,field_precision=self.field_precision,field_maps=self.field_maps,
            field_shifts=self.field_shifts,constant=self.constant,linear=self.linear,quadratic=self.quadratic)


def pair_hamiltonian_orbital(incoming,outgoing,pair,*,species,out_species,
                             b_GeV,regulator,total_transverse_GeV):
    """Original four-physical-field pair Hamiltonian, independent of current."""
    reg,fi,fo,si,so,p,b,_,_,eligible,x,jac,ri,ro=_pair_data(
        incoming,outgoing,pair,b_GeV,0,0,regulator,species,out_species)
    pt=_array(total_transverse_GeV,(2,),"Hamiltonian total transverse momentum")
    d=len(incoming)-1
    ao=np.column_stack((np.eye(d),np.zeros(d)));ai=ao.copy();ai[0]=0.;ai[0,-1]=1.
    lo,li=ro.T@ao,ri.T@ai
    physical=np.vstack((fo.inverse[list(p),:-1]@lo,fi.inverse[list(p),:-1]@li))
    modes=tuple(outgoing[j] for j in p)+tuple(incoming[j] for j in p)
    transform=reg.leg_transform(modes)
    fractions=np.r_[fo.fractions[list(p)],fi.fractions[list(p)]]
    return GaussianHamiltonianOrbital(out_map=lo,in_map=li,
        field_maps=transform@physical,field_shifts=transform@(fractions[:,None]*pt),
        b_GeV=b,field_precision=1/reg.width_GeV**2,measure=jac,constant=float(eligible),
        reference_shape=reg.reference_shape)


def _cutoff(value):
    if isinstance(value,(bool,np.bool_)) or index(value)<0:
        raise ValueError("explicit nonnegative integer oscillator cutoff required")
    return index(value)


def _commutator(d,cutoff,weights,b):
    """Independent ladder construction of [Pi,T_j] through exactly N+1."""
    inside=oscillator_shell(d,cutoff);boundary=oscillator_shell(d,cutoff+1)
    lookup={x:i for i,x in enumerate(boundary)}
    n=len(inside);result=np.zeros((2,len(boundary),len(boundary)),complex)
    derivative=np.array([np.r_[weights,weights],np.r_[-1j*weights,1j*weights]])/(2*b)
    for col,state in enumerate(boundary):
        for mode in range(2*d):
            for step in (-1,1):
                out=list(state);out[mode]+=step
                if out[mode]<0 or sum(out)>cutoff+1:continue
                row=lookup[tuple(out)]
                difference=int(row<n)-int(col<n)
                amplitude=(derivative[:,mode] if step==1 else -derivative[:,mode].conj())
                result[:,row,col]+=difference*np.sqrt(state[mode]+(step==1))*amplitude
    return result


def exact_transverse_projector_boundary(hamiltonian,*,out_cutoff,in_cutoff,
        out_weights,in_weights,out_charges,in_charges,transfer_GeV,total_pplus_GeV,
        quadrature_order):
    """Pi_f K_f H Pi_i + Pi_f H K_i Pi_i, divided by 2P+.

    K = Pi C (1-Pi)+(1-Pi) C Pi, C = integral D(1-t)[Pi,T]D(t).
    D(t)H and HD(1-t) are exact Gaussian integrals, not truncated products.
    Returned array is (x/y,bra,ket); there is deliberately no minus slot.
    """
    if not isinstance(hamiltonian,GaussianHamiltonianOrbital):
        raise ValueError("explicit original Gaussian Hamiltonian orbital required")
    co,ci=_cutoff(out_cutoff),_cutoff(in_cutoff)
    delta=_array(transfer_GeV,(2,),"transverse transfer")
    pp=float(total_pplus_GeV);b=float(hamiltonian.b_GeV)
    if not np.all(np.isfinite([pp,b])) or min(pp,b)<=0:
        raise ValueError("positive finite P+ and oscillator width required")
    if isinstance(quadrature_order,(bool,np.bool_)) or index(quadrature_order)<2:
        raise ValueError("explicit integer path quadrature order at least two required")
    df,di=hamiltonian.out_map.shape[0],hamiltonian.in_map.shape[0]
    groups=[]
    for w,e,d in ((out_weights,out_charges,df),(in_weights,in_charges,di)):
        e=np.asarray(e,float);w=np.asarray(w,float)
        if e.ndim!=1 or not len(e) or w.shape!=(len(e),d) or not np.all(np.isfinite(np.r_[e,w.ravel()])):
            raise ValueError("one finite Jacobi recoil row per physical charge required")
        groups.append((w,e))
    h=hamiltonian.integrate(co,ci)
    result=np.zeros((2,*h.shape),complex)
    nodes,weights=np.polynomial.legendre.leggauss(index(quadrature_order))
    ts,tws=(nodes+1)/2,weights/2
    for side,(recoil,charges) in enumerate(groups):
        d,N=(df,co) if side==0 else (di,ci)
        for w,e in zip(recoil,charges):
            if e==0:continue  # exact neutral physical constituent
            comm=_commutator(d,N,w,b)
            for t,weight in zip(ts,tws):
                if side==0:
                    left=_rectangular_displacement(d,N,N+1,w[:,None]*delta*(1-t),b)
                    right=_rectangular_displacement(d,N+1,N,w[:,None]*delta*t,b)
                    shifted=hamiltonian.integrate(co+1,ci,out_shift=w[:,None]*delta*t)
                    # Q_f removes the Pi_f part only after the exact D H integral.
                    complement=shifted-right@h
                    for j in range(2):result[j]+=e*weight*(left@comm[j]@complement)
                else:
                    left=_rectangular_displacement(d,N,N+1,w[:,None]*delta*(1-t),b)
                    right=_rectangular_displacement(d,N+1,N,w[:,None]*delta*t,b)
                    shifted=hamiltonian.integrate(co,ci+1,in_shift=-w[:,None]*delta*(1-t))
                    complement=shifted-h@left
                    for j in range(2):result[j]+=e*weight*(complement@comm[j]@right)
    return result/(2*pp)


def cubic_hamiltonian_orbital(route,helicities,*,b_GeV,mass_GeV,regulator,
        total_pplus_GeV,total_transverse_GeV):
    """Original color/Fock-stripped cubic H, including its affine momentum.

    Unit sampling extracts an exactly affine polynomial, not a finite
    difference current. Inherited Q5/Q6 routes use their actual physical legs.
    """
    from .m2_gaussian_cluster_current import cubic_current_geometry,cubic_physical_kernel
    from .m2_gaussian_cluster_interactions import _regulator
    reg=_regulator(regulator)
    g=cubic_current_geometry(route,transfer_GeV=(0.,0.),
                             mean_total_transverse_GeV=total_transverse_GeV)
    kinds={"q_to_qg":("q","g","q"),"anti_to_antig":("qbar","g","qbar"),
           "g_to_pair":("q","qbar","g"),"g_to_gg":("g","g","g")}[g["channel"]]
    enabled=all(reg.contains(s,k) for s,k in zip(kinds,g["modes"]))
    args=dict(channel=g["channel"],helicities=helicities,total_k2=g["total_k2"],
              total_pplus_GeV=total_pplus_GeV,mass_GeV=mass_GeV)
    maps,shifts=g["physical_field_maps"],g["physical_field_shifts"]
    constant=cubic_physical_kernel(g["modes"],shifts,**args)
    linear=np.zeros((2,maps.shape[1]),complex)
    for j in range(2):
        for k in range(maps.shape[1]):
            unit=shifts.copy();unit[:,j]+=maps[:,k]
            linear[j,k]=cubic_physical_kernel(g["modes"],unit,**args)-constant
    transform=reg.leg_transform(g["modes"])
    return GaussianHamiltonianOrbital(out_map=g["out_map"],in_map=g["in_map"],
        field_maps=transform@maps,field_shifts=transform@shifts,b_GeV=b_GeV,
        field_precision=1/reg.width_GeV**2,measure=g["jacobian"],
        constant=constant*enabled,linear=linear*enabled,reference_shape=reg.reference_shape)


def quartic_source_hamiltonian_orbital(route,*,b_GeV,regulator,total_transverse_GeV):
    """Actual 1-to-3 source orbital; ordered color/spin/Fock factors external."""
    from .m2_gaussian_cluster_quartic_source_current import quartic_source_geometry
    from .m2_gaussian_cluster_interactions import _regulator
    reg=_regulator(regulator)
    g=quartic_source_geometry(route,transfer_GeV=(0.,0.),
                              mean_total_transverse_GeV=total_transverse_GeV)
    enabled=all(reg.contains(s,k) for s,k in zip(g["species"],g["modes"]))
    transform=reg.leg_transform(g["modes"])
    return GaussianHamiltonianOrbital(out_map=g["out_map"],in_map=g["in_map"],
        field_maps=transform@g["physical_field_maps"],field_shifts=transform@g["physical_field_shifts"],
        b_GeV=b_GeV,field_precision=1/reg.width_GeV**2,measure=g["jacobian"],constant=float(enabled),
        reference_shape=reg.reference_shape)
