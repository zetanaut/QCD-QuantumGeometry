"""Exact two-energy transverse-current matching for the explicit free-Q5 model.

This advances the canonical current but does NOT silently add it to the
interacting QQ/Q6 model. The latter also needs Q5<->Q6 current contacts,
backflow endpoints, gauged-regulator and current/loop counterterm partners.
"""
import numpy as np

from .m2_shell_fock_reference import ShellFockReference
from .m2_transverse_current_projection import free_transverse_current_between
from .m2_transverse_contact_current import transverse_contact_between
from .m2_excluded_transverse_contact import build_excluded_transverse_contact
from .m2_shell_transverse_current import shell_transverse_current_between
from .m2_matched_h2_reduction import _inverse_sqrt


class FreeQ5TransverseMatching:
    """Omega_f† J_parent Omega_i, retaining PP/PQ/QP/QQ and both energies.

    The supplied finite parent has free Q5 and g V1+g² V2 sources.
    Its bare transverse current has J_PP[0]+g J_PP[1], g J_QP[1]
    and J_QQ[0]. There is no local transverse J[2] from the eliminated
    quark field. Missing physical current counterterms are NOT asserted
    zero in nature. No state, physical current fit or calibration is chosen.
    """
    physical=False
    complete_physical_four_current=False
    interacting_QQ_Q6_current=False

    def __init__(self,outgoing,incoming,transfer,*,total_pplus_GeV,charge):
        if not isinstance(outgoing,ShellFockReference) or not isinstance(incoming,ShellFockReference):
            raise ValueError("explicit complete-shell FREE-Q5 references required")
        pf,pi=outgoing.pilot,incoming.pilot
        for name in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV",
                     "gluon_mass_GeV","coupling"):
            if getattr(pf,name)!=getattr(pi,name):
                raise ValueError("current reference endpoints disagree on "+name)
        self.outgoing,self.incoming=outgoing,incoming
        self.transfer=tuple(float(v) for v in transfer)
        self.total_pplus_GeV=float(total_pplus_GeV);self.charge=charge
        args=dict(total_pplus_GeV=self.total_pplus_GeV,charge=charge)
        self.retained_free=free_transverse_current_between(pf,pi,self.transfer,**args)
        self.retained_contact=transverse_contact_between(pf,pi,self.transfer,**args)
        incoming_by_name={s.source.space.sector:s for s in incoming.sectors}
        if {s.source.space.sector for s in outgoing.sectors}!=incoming_by_name.keys():
            raise ValueError("identical declared excluded current sectors required")
        self.sectors=[]
        for sf in outgoing.sectors:
            si=incoming_by_name[sf.source.space.sector]
            f,i=sf.source.space,si.source.space
            for name in ("total_k2","cutoff","target"):
                if getattr(f,name)!=getattr(i,name):
                    raise ValueError("excluded current endpoints disagree on "+name)
            forward=build_excluded_transverse_contact(pi,f,self.transfer,**args)
            backward=build_excluded_transverse_contact(pf,i,tuple(-v for v in self.transfer),**args)
            self.sectors.append((sf,si,forward,backward))

    def coefficients(self,energy_out,energy_in):
        """Coefficients g^0 through g^4 of this supplied finite-parent current."""
        result={n:np.zeros_like(self.retained_free) for n in range(5)}
        result[0]=self.retained_free.copy()
        result[1]=self.retained_contact.copy()
        out_waves={s.source.space.sector:w for s,w in
                   zip(self.outgoing.sectors,self.outgoing.waves(energy_out))}
        in_waves={s.source.space.sector:w for s,w in
                  zip(self.incoming.sectors,self.incoming.waves(energy_in))}
        for sf,si,forward,backward in self.sectors:
            f,i=sf.source.space,si.source.space
            wf,wi=out_waves[f.sector],in_waves[i.sector]
            for j in (0,1):
                # Both off-diagonal endpoints. Do not combine them as twice
                # one term: energies and spin fibers can differ.
                for left,cl,_ in sf.source.components:
                    result[left+1][j][np.ix_(cl,forward.retained_columns)]+=(
                        wf[left].conj().T@forward.current_per_gs[j])
                for right,cr,_ in si.source.components:
                    result[right+1][j][np.ix_(backward.retained_columns,cr)]+=(
                        backward.current_per_gs[j].conj().T@wi[right])
                    current_wave=shell_transverse_current_between(f,i,wi[right],self.transfer,
                        component=j,b_GeV=self.incoming.pilot.width_GeV,
                        total_pplus_GeV=self.total_pplus_GeV,charge=self.charge)
                    for left,cl,_ in sf.source.components:
                        result[left+right][j][np.ix_(cl,cr)]+=wf[left].conj().T@current_wave
        return result

    def effective_current(self,energy_out,energy_in):
        """Unwhitened effective operator; retained states use the norm metric."""
        g=self.incoming.pilot.coupling
        return sum(g**n*value for n,value in self.coefficients(energy_out,energy_in).items())

    def normalized_current(self,energy_out,energy_in):
        """Two endpoint N^-1/2 factors for Euclidean quantum-register states."""
        left=_inverse_sqrt(self.outgoing.norm_metric(energy_out))
        right=_inverse_sqrt(self.incoming.norm_metric(energy_in))
        return left@self.effective_current(energy_out,energy_in)@right
