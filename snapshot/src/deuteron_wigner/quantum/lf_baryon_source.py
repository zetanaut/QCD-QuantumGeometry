"""Complete one-emission/one-q+=0-source jets on a quark exterior carrier.

No production bound-state Hamiltonian or current is selected. Source-free,
source-first, emission-first and mixed sectors are kept at their actual
perturbative order, with all required intermediate modes. Final-window
projection is not a global Fock cutoff. A real source needs both +/-q modes.
"""
from itertools import product
from math import pi, sqrt

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from .canonical_lf_fields import GAMMA, GAMMA_PLUS, polarization, polarization_cartesian, slash
from .canonical_lf_network import momentum
from .finite_lf_coupling import spinors
from .lf_baryon_emission import EmissionLayer, emission_leaf, emission_contact, occupied, mode_key
from .lf_pair_electromagnetic import (
    ALPHA, BETA, CHARGE, PHI_CHARGE, b0, good_spinors, cart, parent_h,
    phi_source_coefficients, fermion_source, full_spinor_source, HARMONICS,
    time_integral, partial_pulse,
)
from .modular_gauge_network import generators
from .transverse_source_matching import transverse_vertex


def gluon_contact(a,k,p,mass,epsilon,volume,*,covariant=False):
    """Independent canonical mixed g_s/e contact, (q12,gluon16,q12).

    B_g=+alphaT.t.AgT for the preserved D_color=partial-i g_s t.Ag;
    B_EM=-alphaT.Q.AEM_T. g_s,e stripped, quantum gluon normalized.
    The full -slash(eps_g) vertex checks the instantaneous expression.
    """
    a,k,p=map(momentum,(a,k,p));e=np.asarray(epsilon,complex);q=a+k-p
    if (abs(q[0])>1e-12 or np.linalg.norm(q[1:])<1e-12 or e.shape!=(4,)
            or not np.isfinite(e).all() or not np.isfinite(volume) or volume<=0):
        raise ValueError('Finite q+=0, nonzero qT source and positive volume required.')
    u,v=(spinors(a,mass),spinors(p,mass)) if covariant else (good_spinors(a,mass),good_spinors(p,mass))
    photon=CHARGE@np.kron(slash(cart(e)),np.eye(6))
    gp=np.kron(GAMMA_PLUS,np.eye(6));em=-sum(e[j+2]*ALPHA[j]@CHARGE for j in range(2))
    result=np.zeros((12,16,12),complex)
    for ih,h in enumerate((-1,1)):
        eps=polarization(*k,h).conj()
        for c,t in enumerate(generators()):
            color=np.kron(np.eye(8),t)
            if covariant:
                vg=-np.kron(slash(polarization_cartesian(eps)),np.kron(np.eye(2),t))
                value=u.conj().T@BETA@(photon@gp@vg/(2*a[0])+vg@gp@photon/(2*p[0]))@v
                value/=sqrt(8*a[0]*k[0]*p[0]*volume)
            else:
                gl=sum(eps[j+2]*ALPHA[j]@color for j in range(2))
                kernel=(em@gl/a[0]+gl@em/p[0]
                    +e[0]*(b0(a,mass)@CHARGE@gl/a[0]**2+gl@CHARGE@b0(p,mass)/p[0]**2))/2
                value=u.conj().T@kernel@v/sqrt(2*k[0]*volume)
            result[:,8*ih+c,:]=value
    return result


def covariant_phi_source(m,c,pout,pin,epsilon,omega):
    """Independent off-shell Lorentz vertex -> canonical J and endpoint W.

    Gamma/(2k+)=J+D_out W+W D_in. Differentiate the actual linear energy
    polynomial with a symmetric exact difference; no current Ward fit.
    """
    a,b=map(momentum,(pout,pin));e=cart(np.asarray(epsilon));c=np.asarray(c)
    if abs(a[0]-b[0])>1e-12:
        raise ValueError('Equal positive plus required.')
    metric=np.array([1,-1,-1,-1])
    def vertex(w):
        incoming=cart([b[0],w,*b[1:]]);outgoing=cart([a[0],w+omega,*a[1:]])
        v0=np.sum((outgoing+incoming)*metric*e)
        vc=np.sum(transverse_vertex(outgoing,incoming,1.)*metric*e)
        return np.kron(v0*np.eye(2)+vc*c,PHI_CHARGE)/(2*b[0])
    raw=vertex(0);w=(vertex(1)-vertex(-1))/4
    j=raw+parent_h(m,a)@w+w@parent_h(m,b)-omega*w
    return j,w


def _move(mask,i,j,*,reference=False):
    """Canonical b_i^dagger b_j; reference uses ordered-list positions."""
    i,j=int(i),int(j)
    if reference:
        order=list(occupied(mask))
        if j not in order:
            return None
        sign=(-1)**order.index(j);order.remove(j)
        if i in order:
            return None
        sign*=(-1)**sum(x<i for x in order);order.append(i)
        return sum(1<<x for x in order),sign
    if not mask & (1<<j): return None
    sign=(-1)**(mask & ((1<<j)-1)).bit_count();out=mask^(1<<j)
    if out & (1<<i): return None
    sign*=(-1)**(out & ((1<<i)-1)).bit_count()
    return out|(1<<i),sign


def _size(key): return 8 if key[0]=='phi' else 16


def source_harmonics(duration,transfer,epsilon,*,pure_gauge=False):
    """Exactly one sin^4 Fourier coefficient per source mode.

    Kept separate so the source itself is tested against direct trigonometric
    values/derivatives, independently of two solvers sharing its input.
    """
    q=np.asarray(transfer,float);e=np.asarray(epsilon,complex)
    if (not np.isfinite(duration) or duration<=0 or q.shape!=(3,) or e.shape!=(4,)
            or not np.isfinite(q).all() or not np.isfinite(e).all()
            or (pure_gauge and np.any(e[1:]!=0))):
        raise ValueError('Finite source/transfer and positive duration required.')
    omega=np.array([-2*pi*k/duration for k,_ in HARMONICS])
    modes=np.array([coefficient*(-1j*np.array([0,frequency,*q[1:]])*e[0]
                                if pure_gauge else e)
                    for frequency,(_,coefficient) in zip(omega,HARMONICS)])
    return omega,modes


class BaryonSource:
    """Sparse exact-order graph for an explicitly supplied initial quark state.

    seed_builder receives the full momentum register; no seed is chosen by
    default. Both species/spin benchmarks and arbitrary correlated exterior
    inputs are supported. Spectator sources are included at both time orders.
    """
    def __init__(self,cell,mass_squared,c,couplings,gs,mass,incoming,transfer,shell,
                 duration,epsilon,seed_builder,*,pure_gauge=False,representation='good'):
        self.q=np.asarray(transfer,float);self.m=np.asarray(mass_squared);self.c=np.asarray(c)
        self.T=float(duration);self.shell=int(shell);self.cell=cell
        self.reference=representation=='dirac'
        if (self.q.shape!=(3,) or not np.isfinite(self.q).all() or abs(self.q[0])>1e-12
                or np.linalg.norm(self.q[1:])<1e-12 or self.T<=0 or not np.isfinite(self.T)
                or isinstance(shell,bool) or self.shell!=shell or shell<0
                or representation not in ('good','dirac')):
            raise ValueError('Finite nonzero qT, q+=0, positive time and integer shell required.')
        # Transfer is a periodic momentum difference, not a positive-plus quantum.
        if not np.allclose(self.q/cell.spacing,np.rint(self.q/cell.spacing),atol=1e-12,rtol=0):
            raise ValueError('Source transfer must be on the periodic cell grid.')
        original=tuple(mode_key(cell.mode(p,'fermion')) for p in incoming)
        all_in=tuple(sorted(set(original)|{mode_key(np.array(p)+self.q) for p in original}))
        extended=shell+int(np.max(abs(np.rint(self.q[1:]/cell.spacing[1:]))))
        self.layer=EmissionLayer(cell,all_in,mass,couplings,gs,extended)
        self.seed=dict(seed_builder(self.layer));self.layer._validate_state(self.seed)
        if not self.seed or any(self.layer.momenta[i//12] not in original for mask in self.seed for i in occupied(mask)):
            raise ValueError('Seed must occupy only the declared original incoming momenta.')
        counts={len(occupied(mask)) for mask in self.seed}
        if len(counts)!=1 or next(iter(counts))<1:
            raise ValueError('One fixed positive quark number required.')
        self.nquarks=next(iter(counts));self.original=original
        # Validate independent source/mass matrices even for a source that vanishes.
        parent_h(self.m,[cell.spacing[0],0,0])
        self.omega=np.array([-2*np.pi*k/self.T for k,_ in HARMONICS])
        eps=np.asarray(epsilon,complex)
        if eps.shape!=(4,) or not np.isfinite(eps).all() or (pure_gauge and np.any(eps[1:]!=0)):
            raise ValueError('Finite four-source; pure gauge theta amplitude only in entry zero.')
        self.omega,self.fourier=source_harmonics(self.T,self.q,eps,pure_gauge=pure_gauge)
        # Covariant/constraint emission coefficients use distinct formulas.
        self.emissions={}
        for ai,k,pi,kind,value in self.layer.routes:
            if kind=='phi':
                leaf=emission_leaf(self.layer.momenta[ai],k,self.layer.momenta[pi],mass,constraint=not self.reference)
                value=np.einsum('iAj,n->inAj',leaf,self.layer.g).reshape(12,8,12)/sqrt(cell.volume)
            self.emissions.setdefault(pi,[]).append((ai,k,kind,value))
        self.current={};self.contact_cache={};self.phi_cache={}
        self.i_keys=tuple(sorted(self.seed));self.i_index={k:i for i,k in enumerate(self.i_keys)}
        self.initial=np.array([self.seed[k] for k in self.i_keys])
        self.ei=np.array([self.layer.energy(k) for k in self.i_keys])
        # Store sparse entries by physical block labels until all reached states exist.
        va=[];vaspec=[];vy=[];jbefore=[[] for _ in self.omega];jq=[[] for _ in self.omega]
        jqspec=[[] for _ in self.omega]
        jc=[[] for _ in self.omega];kw=[[] for _ in self.omega];kf=[[] for _ in self.omega]
        a_keys=set();y_keys=set();f_keys=set()
        for col,mask in enumerate(self.i_keys):
            for out,values,_ in self._quark_source(mask):
                a_keys.add(out)
                for f,value in enumerate(values):
                    if value!=0: jbefore[f].append((out,col,value))
            for key,component,value,_ in self._emit(mask):
                y_keys.add(key);vy.append(((key,component),col,value))
        self.a_keys=tuple(sorted(a_keys));self.a_index={k:i for i,k in enumerate(self.a_keys)}
        self.ea=np.array([self.layer.energy(k) for k in self.a_keys])
        for col,mask in enumerate(self.a_keys):
            for key,component,value,emitter in self._emit(mask):
                if self._final(key):
                    f_keys.add(key);va.append(((key,component),col,value))
                    if len(self.original)==1 and self.layer.momenta[emitter] in self.original:
                        vaspec.append(((key,component),col,value))
        for key in sorted(y_keys):
            kind,k,mask=key;size=_size(key)
            for out,values,source in self._quark_source(mask):
                final=(kind,k,out)
                if not self._final(final): continue
                f_keys.add(final)
                for f,value in enumerate(values):
                    if value!=0:
                        for component in range(size):
                            entry=((final,component),(key,component),value);jq[f].append(entry)
                            if len(self.original)==1 and self.layer.momenta[source] in self.original:
                                jqspec[f].append(entry)
            if kind=='phi':
                final=(kind,mode_key(np.array(k)+self.q),mask)
                if self._final(final):
                    f_keys.add(final);curr,norm=self._collective(k)
                    for f in range(len(self.omega)):
                        for i,j in zip(*np.nonzero(curr[f])): jc[f].append(((final,int(i)),(key,int(j)),curr[f,i,j]))
                        for i,j in zip(*np.nonzero(norm[f])): kw[f].append(((final,int(i)),(key,int(j)),norm[f,i,j]))
        for col,mask in enumerate(self.i_keys):
            for key,component,values in self._contacts(mask):
                if not self._final(key): continue
                f_keys.add(key)
                for f,value in enumerate(values):
                    if value!=0: kf[f].append(((key,component),col,value))
        self.y_keys=tuple(sorted(y_keys));self.f_keys=tuple(sorted(f_keys))
        if not self.f_keys: raise ValueError('No final source/emission routes in the declared window.')
        self.y_index,self.ey,self.uy,self.hy=self._blocks(self.y_keys)
        self.f_index,self.ef,self.uf,self.hf=self._blocks(self.f_keys)
        ni,na,ny,nf=len(self.i_keys),len(self.a_keys),len(self.ey),len(self.ef)
        scalar=lambda entries,rows,cols,shape:self._matrix(entries,rows,cols,shape)
        self.vy=scalar(vy,self.y_index,None,(ny,ni))
        self.va=scalar(va,self.f_index,None,(nf,na))
        self.vaspec=scalar(vaspec,self.f_index,None,(nf,na))
        self.ja=[scalar(x,self.a_index,None,(na,ni)) for x in jbefore]
        self.jq=[scalar(x,self.f_index,self.y_index,(nf,ny)) for x in jq]
        self.jqspec=[scalar(x,self.f_index,self.y_index,(nf,ny)) for x in jqspec]
        self.jc=[scalar(x,self.f_index,self.y_index,(nf,ny)) for x in jc]
        self.kf=[scalar(x,self.f_index,None,(nf,ni)) for x in kf]
        self.kw=[scalar(x,self.f_index,self.y_index,(nf,ny))@self.vy for x in kw]
        self.jtotal=[q+c for q,c in zip(self.jq,self.jc)]
        self.ktotal=[f+w for f,w in zip(self.kf,self.kw)]
        # Exact block eigenbasis is used only by spectral integration, not ODE.
        self.svy=self.uy.getH()@self.vy;self.sva=self.uf.getH()@self.va
        self.svaspec=self.uf.getH()@self.vaspec
        self.sjq=[self.uf.getH()@x@self.uy for x in self.jq]
        self.sjqspec=[self.uf.getH()@x@self.uy for x in self.jqspec]
        self.sjc=[self.uf.getH()@x@self.uy for x in self.jc]
        self.skf=[self.uf.getH()@x for x in self.kf];self.skw=[self.uf.getH()@x for x in self.kw]

    def _inside(self,p):
        return np.max(abs(np.asarray(p)[1:]/self.cell.spacing[1:]))<=self.shell+1e-10

    def _final(self,key):
        return self._inside(key[1]) and all(self._inside(self.layer.momenta[i//12]) for i in occupied(key[2]))

    def _emit(self,mask):
        for jglobal in occupied(mask):
            pi,j=divmod(jglobal,12)
            for ai,k,kind,leaf in self.emissions.get(pi,[]):
                for i,c in zip(*np.nonzero(leaf[:,:,j])):
                    moved=_move(mask,12*ai+int(i),jglobal,reference=self.reference)
                    if moved is not None:
                        out,sign=moved;yield (kind,k,out),int(c),sign*leaf[i,c,j],pi

    def _quark_source(self,mask):
        ff=full_spinor_source if self.reference else fermion_source
        for old in occupied(mask):
            pi,j=divmod(old,12);p=np.array(self.layer.momenta[pi]);out=mode_key(p+self.q)
            if out not in self.layer.lookup:
                # This route cannot enter the final window or a source-first
                # incoming mode. No intermediate inside a requested path is cut.
                continue
            ai=self.layer.lookup[out]
            if pi not in self.current:
                self.current[pi]=np.array([ff(out,p,self.layer.mass,e) for e in self.fourier])
            for i in range(12):
                values=self.current[pi][:,i,j]
                if not np.any(values): continue
                moved=_move(mask,12*ai+i,old,reference=self.reference)
                if moved is not None: yield moved[0],moved[1]*values,pi

    def _collective(self,k):
        if k not in self.phi_cache:
            out=np.array(k)+self.q;cur=[];norm=[]
            for om,e in zip(self.omega,self.fourier):
                if self.reference:
                    j,w=covariant_phi_source(self.m,self.c,out,k,e,om)
                else:
                    j0,j1,j2,w=phi_source_coefficients(self.m,self.c,out,k,e);j=j0+om*j1+om**2*j2
                cur.append(j);norm.append(w)
            self.phi_cache[k]=(np.array(cur),np.array(norm))
        return self.phi_cache[k]

    def _contacts(self,mask):
        for old in occupied(mask):
            pi,j=divmod(old,12);p=np.array(self.layer.momenta[pi])
            if pi not in self.contact_cache:
                events=[];n=int(round(p[0]/self.cell.spacing[0]-.5))
                for z,x,y in product(range(1,n+1),range(-self.shell,self.shell+1),range(-self.shell,self.shell+1)):
                    k=self.cell.spacing*np.array([z,x,y]);a=mode_key(p+self.q-k)
                    if not self._inside(a) or a not in self.layer.lookup: continue
                    phi=[];gl=[]
                    for e in self.fourier:
                        v=emission_contact(a,k,p,self.layer.mass,e,self.cell.volume,covariant=self.reference)
                        phi.append(np.einsum('iAj,n->inAj',v,self.layer.g).reshape(12,8,12))
                        gl.append(self.layer.gs*gluon_contact(a,k,p,self.layer.mass,e,self.cell.volume,covariant=self.reference))
                    events.extend([(self.layer.lookup[a],mode_key(k),'phi',np.array(phi)),
                                   (self.layer.lookup[a],mode_key(k),'g',np.array(gl))])
                self.contact_cache[pi]=events
            for ai,k,kind,leaf in self.contact_cache[pi]:
                for i,c in zip(*np.nonzero(np.any(leaf[:,:,:,j]!=0,axis=0))):
                    moved=_move(mask,12*ai+int(i),old,reference=self.reference)
                    if moved is not None: yield (kind,k,moved[0]),int(c),moved[1]*leaf[:,i,c,j]

    def _blocks(self,keys):
        index={};energies=[];us=[];hs=[];pos=0;cache={}
        for key in keys:
            kind,k,mask=key;n=_size(key)
            if (kind,k) not in cache:
                h=parent_h(self.m,k) if kind=='phi' else np.eye(16)*np.dot(k[1:],k[1:])/(2*k[0])
                w,u=np.linalg.eigh(h);cache[kind,k]=(h,w,u)
            h,w,u=cache[kind,k];eq=self.layer.energy(mask)
            hs.append(sparse.csr_matrix(h+eq*np.eye(n)));us.append(sparse.csr_matrix(u))
            energies.extend(w+eq)
            for c in range(n): index[key,c]=pos+c
            pos+=n
        return index,np.array(energies),sparse.block_diag(us,format='csr'),sparse.block_diag(hs,format='csr')

    @staticmethod
    def _matrix(entries,rows,cols,shape):
        rr=[];cc=[];vv=[]
        for r,c,v in entries:
            rr.append(rows[r]);cc.append(c if cols is None else cols[c]);vv.append(v)
        return sparse.coo_matrix((vv,(rr,cc)),shape=shape,dtype=complex).tocsr()

    def wave(self,vector):
        return {key:np.array([vector[self.f_index[key,c]] for c in range(_size(key))]) for key in self.f_keys}

    def response(self,*,nodes=64,omit=None):
        """Exact lower-order spectral jets, outer Gauss quadrature of full wave.

        Source amplitude is GeV, or dimensionless theta for pure gauge.
        No normalization/refit is applied to the mixed derivative coefficient.
        """
        if omit not in (None,'before','after_quark','collective','fermion_contact','collective_endpoint','outside_intermediates','spectators'):
            raise ValueError('Unknown omission control.')
        if omit=='spectators' and len(self.original)!=1:
            raise ValueError('Spectator omission diagnostic requires a single initial momentum.')
        x,w=np.polynomial.legendre.leggauss(nodes);result=np.zeros(len(self.ef),complex)
        groups=[]
        for energy in np.unique(self.ei):
            initial=self.initial*(self.ei==energy)
            groups.append((energy,initial,self.svy@initial,np.array([j@initial for j in self.ja])))
        keep_y=np.repeat([self._final(k) for k in self.y_keys],[_size(k) for k in self.y_keys])
        for t,weight in zip((x+1)*self.T/2,w*self.T/2):
            f=np.sin(pi*t/self.T)**4;ph=np.exp(-1j*self.omega*t)
            y=np.zeros(len(self.ey),complex);a=np.zeros(len(self.ea),complex)
            initial=np.exp(-1j*self.ei*t)*self.initial
            for ei,_,y0,a0 in groups:
                y+=-1j*np.exp(-1j*self.ey*t)*partial_pulse(ei-self.ey,t,self.T)*y0
                a+=-1j*np.exp(-1j*self.ea*t)*np.sum(time_integral(ei-self.ea[None,:]+self.omega[:,None],t)*a0,axis=0)
            if omit=='outside_intermediates': y*=keep_y
            drive=f*(self.sva@a)*(omit!='before')
            if omit=='spectators': drive-=f*(self.svaspec@a)
            for i,phase in enumerate(ph):
                if omit!='after_quark': drive+=phase*(self.sjq[i]@y)
                if omit=='spectators': drive-=phase*(self.sjqspec[i]@y)
                if omit!='collective': drive+=phase*(self.sjc[i]@y)
                if omit!='fermion_contact': drive+=f*phase*(self.skf[i]@initial)
                if omit!='collective_endpoint': drive+=f*phase*(self.skw[i]@initial)
            result+=-1j*weight*np.exp(-1j*self.ef*(self.T-t))*drive
        return self.wave(self.uf@result)

    def ode_response(self,*,rtol=3e-10,atol=3e-12):
        """Independent physical-basis full sparse initial-value solve."""
        na,ny,nf=len(self.ea),len(self.ey),len(self.ef)
        def rhs(t,vector):
            a=vector[:na];y=vector[na:na+ny];z=vector[na+ny:]
            initial=np.exp(-1j*self.ei*t)*self.initial
            f=np.sin(pi*t/self.T)**4;ph=np.exp(-1j*self.omega*t)
            da=self.ea*a;dy=self.hy@y+f*(self.vy@initial);dz=self.hf@z+f*(self.va@a)
            for i,phase in enumerate(ph):
                da+=phase*(self.ja[i]@initial)
                dz+=phase*(self.jtotal[i]@y)+f*phase*(self.ktotal[i]@initial)
            return -1j*np.concatenate((da,dy,dz))
        sol=solve_ivp(rhs,(0,self.T),np.zeros(na+ny+nf,complex),method='DOP853',rtol=rtol,atol=atol)
        if not sol.success: raise RuntimeError(sol.message)
        return self.wave(sol.y[na+ny:,-1]),sol.nfev

    def diagnostics(self):
        return dict(initial=len(self.ei),source_first=len(self.ea),emission_first=len(self.ey),
            mixed_final=len(self.ef),outside_intermediate_blocks=sum(not self._final(k) for k in self.y_keys),
            final_blocks=len(self.f_keys),quarks=self.nquarks,
            final_domain_is_global_Fock_cutoff=False)
