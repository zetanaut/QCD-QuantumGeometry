"""Complete first-emission action on sparse quark exterior states.

Canonical-spin/LF-helicity transport is a one-particle basis change, not a
solution of interacting nucleon spin. The saved p/n interpolators are test
inputs, never a production wavefunction or scalar-diquark restriction.
All reached modes in a declared output window, both collective nodes and
canonical gluon helicities/colors are retained. No H2/HO/regulator is used.
Only one emission is qualified; this is not a closed higher-order Hamiltonian.
"""
from itertools import permutations, product
from math import factorial, sqrt

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm

from .canonical_lf_fields import GAMMA, GAMMA_PLUS, lf_spinor, slash
from .canonical_lf_network import four_vector, momentum
from .finite_lf_coupling import spinors, pulse_integral
from .graded_fock_network import FockRegister, Mode, Ladder, Word
from .lf_pair_electromagnetic import (
    BETA, BILINEARS, ALPHA, CHARGE, b0, good_spinors, parent_h, PHI_CHARGE, cart,
)
from .modular_baryon_algebra import baryon, one_body, wedge
from .modular_gauge_network import generators


def occupied(mask):
    return tuple(i for i in range(mask.bit_length()) if mask & (1 << i))


def parity(indices):
    return (-1)**sum(a > b for i, a in enumerate(indices) for b in indices[i+1:])


def canonical_spinors(p, mass):
    """Rotationless-boost Dirac u, canonical (up,down), ubar u=2m."""
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('Positive mass required for this canonical rest-spin map.')
    e, x, y, z = four_vector(p, mass)
    sigma_p = np.array([[z, x-1j*y], [x+1j*y, -z]])
    return sqrt(e+mass)*np.vstack((np.eye(2), sigma_p/(e+mass)))


def spin_map(p, mass):
    """R[h,s] with u_canonical = u_LF R; LF order (-,+), canonical (+,-).

    Positive-energy inner product divided by 2E is equivalent to the Dirac
    inner product divided by 2m. No many-body Melosh ansatz is imposed.
    """
    u = np.column_stack([lf_spinor(*momentum(p), mass, h) for h in (-1, 1)])
    return u.conj().T@canonical_spinors(p, mass)/(2*four_vector(p, mass)[0])


def emission_leaf(outgoing, boson, incoming, mass, *, constraint=False):
    """q->q phi_A coefficient (q_out12,A4,q_in12), coupling/V stripped.

    The full-Dirac expression and independent good-field derivative of
    B (1/p+) B/2 both keep positive quark Fourier denominators.
    """
    a, k, p = map(momentum, (outgoing, boson, incoming))
    if not np.allclose(a+k, p, rtol=0, atol=1e-12):
        raise ValueError('Conserved plus/transverse momentum required.')
    if constraint:
        u, v = good_spinors(a, mass), good_spinors(p, mass)
        values = [u.conj().T@(b0(a, mass)@BETA@z/a[0]
                              + BETA@z@b0(p, mass)/p[0])@v/2/sqrt(2*k[0])
                  for z in BILINEARS]
    else:
        u, v = spinors(a, mass), spinors(p, mass)
        values = [u.conj().T@BETA@z@v/sqrt(8*a[0]*k[0]*p[0]) for z in BILINEARS]
    return np.array(values).transpose(1, 0, 2)


def mode_key(p):
    return tuple(float(x) for x in p)


def emission_contact(outgoing, boson, incoming, mass, epsilon, volume, *, covariant=False):
    """Crossed mixed Yukawa/EM fermion contact, g and e stripped.

    a+k=p+q, q+=0, nonzero qT; upper source components (+,-,x,y).
    Derivative of the SAME B (p+-eQA+)^-1 B/2, with both quark Fourier
    denominators positive. This leaf is not the complete sourced baryon
    response: sequential sources, collective endpoints and spectators remain.
    """
    a,k,p=map(momentum,(outgoing,boson,incoming));q=a+k-p
    e=np.asarray(epsilon,complex)
    if (abs(q[0])>1e-12 or np.linalg.norm(q[1:])<1e-12 or e.shape!=(4,)
            or not np.isfinite(e).all() or not np.isfinite(volume) or volume<=0):
        raise ValueError('Finite source/volume, zero-plus and nonzero transverse transfer required.')
    if covariant:
        u,v=spinors(a,mass),spinors(p,mass)
        gp=np.kron(GAMMA_PLUS,np.eye(6));vertex=CHARGE@np.kron(slash(cart(e)),np.eye(6))
        values=[u.conj().T@BETA@(vertex@gp@z/(2*a[0])+z@gp@vertex/(2*p[0]))@v
                /sqrt(8*a[0]*k[0]*p[0]*volume) for z in BILINEARS]
    else:
        u,v=good_spinors(a,mass),good_spinors(p,mass)
        photon=-sum(e[j+2]*ALPHA[j]@CHARGE for j in range(2))
        values=[]
        for z in BILINEARS:
            y=BETA@z
            kernel=(photon@y/a[0]+y@photon/p[0]
                +e[0]*(b0(a,mass)@CHARGE@y/a[0]**2+y@CHARGE@b0(p,mass)/p[0]**2))/2
            values.append(u.conj().T@kernel@v/sqrt(2*k[0]*volume))
    return np.array(values).transpose(1,0,2)


class EmissionLayer:
    """Sparse general quark carrier with the full declared first-emission window.

    Quark modes: momentum, helicity(-,+), flavor(u,d), color. Output keys
    (kind, boson momentum, quark mask) have full 8(phi) or 16(g) amplitudes.
    Output wave amplitudes from different emitters are SUMMED, not treated
    as independent probabilities. No bound-state or source completion follows.
    """
    def __init__(self, cell, incoming_momenta, mass, couplings, gs, shell):
        self.cell = cell
        self.mass = float(mass)
        self.g = np.asarray(couplings, float)
        if (self.g.shape != (2,) or not np.isfinite(self.g).all() or not np.isfinite(gs)
                or not np.isfinite(mass) or mass <= 0):
            raise ValueError('Two real finite Yukawas, finite gs and positive mass required.')
        if isinstance(shell, bool) or int(shell) != shell or shell < 0:
            raise ValueError('Nonnegative integer output shell required.')
        self.gs, self.shell = float(gs), int(shell)
        self.incoming = tuple(mode_key(cell.mode(p, 'fermion')) for p in incoming_momenta)
        if not self.incoming or len(set(self.incoming)) != len(self.incoming):
            raise ValueError('Nonempty distinct incoming momenta required.')
        momenta = set(self.incoming)
        raw = []
        for p0 in self.incoming:
            p = np.array(p0)
            # AP parent n+1/2: periodic boson integer 1,...,n.
            n = int(round(p[0]/cell.spacing[0]-.5))
            for j, x, y in product(range(1, n+1), range(-self.shell, self.shell+1), range(-self.shell, self.shell+1)):
                k = cell.spacing*np.array([j, x, y]); a = cell.mode(p-k, 'fermion')
                if np.max(abs(a[1:]/cell.spacing[1:])) > shell+1e-10:
                    continue
                momenta.add(mode_key(a)); raw.append((mode_key(a), mode_key(k), p0))
        self.momenta = tuple(sorted(momenta))
        self.lookup = {p: i for i, p in enumerate(self.momenta)}
        self.routes = []
        for a, k, p in raw:
            leaf = emission_leaf(a, k, p, mass)
            phi = np.einsum('iAj,n->inAj', leaf, self.g).reshape(12, 8, 12)/sqrt(cell.volume)
            gluon = cell.cubic('q_to_qg', a, k, p, mass, gs).reshape(2, 3, 2, 8, 2, 3)
            gluon = np.einsum('hclasd,fg->hfclasgd', gluon, np.eye(2)).reshape(12, 16, 12)
            for kind, value in (('phi', phi), ('g', gluon)):
                self.routes.append((self.lookup[a], k, self.lookup[p], kind, value))

    def interpolator(self, p, species, spin, *, canonical=True):
        """Transport the saved one-orbital interpolator, preserving exterior signs."""
        return self.wavepacket_interpolator({mode_key(p): 1.}, species, spin, canonical=canonical)

    def wavepacket_interpolator(self, orbital, species, spin, *, canonical=True):
        """Exterior lift of a normalized orbital packet, including unequal momenta.

        This produces a benchmark family only. apply() accepts arbitrary
        correlated exterior inputs; the packet is not a production restriction.
        """
        if (not orbital or any(p not in self.incoming for p in orbital)
                or not all(np.isfinite(v) for v in orbital.values())
                or abs(sum(abs(v)**2 for v in orbital.values())-1) > 1e-12):
            raise ValueError('Normalized finite packet on declared incoming momenta required.')
        matrices = {p: spin_map(p, self.mass) if canonical else np.array([[0, 1], [1, 0]]) for p in orbital}
        result = {}
        for mask, amplitude in baryon(0, species, spin).items():
            state = {0: amplitude}
            for old in occupied(mask):
                f, s, c = old//6, old//3 % 2, old % 3
                state = wedge(state, {1 << (12*self.lookup[p]+6*h+3*f+c): value*matrices[p][h,s]
                                      for p,value in orbital.items() for h in range(2)})
            for key, v in state.items():
                result[key] = result.get(key, 0)+v
        return {k: v for k, v in result.items() if v != 0}

    def _validate_state(self, state):
        if not state:
            raise ValueError('A nonempty finite quark state is required.')
        for mask, value in state.items():
            if not isinstance(mask, int) or mask < 0 or not np.isfinite(value):
                raise ValueError('Nonnegative occupation mask and finite amplitude required.')
            if any(i//12 >= len(self.momenta) or self.momenta[i//12] not in self.incoming for i in occupied(mask)):
                raise ValueError('Input occupies an undeclared incoming momentum.')

    def apply(self, state, *, routes=None):
        """Second-quantized sum b_out^dagger a^dagger b_in on all fermions."""
        self._validate_state(state)
        result = {}
        for a, k, p, kind, leaf in self.routes if routes is None else routes:
            for c in range(leaf.shape[1]):
                entries = {(12*a+int(i), 12*p+int(j)): leaf[i, c, j]
                           for i, j in zip(*np.nonzero(leaf[:, c, :]))}
                transformed = one_body(state, entries)
                for mask, value in transformed.items():
                    key = (kind, k, mask)
                    if key not in result:
                        result[key] = np.zeros(leaf.shape[1], complex)
                    result[key][c] += value
        return result

    def tensor_reference(self, state, *, slots=None):
        """Independent labelled-particle antisymmetric tensor, then exterior projection.

        Factorial reference limited to N<=4, NOT a limitation on apply().
        slots is only an explicit omitted-emitter negative control.
        """
        self._validate_state(state)
        counts = {len(occupied(k)) for k in state}
        if len(counts) != 1 or not 1 <= next(iter(counts)) <= 4:
            raise ValueError('Fixed 1<=N<=4 needed only for this factorial reference.')
        n = next(iter(counts)); selected = range(n) if slots is None else slots
        result = {}
        for mask, amplitude in state.items():
            for initial in permutations(occupied(mask)):
                v = parity(initial)*amplitude/factorial(n)
                for s in selected:
                    p, j = divmod(initial[s], 12)
                    for a, k, parent, kind, leaf in self.routes:
                        if parent != p:
                            continue
                        for i in range(12):
                            if not np.any(leaf[i, :, j]):
                                continue
                            out = list(initial); out[s] = 12*a+i
                            if len(set(out)) != n:
                                continue
                            key = (kind, k, sum(1 << x for x in out))
                            if key not in result:
                                result[key] = np.zeros(leaf.shape[1], complex)
                            result[key] += parity(out)*v*leaf[i, :, j]
        return result

    def adjoint(self, wave):
        """Absorption into the declared input carrier, exact Hermitian adjoint."""
        result = {}
        for a, k, p, kind, leaf in self.routes:
            for c in range(leaf.shape[1]):
                state = {mask: value[c] for (ch, kb, mask), value in wave.items() if ch == kind and kb == k}
                entries = {(12*p+int(j), 12*a+int(i)): np.conj(leaf[i, c, j])
                           for i, j in zip(*np.nonzero(leaf[:, c, :]))}
                for mask, value in one_body(state, entries).items():
                    if all(self.momenta[i//12] in self.incoming for i in occupied(mask)):
                        result[mask] = result.get(mask, 0)+value
        return result

    def graded_reference(self, state, *, kind):
        """Existing Jordan--Wigner tensor compiler, one whole declared channel.

        Sparse contractions, no exponentially large dense register allocation.
        Use a small momentum window for this independent implementation check.
        """
        self._validate_state(state)
        routes = [r for r in self.routes if r[3] == kind]
        if kind not in ('phi', 'g'):
            raise ValueError('Explicit phi/g channel required.')
        n = 12*len(self.momenta)
        labels = sorted({(r[1], c) for r in routes for c in range(r[4].shape[1])})
        modes = [Mode(str(i), 'fermion', 1) for i in range(n)]
        modes += [Mode(str((kind, k, c)), 'boson', 1) for k, c in labels]
        register = FockRegister(modes); index = {key: n+i for i, key in enumerate(labels)}
        words = []
        for a, k, p, _, leaf in routes:
            for i, c, j in zip(*np.nonzero(leaf)):
                words.append(Word(leaf[i,c,j], (Ladder(12*a+int(i),True),
                    Ladder(index[k,int(c)],True), Ladder(12*p+int(j),False))))
        incoming = {tuple(int(bool(mask & (1 << i))) for i in range(n))+(0,)*len(labels): value
                    for mask, value in state.items()}
        output = register.tensor_action(words, incoming)
        result = {}
        for occupations, value in output.items():
            if value == 0:
                continue
            bs = [j for j in range(len(labels)) if occupations[n+j]]
            if len(bs) != 1:
                raise AssertionError('First emission must have exactly one boson.')
            k, c = labels[bs[0]]; mask = sum(1 << i for i in range(n) if occupations[i])
            key = (kind, k, mask)
            if key not in result:
                result[key] = np.zeros(8 if kind == 'phi' else 16, complex)
            result[key][c] += value
        return result

    def energy(self, mask):
        return sum((self.mass**2+np.dot(p[1:],p[1:]))/(2*p[0])
                   for p in (self.momenta[i//12] for i in occupied(mask)))

    def pulse(self, state, mass_squared, duration, *, direct=False):
        """Coefficient of one smoothly switched emission, NOT a bound state.

        All input energies, all emitter interference and node mixing remain.
        Spectral pulse integration and separate adaptive expm quadrature agree.
        The phi/g norms are leading coefficients, not all-order probabilities.
        """
        import torch
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('Finite positive LF duration required.')
        self._validate_state(state)
        grouped = {}
        for mask, v in state.items():
            e = self.energy(mask); grouped.setdefault(e, {})[mask] = v
        result = {}
        for ei, part in grouped.items():
            for (kind, k, mask), vector in self.apply(part).items():
                h = parent_h(mass_squared, k) if kind == 'phi' else np.eye(16)*np.dot(k[1:],k[1:])/(2*k[0])
                h = h+self.energy(mask)*np.eye(len(vector))
                if direct:
                    value, _ = quad_vec(lambda t: -1j*np.sin(np.pi*t/duration)**4
                        *np.exp(-1j*ei*t)*expm(-1j*h*(duration-t))@vector,
                        0, duration, epsabs=1e-11, epsrel=1e-11)
                else:
                    e, u = np.linalg.eigh(h)
                    f = pulse_integral(torch.tensor(ei-e), duration).numpy()
                    value = u@(-1j*np.exp(-1j*e*duration)*f*(u.conj().T@vector))
                key = (kind, k, mask)
                result[key] = result.get(key, 0)+value
        return result


def norm2(wave):
    return float(sum(np.vdot(x,x).real for x in wave.values()))


def difference(a, b):
    return sqrt(norm2({k: a.get(k,0)-b.get(k,0) for k in a.keys() | b.keys()}))


def total_generator(layer, wave, *, charge=False, color=None):
    """Total quark+boson electric/color generator, including gluon adjoint."""
    if charge == (color is not None):
        raise ValueError('Choose electric charge or one color generator.')
    if charge:
        q = np.diag([2/3,-1/3]); quark = np.kron(np.kron(np.eye(2),q),np.eye(3))
        bosons = {'phi': np.kron(np.eye(2),PHI_CHARGE), 'g': np.zeros((16,16))}
    else:
        ts = generators(); t = ts[color]; quark = np.kron(np.eye(4),t)
        # Adjoint Hermitian generator: (T_a)_bc = -i f_abc.
        adj = np.array([[-2*np.trace((t@b-b@t)@c) for c in ts] for b in ts])
        bosons = {'phi': np.zeros((8,8)), 'g': np.kron(np.eye(2),adj)}
    entries = {(12*p+int(i),12*p+int(j)): quark[i,j] for p in range(len(layer.momenta))
               for i,j in zip(*np.nonzero(quark))}
    result = {}
    groups = {(kind,k) for kind,k,_ in wave}
    for kind,k in groups:
        size = len(bosons[kind])
        for c in range(size):
            state = {mask: value[c] for (ch,kb,mask),value in wave.items() if ch==kind and kb==k}
            for mask,value in one_body(state,entries).items():
                key = (kind,k,mask)
                if key not in result:
                    result[key] = np.zeros(size,complex)
                result[key][c] += value
        for (ch,kb,mask),value in wave.items():
            if ch==kind and kb==k:
                key = (kind,k,mask)
                result[key] = result.get(key,0)+bosons[kind]@value
    return result
