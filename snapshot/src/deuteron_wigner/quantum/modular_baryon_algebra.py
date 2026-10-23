"""Sparse fermionic interpolator algebra for variable nuclear composition.

This creates color-singlet, spin/isospin-1/2 uud/udd interpolating operators,
not solved nucleon wavefunctions. Their products are states in a finite quark
orbital carrier, not nuclear bound states. Composite creation operators obey
odd-fermion exchange, but their creation/annihilation algebra is not assumed
to be that of independent elementary nucleon modes at overlapping orbitals.
"""

from functools import lru_cache
from itertools import permutations, product
from math import fsum

import numpy as np


def norm2(state):
    return fsum(float(abs(v)**2) for v in state.values())


def _bit_count(value):
    # Preserve the package's Python >=3.9 compatibility.
    return value.bit_count() if hasattr(int, "bit_count") else bin(value).count("1")


def add(states):
    result = {}
    for factor, state in states:
        for key, value in state.items():
            result[key] = result.get(key, 0)+factor*value
    return {k: v for k, v in result.items() if v != 0}


def wedge(left, right):
    """Exterior product in an ordered occupation basis, including cross exchange."""
    result = {}
    for a, x in left.items():
        for b, y in right.items():
            if a & b:
                continue
            inversions = 0
            remaining = a
            while remaining:
                bit = remaining & -remaining
                inversions += _bit_count(b & (bit-1))
                remaining -= bit
            key = a | b
            result[key] = result.get(key, 0)+(-1)**inversions*x*y
    return {k: v for k, v in result.items() if v != 0}


def _mode(site, flavor, spin, color):
    # flavor 0=u,1=d; spin 0=up,1=down; three colors.
    return 12*site+6*flavor+3*spin+color


@lru_cache(maxsize=128)
def _baryon(site, species, spin):
    if isinstance(site, bool) or not isinstance(site, int) or site < 0 or species not in ("p", "n") or spin not in (-1, 1):
        raise ValueError("Nonnegative orbital, p/n species and twice spin +/-1 required.")
    terms = []
    for colors in permutations(range(3)):
        parity = (-1)**sum(colors[i] > colors[j] for i in range(3) for j in range(i+1, 3))
        for s, t, sign in ((0, 1, 1), (1, 0, -1)):
            modes = (_mode(site, 0, s, colors[0]), _mode(site, 1, t, colors[1]),
                     _mode(site, int(species == "n"), int(spin == -1), colors[2]))
            state = {0: 1.}
            for mode in modes:
                state = wedge(state, {1 << mode: 1.})
            terms.append((parity*sign, state))
    result = add(terms)
    scale = np.sqrt(norm2(result))
    return tuple((key, value/scale) for key, value in result.items())


def baryon(site, species, spin):
    """Normalized one-orbital epsilon_color (ud)_S=I=0 q interpolator."""
    return dict(_baryon(site, species, spin))


def one_body(state, entries):
    """Apply sum_ij entries[i,j] c_i^dagger c_j with exact fermion signs."""
    result = {}
    for (i, j), coefficient in entries.items():
        if coefficient == 0:
            continue
        for mask, value in state.items():
            if not mask & (1 << j):
                continue
            sign = (-1)**_bit_count(mask & ((1 << j)-1))
            removed = mask ^ (1 << j)
            if removed & (1 << i):
                continue
            sign *= (-1)**_bit_count(removed & ((1 << i)-1))
            key = removed | (1 << i)
            result[key] = result.get(key, 0)+sign*coefficient*value
    return {k: v for k, v in result.items() if v != 0}


def spin_action(state, sites, kind):
    if kind not in ("z", "+", "-"):
        raise ValueError("Choose z,+,- spin generator.")
    entries = {}
    for site in sites:
        for f, c in product(range(2), range(3)):
            up, down = _mode(site, f, 0, c), _mode(site, f, 1, c)
            if kind == "z":
                entries[up, up], entries[down, down] = .5, -.5
            else:
                entries[(up, down) if kind == "+" else (down, up)] = 1.
    return one_body(state, entries)


def spin_squared(state, sites):
    z = spin_action(state, sites, "z")
    minus = spin_action(state, sites, "-")
    return add(((1, spin_action(z, sites, "z")), (1, spin_action(minus, sites, "+")), (-1, z)))


def color_action(state, site, generator):
    entries = {(_mode(site, f, s, a), _mode(site, f, s, b)): generator[a, b]
               for f, s, a, b in product(range(2), range(2), range(3), range(3))}
    return one_body(state, entries)


def isospin_raise(state, sites):
    return one_body(state, {(_mode(site, 0, s, c), _mode(site, 1, s, c)): 1.
                            for site in sites for s, c in product(range(2), range(3))})


def quantum_numbers(mask):
    quarks = _bit_count(mask)
    up = sum(bool(mask & (1 << i)) for i in range(mask.bit_length()) if (i % 12)//6 == 0)
    return {"quarks": quarks, "baryon_number": quarks/3, "charge": (3*up-quarks)/3}


@lru_cache(maxsize=128)
def _cg(j1, m1, j2, m2, j, m):
    from sympy import Rational
    from sympy.physics.wigner import clebsch_gordan
    return float(clebsch_gordan(*[Rational(x, 2) for x in (j1, j2, j, m1, m2, m)]))


def coupled_cluster(species, sites, intermediate_twice, magnetic_twice):
    """Left-associated exact SU(2) spin coupling; no nuclear interaction.

    Intermediate spins list total 2J after adding particles 2,...,A. Flavor
    is resolved explicitly, not asserted to be a total-isospin eigenstate.
    Returned composite norm is NOT forced to one: exchange can alter it.
    """
    count = len(species)
    if count < 1 or len(sites) != count or len(intermediate_twice) != count-1:
        raise ValueError("One orbital/species per baryon and A-1 intermediate spins required.")
    previous = 1
    for target in intermediate_twice:
        if target not in (previous+1, previous-1) or target < 0:
            raise ValueError("Forbidden spin coupling.")
        previous = target
    if abs(magnetic_twice) > previous or (magnetic_twice-previous) % 2:
        raise ValueError("Magnetic label outside target representation.")
    terms = []
    for spins in product((-1, 1), repeat=count):
        if sum(spins) != magnetic_twice:
            continue
        coefficient, j, m = 1., 1, spins[0]
        for spin, target in zip(spins[1:], intermediate_twice):
            coefficient *= _cg(j, m, 1, spin, target, m+spin)
            j, m = target, m+spin
        if coefficient == 0:
            continue
        state = {0: 1.}
        for site, particle, spin in zip(sites, species, spins):
            state = wedge(state, baryon(site, particle, spin))
        terms.append((coefficient, state))
    return add(terms)


class ClusterEmbedding:
    """Common physical Gram map for nonorthogonal composite module columns.

    Normalization, Pauli quotient and operator transport use the SAME map.
    A tolerance is explicit; it identifies numerically unresolved Gram modes,
    not a general license to discard state-weak operator-sensitive directions.
    Only small, directly checked exact/null examples are qualified here.
    """
    def __init__(self, columns, *, relative_tolerance):
        from scipy import sparse
        self.columns = tuple(dict(c) for c in columns)
        if not self.columns or not 0 < relative_tolerance < 1:
            raise ValueError("Columns and explicit 0<tolerance<1 required.")
        self.occupations = tuple(sorted({k for column in self.columns for k in column}))
        lookup = {mask: i for i, mask in enumerate(self.occupations)}
        rows, cols, values = [], [], []
        for j, column in enumerate(self.columns):
            for mask, value in column.items():
                rows.append(lookup[mask]); cols.append(j); values.append(value)
        if not np.all(np.isfinite(values)):
            raise ValueError("Finite cluster amplitudes required.")
        self.matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(lookup), len(self.columns)), dtype=complex)
        self.gram = (self.matrix.getH()@self.matrix).toarray()
        eigenvalues, vectors = np.linalg.eigh(self.gram)
        if eigenvalues[-1] <= 0:
            raise ValueError("All proposed cluster columns vanish by Pauli exclusion.")
        active = eigenvalues > relative_tolerance*eigenvalues[-1]
        self.gram_eigenvalues = eigenvalues
        self.rank = int(active.sum())
        self.whitener = vectors[:, active]/np.sqrt(eigenvalues[active])
        self.null_vectors = vectors[:, ~active]
        self.isometry = self.matrix@self.whitener

    def kernel(self, action):
        """E^dagger O E from independently applied full-carrier operator O."""
        result = np.zeros((len(self.columns), len(self.columns)), complex)
        for j, column in enumerate(self.columns):
            transformed = action(column)
            for i, left in enumerate(self.columns):
                result[i, j] = sum(np.conj(v)*transformed.get(k, 0) for k, v in left.items())
        return result

    def operator(self, action):
        return self.whitener.conj().T@self.kernel(action)@self.whitener

    def full_state(self, coefficients):
        c = np.asarray(coefficients, complex)
        if c.shape != (len(self.columns),):
            raise ValueError("One coefficient per cluster column required.")
        vector = self.matrix@c
        length = np.linalg.norm(vector)
        if length < 1e-13*np.linalg.norm(c):
            raise ValueError("Coefficients represent a null or unresolved physical state.")
        if length == 0:
            raise ValueError("Zero physical state.")
        return vector/length
