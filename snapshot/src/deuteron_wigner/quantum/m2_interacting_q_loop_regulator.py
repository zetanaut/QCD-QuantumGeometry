"""Q-projector-only current for the complete canonical H2 return through g^4.

S=g(V1,0)+g^2(V2,W6P), U=U65+U56 and W=W55. With the embedded INNER
free resolvent R, the only words required are R, RUR, RWR and RURUR.
Differentiate every R, holding the unprojected OUTER S/U/W fixed. This
includes the direct/iterated Q6 cross terms and both Q5 backflow legs.
Adding a separate projected-U/W boundary here would double-count it.

This is not a complete physical current or common field regulator. P/source
projector motion, gauged parent vertices and loop/current counterterms are
separate. Both transverse components are regular path representatives.
"""
import numpy as np

from .m2_fourth_order_h2 import FourthOrderH2, _norm_inverse_sqrt
from .m2_q5_loop_regulator_current import _sources
from .m2_shell_fock_reference import ShellFockReference
from .m2_shell_projector_current import QShellProjectorFamily
from .m2_projected_shell_resolvent import ProjectedShellFreeResolvent
from .m2_wick_h2_matching import WickMatchedFourthOrderH2


# (left source order, right source order, internal order, operator word).
# U changes Q5<->Q6; W acts only on Q5. Structural zeros need not be fit.
RETURN_WORDS = tuple(
    (left, right, order, word)
    for left in (1, 2) for right in (1, 2)
    for order, word in ((0, ("R",)), (1, ("R", "U", "R")),
                        (2, ("R", "W", "R")), (2, ("R", "U", "R", "U", "R")))
    if left+right+order <= 4
)


def _add(a, b):
    return {n: a[n]+b[n] for n in a}


def _inner(a, b):
    return sum(a[n].conj().T@b[n] for n in a)


def _vectors(model, values):
    n = len(model.pilot.matrix)
    a = np.eye(n, dtype=complex) if values is None else np.asarray(values, complex)
    if a.ndim != 2 or a.shape[0] != n or not np.all(np.isfinite(a)):
        raise ValueError("finite retained columns at the named endpoint required")
    return a


class InteractingQLoopRegulatorCurrent:
    """Full H2 Q-only cutoff partner before, or between, its two local norms.

    Endpoints must be actual outer FourthOrderH2 models, or both be paired
    WickMatchedFourthOrderH2 models with identical explicit operators.
    The latter adds the Q5 one-body insertion to W, not H0 or the vertex
    mass. Q6 one-body insertions first return at g^6 and remain excluded.
    Each named Q5/Q6
    inner cutoff is explicit and may differ between sectors. Matching uses
    their actual packed coordinates, not dimension equivalence.
    """
    physical = False
    Q_only_H2_order4_transport_complete = True
    full_H2_order4_regulator_matching_complete = False
    common_field_regulator_complete = False

    def __init__(self, outgoing, incoming, *, inner_cutoffs, energy_out, energy_in,
                 transfer, charge, total_pplus_GeV, quadrature_order):
        self.input_models = {"outgoing": outgoing, "incoming": incoming}
        paired = [isinstance(m, WickMatchedFourthOrderH2) for m in (outgoing, incoming)]
        if any(paired) and not all(paired):
            raise ValueError("both endpoints must use the same paired Wick prescription")
        self.wick_operators = None
        self.wick_Q5 = {}
        self.paired_Wick_in_regulator_wrapper = all(paired)
        if all(paired):
            if outgoing.operators != incoming.operators:
                raise ValueError("identical field trace and finite Wick conditions required")
            self.wick_operators = incoming.operators
            self.wick_Q5 = {"outgoing": outgoing.qq, "incoming": incoming.qq}
            outgoing, incoming = outgoing.base, incoming.base
        for model in (outgoing, incoming):
            if not isinstance(model, FourthOrderH2) or not isinstance(model.reference, ShellFockReference):
                raise ValueError("actual uncompressed canonical outer H2 endpoints required")
        pf, pi = outgoing.pilot, incoming.pilot
        for key in ("total_k2", "cutoff", "target", "width_GeV", "quark_mass_GeV",
                    "gluon_mass_GeV", "coupling"):
            if getattr(pf, key) != getattr(pi, key):
                raise ValueError("H2 regulator endpoints disagree on "+key)
        ef, ei, pp = map(float, (energy_out, energy_in, total_pplus_GeV))
        if not np.all(np.isfinite([ef, ei, pp])) or pp <= 0:
            raise ValueError("finite endpoint energies and positive total P+ required")
        from .m2_gaussian_cluster_h2 import GaussianClusterH2, require_same_cluster_definition
        cluster = require_same_cluster_definition(pf, pi, total_pplus_GeV=pp)
        if cluster:
            if all(paired) or not all(isinstance(m, GaussianClusterH2) for m in (outgoing, incoming)):
                raise ValueError("use actual common-cluster H2 models without an extra Wick wrapper")
            # W already contains this one-body operator; do not insert it
            # again through the older independent-field Wick wrapper.
            self.wick_operators = incoming.wick_operators
        self.outgoing, self.incoming = outgoing, incoming
        self.energy_out, self.energy_in, self.total_pplus_GeV = ef, ei, pp
        sf = {**outgoing.spaces5, **outgoing.spaces6}
        si = {**incoming.spaces5, **incoming.spaces6}
        self.projector = QShellProjectorFamily(sf, si, inner_cutoffs=inner_cutoffs,
            transfer=transfer, b_GeV=pi.width_GeV, charge=charge, quadrature_order=quadrature_order)
        self.resolvents = {}
        self.sources = {}
        for endpoint, model, spaces in (("outgoing", outgoing, sf), ("incoming", incoming, si)):
            self.resolvents[endpoint] = {n: ProjectedShellFreeResolvent(s,
                inner_cutoff=inner_cutoffs[n], b_GeV=pi.width_GeV,
                quark_mass_GeV=pi.quark_mass_GeV, gluon_mass_GeV=pi.gluon_mass_GeV)
                for n, s in spaces.items()}
            src = _sources(model.reference)
            src[1].update({n: np.zeros_like(v) for n, v in model.six.W.items()})
            src[2].update(model.six.W)
            self.sources[endpoint] = src

    def _model(self, endpoint):
        if endpoint not in ("incoming", "outgoing"):
            raise ValueError("explicit incoming or outgoing endpoint required")
        return getattr(self, endpoint)

    def action(self, operator, values, *, endpoint):
        model = self._model(endpoint)
        a = self.projector.columns(values, endpoint=endpoint)
        if operator not in ("R", "H0", "U", "W"):
            raise ValueError("operator must be R, H0, U or W")
        # Only exact structural zeros are skipped. No amplitude threshold,
        # sparsification, source-rank test or approximation is introduced.
        if not any(np.any(v) for v in a.values()):
            return {n: np.zeros_like(v) for n, v in a.items()}
        if operator == "R":
            energy = self.energy_in if endpoint == "incoming" else self.energy_out
            return {n: self.resolvents[endpoint][n].resolve(energy, v) for n, v in a.items()}
        if operator == "H0":
            free = {**model.six.propagation5, **model.six.propagation6}
            return {n: free[n].apply(v) for n, v in a.items()}
        if operator == "U":
            five = {n: a[n] for n in model.spaces5}
            six = {n: a[n] for n in model.spaces6}
            up = (model.six.U.apply(five) if any(np.any(v) for v in five.values())
                  else {n: np.zeros_like(a[n]) for n in six})
            down = (model.six.U.adjoint(six) if any(np.any(v) for v in six.values())
                    else {n: np.zeros_like(a[n]) for n in five})
            return {**up, **down}
        if operator == "W":
            five = model.qq.action.apply({n: a[n] for n in model.spaces5})
            if self.paired_Wick_in_regulator_wrapper:
                five = {n: v+self.wick_Q5[endpoint][n][:, None]*a[n] for n, v in five.items()}
            return {**five,
                    **{n: np.zeros_like(a[n]) for n in model.spaces6}}
        raise ValueError("operator must be R, H0, U or W")

    def regulator_action(self, values, *, component):
        return self.projector.resolvent_boundary(values, component=component,
            resolve_incoming=lambda a: self.action("R", a, endpoint="incoming"),
            resolve_outgoing=lambda a: self.action("R", a, endpoint="outgoing"),
            parent_incoming=lambda a: self.action("H0", a, endpoint="incoming"),
            parent_outgoing=lambda a: self.action("H0", a, endpoint="outgoing"),
            total_pplus_GeV=self.total_pplus_GeV)

    def word_action(self, word, values, *, endpoint):
        a = self.projector.columns(values, endpoint=endpoint)
        for op in reversed(word):
            a = self.action(op, a, endpoint=endpoint)
        return a

    def gauged_word(self, word, values, *, vertices):
        """Leibniz rule: outgoing operators left, incoming operators right.

        vertices maps each gauged operator to a two-endpoint action. Absent
        operators are held fixed, not asserted to have zero physical current.
        The caller may use this to check parent-current/product-rule partners.
        """
        if not word or any(op not in ("R", "U", "W") for op in word):
            raise ValueError("nonempty H2 return word required")
        if not set(vertices) <= {"R", "U", "W"}:
            raise ValueError("only the explicitly named word vertices are allowed")
        a = self.projector.columns(values, endpoint="incoming")
        count = next(iter(a.values())).shape[1]
        da = {n: np.zeros((s.dimension, count), complex) for n, s in self.projector.outgoing.items()}
        for op in reversed(word):
            da = self.action(op, da, endpoint="outgoing")
            if op in vertices:
                da = _add(da, self.projector.columns(vertices[op](a), endpoint="outgoing"))
            a = self.action(op, a, endpoint="incoming")
        return da

    def _source_columns(self, endpoint, vectors):
        return {p: {n: a@vectors for n, a in src.items()}
                for p, src in self.sources[endpoint].items()}

    def _cached_words(self, endpoint, sources):
        # Per-call exact vectors only: no rank reduction or global state.
        cache = {((), p): a for p, a in sources.items()}
        def get(word, power):
            key = (tuple(word), power)
            if key not in cache:
                cache[key] = self.action(word[0], get(word[1:], power), endpoint=endpoint)
            return cache[key]
        return get

    def coefficients(self, *, outgoing_vectors=None, incoming_vectors=None):
        """Order 2,3,4 transverse coefficients, BEFORE retained whitening."""
        vf, vi = _vectors(self.outgoing, outgoing_vectors), _vectors(self.incoming, incoming_vectors)
        left, right = self._source_columns("outgoing", vf), self._source_columns("incoming", vi)
        lf, ri = self._cached_words("outgoing", left), self._cached_words("incoming", right)
        acted = {}
        result = {p: np.zeros((2, vf.shape[1], vi.shape[1]), complex) for p in (2, 3, 4)}
        for l, r, order, word in RETURN_WORDS:
            for k, op in enumerate(word):
                if op != "R":
                    continue
                # Every endpoint R/U/W is Hermitian at its real energy.
                # Move the left word to the bra; retain its reversed order.
                bra = lf(tuple(reversed(word[:k])), l)
                for j in (0, 1):
                    key = (word[k+1:], r, j)
                    if key not in acted:
                        acted[key] = self.regulator_action(ri(word[k+1:], r), component=j)
                    result[l+r+order][j] += _inner(bra, acted[key])
        return result

    def return_coefficients(self, *, endpoint, norm=False):
        """Self-energy or -d_E self-energy, independent of the path current.

        In the latter, insert +R^2 at every R in the same words. This is the
        full Q6 norm and both five-body backflow terms, not just A6^dag A6.
        """
        if not isinstance(norm, (bool, np.bool_)):
            raise ValueError("explicit norm boolean required")
        model = self._model(endpoint)
        src = self.sources[endpoint]
        waves = self._cached_words(endpoint, src)
        size = len(model.pilot.matrix)
        result = {p: np.zeros((size, size), complex) for p in (2, 3, 4)}
        for l, r, order, word in RETURN_WORDS:
            if not norm:
                value = _inner(src[l], waves(word, r))
            else:
                # At equal endpoints, -R'(E)=R^2. Factor the inserted
                # R^2 into bra/ket waves to reuse exact propagations.
                value = sum(_inner(waves(("R",)+tuple(reversed(word[:k])), l),
                                   waves(("R",)+word[k+1:], r))
                            for k, op in enumerate(word) if op == "R")
            result[l+r+order] += value
        return result

    def metric(self, *, endpoint):
        model = self._model(endpoint)
        g = model.pilot.coupling
        n = np.eye(len(model.pilot.matrix), dtype=complex)
        for power, value in self.return_coefficients(endpoint=endpoint, norm=True).items():
            n += g**power*value
        if np.linalg.norm(n-n.conj().T) > 3e-11:
            raise ArithmeticError("H2 Q-loop norm failed Hermiticity")
        return n

    def effective_hamiltonian(self, *, endpoint):
        """Same explicit inner-Q H2, including paired retained g^2 mass terms.

        This does not add a P-projector current, change the vertex mass,
        or resum the Q interaction. It supplies an independent comparison
        to an inner canonical or WickMatchedFourthOrderH2 snapshot.
        """
        model = self._model(endpoint)
        g = model.pilot.coupling
        h = model.pilot.matrix.copy()
        if self.paired_Wick_in_regulator_wrapper:
            h += g*g*self.input_models[endpoint].pp
        for power, value in self.return_coefficients(endpoint=endpoint).items():
            h += g**power*value
        if not np.all(np.isfinite(h)) or np.linalg.norm(h-h.conj().T) > 3e-11:
            raise ArithmeticError("H2 Q-loop Hamiltonian failed Hermiticity")
        return h

    def current(self, *, normalized, outgoing_vectors=None, incoming_vectors=None):
        """Explicitly choose raw or locally whitened Q-only correction.

        Whitening is exactly the existing finite H2 convention, not an
        additional regulator-current term or an all-order renormalization.
        P/current source partners remain required for a full current.
        """
        if not isinstance(normalized, (bool, np.bool_)):
            raise ValueError("explicit normalization choice required")
        vf, vi = _vectors(self.outgoing, outgoing_vectors), _vectors(self.incoming, incoming_vectors)
        if normalized:
            vf = _norm_inverse_sqrt(self.metric(endpoint="outgoing"))@vf
            vi = _norm_inverse_sqrt(self.metric(endpoint="incoming"))@vi
        c = self.coefficients(outgoing_vectors=vf, incoming_vectors=vi)
        g = self.incoming.pilot.coupling
        return sum(g**p*a for p, a in c.items())
