"""Retained/source and Q-projector currents in one finite H2 convention.

All P-return boundaries use actual outer-P source actions on selected columns,
never a dense Q-by-Pouter source matrix. Both endpoint energies and full Q
return words (including an explicitly paired Wick insertion) are retained.
This supplies oscillator-projector terms only: not parent physical currents,
a common field regulator, gauged contractions or finite physical matching.
"""
import numpy as np

from .m2_interacting_q_loop_regulator import InteractingQLoopRegulatorCurrent, RETURN_WORDS, _inner, _vectors
from .m2_fourth_order_h2 import _norm_inverse_sqrt
from .m2_matched_h2_reduction import nested_pilot_isometry
from .m2_galerkin_projector_current import retained_projector_vertex_between
from .m2_excluded_fock_action import apply_excluded_source
from .m2_instantaneous_pair_exit import apply_pair_source
from .m2_instantaneous_sea_exit import apply_instantaneous_sea_source
from .m2_six_body_quartic import COMPONENTS, apply_direct_source
from .m2_source_vectors import validate_source_vectors


def canonical_sources_on_vectors(pilot, vectors, *, spaces):
    """All coherent Q5/Q6 sources, preserving an explicit cluster pilot's kernel."""
    from .m2_gaussian_cluster_h2 import GaussianClusterPilot
    from .m2_gaussian_field_sources import apply_field_source
    v = validate_source_vectors(pilot, vectors)
    if set(spaces) != {"qqqgg", "qqqq-qbar", "qqqggg", "qqqq-qbar-g"}:
        raise ValueError("all four canonical raw Q sectors required")
    result = {p: {n: np.zeros((s.dimension, v.shape[1]), complex) for n, s in spaces.items()}
              for p in (1, 2)}
    def checked(image):
        target = spaces[image.space.sector]
        for name in ("sector", "total_k2", "cutoff", "target", "jz_twice"):
            if getattr(image.space, name) != getattr(target, name):
                raise ValueError("source action disagrees with Q coordinates on "+name)
        if (image.space.longitudinal != target.longitudinal or image.space.orbitals != target.orbitals
                or not np.array_equal(image.space.coordinates, target.coordinates)):
            raise ValueError("exact raw source-action coordinate map is missing")
        return image.values
    if isinstance(pilot, GaussianClusterPilot):
        def regulated(kind, sector):
            return checked(apply_field_source(pilot, v, kind=kind, out_cutoff=spaces[sector].cutoff,
                regulator=pilot.field_regulator, total_pplus_GeV=pilot.total_pplus_GeV,
                total_transverse_GeV=(0., 0.)))
        for n in ("qqqgg", "qqqq-qbar"):
            result[1][n] = regulated(n, n)
        result[2]["qqqgg"] = regulated("combined", "qqqgg")
        result[2]["qqqq-qbar"] = regulated("sea", "qqqq-qbar")
        for component in COMPONENTS:
            n = "qqqq-qbar-g" if component in ("valence_sea", "gluon_IF_pair", "gluon_IG_pair") else "qqqggg"
            result[2][n] += regulated(component, n)
        return result
    for n in ("qqqgg", "qqqq-qbar"):
        result[1][n] = checked(apply_excluded_source(pilot, n, v, out_cutoff=spaces[n].cutoff))
    result[2]["qqqgg"] = checked(apply_pair_source(pilot, v, out_cutoff=spaces["qqqgg"].cutoff,
                                                component="combined"))
    result[2]["qqqq-qbar"] = checked(apply_instantaneous_sea_source(
        pilot, v, out_cutoff=spaces["qqqq-qbar"].cutoff))
    for component in COMPONENTS:
        n = "qqqq-qbar-g" if component in ("valence_sea", "gluon_IF_pair", "gluon_IG_pair") else "qqqggg"
        image = apply_direct_source(pilot, v, out_cutoff=spaces[n].cutoff, component=component)
        result[2][n] += checked(image)
    return result


class H2ProjectorCurrent:
    physical = False
    finite_P_Q_oscillator_projector_transport_complete = True
    common_field_regulator_complete = False
    complete_physical_four_current = False

    def __init__(self, Q_current, *, parent_outgoing, parent_incoming, quadrature_order):
        if not isinstance(Q_current, InteractingQLoopRegulatorCurrent):
            raise ValueError("explicit canonical or paired-Wick Q-regulator engine required")
        self.Q_current = Q_current
        self.parents = {"outgoing": parent_outgoing, "incoming": parent_incoming}
        from .m2_gaussian_cluster_h2 import require_same_cluster_definition
        for endpoint, parent in self.parents.items():
            require_same_cluster_definition(Q_current._model(endpoint).pilot, parent,
                                            total_pplus_GeV=Q_current.total_pplus_GeV)
        self.embeddings = {
            endpoint: nested_pilot_isometry(Q_current._model(endpoint).pilot, parent)
            for endpoint, parent in self.parents.items()}
        first = next(iter(Q_current.projector.vertices.values()))
        self.transfer, self.charge = first.transfer.copy(), first.charge
        if Q_current.outgoing.pilot.cutoff != Q_current.incoming.pilot.cutoff:
            raise ValueError("same inner P cutoff at both current endpoints required")
        self.vertices = retained_projector_vertex_between(
            self.parents["outgoing"], self.parents["incoming"],
            inner_cutoff=self.Q_current.incoming.pilot.cutoff,
            transfer=self.transfer, charge=self.charge, quadrature_order=quadrature_order)
        self.parent_hamiltonians = {a: p.matrix.copy() for a, p in self.parents.items()}
        for endpoint, parent in self.parents.items():
            expected = Q_current._model(endpoint).pilot.matrix.copy()
            if Q_current.paired_Wick_in_regulator_wrapper:
                g2 = parent.coupling**2
                self.parent_hamiltonians[endpoint] += g2*Q_current.wick_operators.retained_mass_coefficient(parent)
                expected += g2*Q_current.input_models[endpoint].pp
            p = self.embeddings[endpoint]
            reduced = p.conj().T@self.parent_hamiltonians[endpoint]@p
            if not np.allclose(reduced, expected, atol=3e-11, rtol=3e-12):
                raise ValueError("P parent and Q engine must represent the same inner Hamiltonian")

    def source_columns(self, endpoint, vectors):
        if endpoint not in self.parents:
            raise ValueError("explicit incoming or outgoing endpoint required")
        return canonical_sources_on_vectors(self.parents[endpoint], vectors,
            spaces=getattr(self.Q_current.projector, endpoint))

    def parent_return_bilinears(self, *, endpoint, left, right, norm=False):
        """Exact finite-parent return on arbitrary columns, for Ward/reference checks."""
        if endpoint not in self.parents:
            raise ValueError("explicit incoming or outgoing endpoint required")
        if not isinstance(norm, (bool, np.bool_)):
            raise ValueError("explicit norm boolean required")
        l = validate_source_vectors(self.parents[endpoint], left)
        r = validate_source_vectors(self.parents[endpoint], right)
        src = self.source_columns(endpoint, np.column_stack((l, r)))
        left_src = {p: {n: a[:, :l.shape[1]] for n, a in s.items()} for p, s in src.items()}
        right_src = {p: {n: a[:, l.shape[1]:] for n, a in s.items()} for p, s in src.items()}
        waves = self.Q_current._cached_words(endpoint, right_src)
        left_waves = self.Q_current._cached_words(endpoint, left_src)
        result = {p: np.zeros((l.shape[1], r.shape[1]), complex) for p in (2, 3, 4)}
        for lp, rp, order, word in RETURN_WORDS:
            if norm:
                value = sum(_inner(left_waves(("R",)+tuple(reversed(word[:k])), lp),
                                   waves(("R",)+word[k+1:], rp))
                            for k, op in enumerate(word) if op == "R")
            else:
                value = _inner(left_src[lp], waves(word, rp))
            result[lp+rp+order] += value
        return result

    def _check_inner_sources(self, endpoint, applied, vectors):
        # Prove the shared-source equality on the actual requested columns.
        # In normalized currents these are already whitened columns. No
        # source identity is inferred merely from matching dimensions.
        expected = self.Q_current._source_columns(endpoint, vectors)
        for power, sources in applied.items():
            for name, value in sources.items():
                ref = expected[power][name]
                if np.linalg.norm(value-ref) > 3e-11*max(1., np.linalg.norm(ref)):
                    raise ValueError("outer P source and Q engine disagree on the requested inner columns")

    def loop_coefficients(self, *, outgoing_vectors, incoming_vectors):
        """P-only return coefficients at g^2,g^3,g^4, before both norms.

        B_i = S_i K_P^dag p_f v_f, A_i = S_i p_i v_i;
        A_f = S_f p_f v_f, B_f = S_f K_P p_i v_i.
        Return [B_i^dag C_i A_i + A_f^dag C_f B_f]/(2P+).
        No additional Q-projector source vertex is inserted here.
        """
        q = self.Q_current
        vf, vi = _vectors(q.outgoing, outgoing_vectors), _vectors(q.incoming, incoming_vectors)
        pf, pi = self.embeddings["outgoing"], self.embeddings["incoming"]
        a_f, a_i = pf@vf, pi@vi
        nf, ni = vf.shape[1], vi.shape[1]
        src_i = self.source_columns("incoming", np.column_stack(
            (a_i, *(k.conj().T@a_f for k in self.vertices))))
        src_f = self.source_columns("outgoing", np.column_stack(
            (a_f, *(k@a_i for k in self.vertices))))
        # Move C_f to the bra using its reversed Hermitian word. Thus only
        # unperturbed A columns need to be propagated, not every K column.
        base_i = {p: {n: a[:, :ni] for n, a in src.items()} for p, src in src_i.items()}
        base_f = {p: {n: a[:, :nf] for n, a in src.items()} for p, src in src_f.items()}
        self._check_inner_sources("incoming", base_i, vi)
        self._check_inner_sources("outgoing", base_f, vf)
        wi, wf = q._cached_words("incoming", base_i), q._cached_words("outgoing", base_f)
        result = {p: np.zeros((2, nf, ni), complex) for p in (2, 3, 4)}
        for l, r, order, word in RETURN_WORDS:
            ci, cf = wi(word, r), wf(tuple(reversed(word)), l)
            for j in (0, 1):
                bi = {n: a[:, ni+j*nf:ni+(j+1)*nf] for n, a in src_i[l].items()}
                bf = {n: a[:, nf+j*ni:nf+(j+1)*ni] for n, a in src_f[r].items()}
                result[l+r+order][j] += (_inner(bi, ci)+_inner(cf, bf))/(2*q.total_pplus_GeV)
        return result

    def retained_current(self, *, outgoing_vectors, incoming_vectors):
        """Actual outer retained H boundary, with the same paired Wick PP term."""
        q = self.Q_current
        vf, vi = _vectors(q.outgoing, outgoing_vectors), _vectors(q.incoming, incoming_vectors)
        f, i = self.embeddings["outgoing"]@vf, self.embeddings["incoming"]@vi
        h = self.parent_hamiltonians
        return np.array([(f.conj().T@k@h["incoming"]@i
                          +f.conj().T@h["outgoing"]@k@i)/(2*q.total_pplus_GeV) for k in self.vertices])

    def current(self, *, normalized, outgoing_vectors, incoming_vectors):
        """Sum P-H, P-return and Q-return cutoff partners, not physical tree currents."""
        if not isinstance(normalized, (bool, np.bool_)):
            raise ValueError("explicit normalization choice required")
        q = self.Q_current
        vf, vi = _vectors(q.outgoing, outgoing_vectors), _vectors(q.incoming, incoming_vectors)
        if normalized:
            vf = _norm_inverse_sqrt(q.metric(endpoint="outgoing"))@vf
            vi = _norm_inverse_sqrt(q.metric(endpoint="incoming"))@vi
        args = dict(outgoing_vectors=vf, incoming_vectors=vi)
        p = self.loop_coefficients(**args)
        r = q.coefficients(**args)
        g = q.incoming.pilot.coupling
        return self.retained_current(**args)+sum(g**n*(p[n]+r[n]) for n in p)
