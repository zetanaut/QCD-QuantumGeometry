"""Nested finite-reference H2 wave defects and good-current error bounds.

These compare two explicitly identified finite Q spaces with one unchanged
retained pilot. They do not bound the infinite oscillator/Fock complement,
alter an operator, select finite physical conditions, or supply J-minus.
"""
from dataclasses import dataclass
import numpy as np

from .m2_canonical_good_current import parton_charge, good_current_density_between
from .m2_excluded_fock_kinematics import SPECIES as FIVE
from .m2_six_body_kinematics import SPECIES as SIX
from .m2_shell_current import shell_density_between


def nested_rows(smaller, larger):
    """Exact packed injection, checked by physical coordinates, not dimension."""
    for name in ('sector', 'total_k2', 'target', 'jz_twice', 'ncolor'):
        if getattr(smaller, name) != getattr(larger, name):
            raise ValueError('nested spaces disagree on ' + name)
    if smaller.cutoff > larger.cutoff or not np.array_equal(smaller.spins, larger.spins):
        raise ValueError('nested shells with identical ordered spin/color coordinates required')
    li = {tuple(s): k for k, s in enumerate(larger.longitudinal)}
    oi = {tuple(s): k for k, s in enumerate(larger.orbitals)}
    try:
        longs = np.array([li[tuple(s)] for s in smaller.longitudinal])
        orbs = np.array([oi[tuple(s)] for s in smaller.orbitals])
    except KeyError as exc:
        raise ValueError('missing physical coordinate in reference space') from exc
    c = smaller.coordinates
    raw = np.ravel_multi_index((longs[c[:, 0]], orbs[c[:, 1]], c[:, 2], c[:, 3]), larger.shape)
    rows = larger.lookup[raw]
    if np.any(rows < 0) or len(np.unique(rows)) != smaller.dimension:
        raise ValueError('packed injection must be a one-to-one exact coordinate map')
    return rows


def _inject(values, smaller, larger):
    a = np.asarray(values, complex)
    if a.ndim != 2 or a.shape[0] != smaller.dimension or not np.all(np.isfinite(a)):
        raise ValueError('finite columns on the explicit smaller space required')
    result = np.zeros((larger.dimension, a.shape[1]), complex)
    result[nested_rows(smaller, larger)] = a
    return result


def _gap(propagation, energy):
    gap = min(float(np.min(abs(energy - block.mass - block.eigenvalues)))
              for block in propagation.blocks)
    if not np.isfinite(gap) or gap < 1e-10:
        raise ValueError('off-pole finite-reference resolvent required; no clipping')
    return gap


@dataclass(frozen=True)
class WaveDefect:
    lower: object
    reference_model: object
    embedded: dict
    errors: dict
    raw_defects: dict
    effective_defects: dict
    error_bounds: dict
    gaps: dict
    leading_defect_parts: dict


def wave_defect(lower, reference_model):
    """Transport exact finite-reference residuals in triangular H2 order.

    A1/A2 errors are R5 r1/r2; A6 error is R6(r6+U dA1);
    C3 error is R5(r3+W55 dA1+Udag dA6). The same source,
    auxiliary-projector and Hamiltonian actions as the reference are used.
    No compression identity for W55 or coupling resummation is assumed.
    """
    m, r = lower.owner, reference_model
    if m.pilot is not r.pilot:
        raise ValueError('one unchanged retained pilot object required')
    if m.spaces5.keys() != r.spaces5.keys() or m.spaces6.keys() != r.spaces6.keys():
        raise ValueError('same explicitly represented Fock sectors required')
    energy = lower.energy_GeV2
    embedded = {}
    for key, waves, ls, rs in (
        ('A1', lower.five_wave_coefficients[0], m.spaces5, r.spaces5),
        ('A2', lower.five_wave_coefficients[1], m.spaces5, r.spaces5),
        ('C3', lower.five_wave_coefficients[2], m.spaces5, r.spaces5),
        ('A6', lower.six_wave_coefficient, m.spaces6, r.spaces6),
    ):
        embedded[key] = {n: _inject(v, ls[n], rs[n]) for n, v in waves.items()}
    p5, p6 = r.qq.propagation, r.six.propagation6
    gaps = {n: _gap(p, energy) for n, p in {**p5, **p6}.items()}
    raw, effective, errors, bounds, leading_parts = {}, {}, {}, {}, {}

    def carry(key, source, propagation, correction=None):
        a = embedded[key]
        raw[key] = {n: source[n] - energy*a[n] + propagation[n].apply(a[n]) for n in a}
        effective[key] = {n: raw[key][n] + (0. if correction is None else correction[n]) for n in a}
        errors[key] = {n: propagation[n].resolve(energy, v) for n, v in effective[key].items()}
        bounds[key] = {n: float(np.linalg.norm(v))/gaps[n] for n, v in effective[key].items()}

    for key, power in (('A1', 1), ('A2', 2)):
        source = r.qq.sources_by_power[power]
        # Actual source restriction is verified, not inferred from matrix size.
        for n in source:
            np.testing.assert_allclose(source[n][nested_rows(m.spaces5[n], r.spaces5[n])],
                                       m.qq.sources_by_power[power][n], atol=3e-14, rtol=3e-12)
        carry(key, source, p5)
        leading_parts[key] = {}
        for n in source:
            ls, rs = m.spaces5[n], r.spaces5[n]
            low_wave = lower.five_wave_coefficients[power-1][n]
            low_source = m.qq.sources_by_power[power][n]
            low_h = m.qq.propagation[n].apply(low_wave)
            leading_parts[key][n] = dict(
                source=source[n]-_inject(low_source, ls, rs),
                kinetic=p5[n].apply(embedded[key][n])-_inject(low_h, ls, rs),
                lower_equation=_inject(low_source-energy*low_wave+low_h, ls, rs))
    u = r.six.U.apply(embedded['A1'])
    carry('A6', {n: r.six.W[n]+u[n] for n in u}, p6, r.six.U.apply(errors['A1']))
    w, u = r.qq.action.apply(embedded['A1']), r.six.U.adjoint(embedded['A6'])
    dw, du = r.qq.action.apply(errors['A1']), r.six.U.adjoint(errors['A6'])
    carry('C3', {n: w[n]+u[n] for n in w}, p5, {n: dw[n]+du[n] for n in dw})
    return WaveDefect(lower, r, embedded, errors, raw, effective, bounds, gaps, leading_parts)


def _charge_bound(space, charge):
    species = (FIVE if space.sector in FIVE else SIX)[space.sector]
    spins = np.unique(space.coordinates[:, 3])
    # Sum of absolute physical charges on each actual spin/flavor tuple.
    # Translation is unitary before orthogonal projection, hence a contraction.
    return float(np.max(sum(abs(parton_charge(space.spins[spins, k], s, charge))
                            for k, s in enumerate(species))))


def plus_and_metric_bounds(outgoing, incoming, transfer, *, charge):
    """Bound each good-current coefficient using the transported wave errors.

    For Aref=A+da and Bref=B+db, use either exact telescoping order:
      ||da||F ||Bref||2 + ||A||2 ||db||F,
      ||Aref||2 ||db||F + ||da||F ||B||2.
    Multiply by the projected density norm bound. At zero transfer the
    metric uses the identity instead of charge, including every g4 cross term.
    Both positive endpoint metric factors are included in the final bound.
    This is finite-reference numerical evaluation of an analytic inequality,
    not interval certification, a physical error budget or a four-current bound.
    """
    f, i = outgoing, incoming
    if f.reference_model is not i.reference_model or f.lower.owner is not i.lower.owner:
        raise ValueError('one explicit lower/reference model pair at both energies required')
    r = i.reference_model
    delta = np.asarray(transfer, float)
    if delta.shape != (2,) or not np.all(np.isfinite(delta)):
        raise ValueError('finite two-component transfer required')
    terms = {2: (('A1', 'A1'),), 3: (('A1', 'A2'), ('A2', 'A1')),
             4: (('A2', 'A2'), ('A1', 'C3'), ('C3', 'A1'), ('A6', 'A6'))}

    def coefficients(left, right, *, metric=False):
        values = {n: 0. for n in range(5)}
        for power, pairs in terms.items():
            for a, b in pairs:
                spaces = r.spaces6 if a == 'A6' else r.spaces5
                for sector, space in spaces.items():
                    av, bv = left.embedded[a][sector], right.embedded[b][sector]
                    ar = av + left.errors[a][sector]
                    br = bv + right.errors[b][sector]
                    ea, eb = left.error_bounds[a][sector], right.error_bounds[b][sector]
                    bound = min(ea*np.linalg.norm(br, 2)+np.linalg.norm(av, 2)*eb,
                                np.linalg.norm(ar, 2)*eb+ea*np.linalg.norm(bv, 2))
                    values[power] += (1. if metric else _charge_bound(space, charge))*float(bound)
        return values

    raw_bounds = coefficients(f, i)
    metric_bounds = [coefficients(s, s, metric=True) for s in (f, i)]
    g = abs(r.pilot.coupling)
    norm_bounds = [sum(g**n*v for n, v in a.items()) for a in metric_bounds]
    current_bound = sum(g**n*v for n, v in raw_bounds.items())
    # Reconstruct the reference from the independent residual solves, not a
    # reference snapshot. Tests compare against independently built snapshots.
    def reconstructed(s):
        waves = {k: {n: v+s.errors[k][n] for n, v in a.items()} for k, a in s.embedded.items()}
        metric = np.eye(len(r.pilot.matrix), dtype=complex)
        for power, pairs in terms.items():
            for a, b in pairs:
                metric += r.pilot.coupling**power*sum(waves[a][n].conj().T@waves[b][n] for n in waves[a])
        return waves, metric
    wf, nf = reconstructed(f)
    wi, ni = reconstructed(i)
    def current_coefficients(wf, wi, model):
        raw = {n: np.zeros_like(model.pilot.matrix) for n in range(5)}
        raw[0] = good_current_density_between(model.pilot, model.pilot, delta, charge=charge)
        for power, pairs in terms.items():
            for a, b in pairs:
                spaces = model.spaces6 if a == 'A6' else model.spaces5
                for sector, space in spaces.items():
                    acted = shell_density_between(space, space, wi[b][sector], delta,
                        b_GeV=model.pilot.width_GeV, charge=charge)
                    raw[power] += wf[a][sector].conj().T@acted
        return raw

    raw = current_coefficients(wf, wi, r)
    def lower_waves(s):
        return {**dict(zip(('A1', 'A2', 'C3'), s.five_wave_coefficients)),
                'A6': s.six_wave_coefficient}
    lower_raw = current_coefficients(lower_waves(f.lower), lower_waves(i.lower), i.lower.owner)
    total = sum(r.pilot.coupling**n*v for n, v in raw.items())
    eta = min(float(np.linalg.eigvalsh(n)[0]) for n in (nf, ni, f.lower.metric, i.lower.metric))
    if eta <= 0:
        raise ValueError('positive endpoint norms required; no clipping')
    normalized = current_bound/eta + np.linalg.norm(total)*sum(norm_bounds)/(2*eta**2)
    return dict(raw_coefficient_bounds=raw_bounds, metric_coefficient_bounds=metric_bounds,
                endpoint_metric_bounds=norm_bounds, raw_total_bound=current_bound,
                normalized_total_bound=float(normalized), metric_gap=eta,
                reference_raw_coefficients=raw, lower_raw_coefficients=lower_raw,
                reference_metrics=(nf, ni))
