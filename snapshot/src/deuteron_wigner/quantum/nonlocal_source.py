"""Exact Abelian probe insertions on a prescribed spacetime Wilson path.

The color transporter can multiply these Abelian factors; it is not replaced
by a scalar trace. These are nonlocal kernel vertices, not a completed bound
nucleon electromagnetic current. Coordinates use GeV^-1 and momenta GeV.
"""
import numpy as np

ETA = np.diag([1., -1., -1., -1.])


def _path(points):
    x = np.asarray(points, float)
    if x.ndim != 2 or x.shape[1] != 4 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite Cartesian spacetime points required.')
    return x


def _momentum(value):
    q = np.asarray(value, float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('Finite Cartesian probe momentum required.')
    return q


def line_integral(points, momentum):
    """I^mu(q)=integral dz^mu exp(i q.z), exact on each straight segment.

    sinc evaluates the q.segment=0 limit without a singular division.
    q_mu I^mu=(exp(i q.end)-exp(i q.start))/i for any piecewise linear path.
    """
    x,q = _path(points),_momentum(momentum)
    displacement, middle = np.diff(x,axis=0), (x[1:]+x[:-1])/2
    phase = np.exp(1j*(middle@ETA@q))*np.sinc((displacement@ETA@q)/(2*np.pi))
    return np.sum(displacement*phase[:,None],axis=0)


def vertex(points, momentum, charge):
    """d exp(i Q integral A.dz)/d A_mu(q) at A=0; contravariant mu."""
    if not np.isfinite(charge):
        raise ValueError('Finite explicit charge required.')
    return 1j*charge*line_integral(points,momentum)


def second_vertex(points, first_momentum, second_momentum, charge):
    """Two commuting EM insertions, including the seagull/contact partner."""
    return np.outer(vertex(points,first_momentum,charge),vertex(points,second_momentum,charge))


def endpoint_difference(points, momentum, charge):
    x,q = _path(points),_momentum(momentum)
    return charge*(np.exp(1j*x[-1]@ETA@q)-np.exp(1j*x[0]@ETA@q))


def wilson_probe(points, momenta, polarizations, amplitudes, charge):
    """Finite Abelian plane-wave source; complex Fourier amplitudes allowed.

    The source couples A_mu I^mu; polarizations here are contravariant so the
    metric is explicit. For a real spacetime field include conjugate modes.
    """
    if not (len(momenta)==len(polarizations)==len(amplitudes)):
        raise ValueError('One polarization and amplitude per source momentum required.')
    exponent = 0j
    for q,e,a in zip(momenta,polarizations,amplitudes):
        e = np.asarray(e,complex)
        if e.shape != (4,) or not np.isfinite(e).all() or not np.isfinite(a):
            raise ValueError('Finite source polarization/amplitude required.')
        exponent += a*(e@ETA@line_integral(points,q))
    return np.exp(1j*charge*exponent)
