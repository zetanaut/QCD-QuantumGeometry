"""Complete raw cubic/direct-quartic H2 source images in one field prescription."""
from dataclasses import dataclass
from functools import partial
import numpy as np

from .m2_gaussian_field_regulator import GaussianFieldRegulator, GaussianClusterRegulator
from .m2_gaussian_field_vertices import cubic_field_orbital, pair_field_orbital, direct_field_orbital
from .m2_source_vectors import SourceImage, validate_source_vectors
from .m2_excluded_fock_action import build_excluded_source
from .m2_instantaneous_pair_exit import _build_pair_source, pair_mass_kernel, adjoint_pair_mass_kernel
from .m2_instantaneous_sea_exit import build_instantaneous_sea_source
from .m2_six_body_quartic import COMPONENTS, build_direct_source


@dataclass(frozen=True)
class FieldRegulatedSourceImage(SourceImage):
    field_regulator: GaussianFieldRegulator | GaussianClusterRegulator
    total_pplus_GeV: float
    total_transverse_GeV: tuple

    physical = False
    complete_regulated_H2_matching = False


def apply_field_source(pilot, vectors, *, out_cutoff, kind, regulator,
                       total_pplus_GeV, total_transverse_GeV):
    """Exact participating-field source action; no spectator form factor.

    This labels actual regulated sources explicitly. It does not combine
    them with bare QQ/U actions, propagators or tree currents by assumption.
    """
    if not isinstance(regulator, (GaussianFieldRegulator, GaussianClusterRegulator)):
        raise ValueError("explicit Gaussian field prescription required")
    pp, pt = float(total_pplus_GeV), np.asarray(total_transverse_GeV, float)
    if not np.isfinite(pp) or pp <= 0 or pt.shape != (2,) or not np.all(np.isfinite(pt)):
        raise ValueError("explicit finite total transverse momentum and positive p+ required")
    if isinstance(regulator, GaussianFieldRegulator) and not np.allclose(
            pt, pp*np.asarray(regulator.reference_transverse_velocity), atol=2e-14, rtol=2e-14):
        raise ValueError("fixed-Jz field-leg source needs the comoving reference slice; off-frame orbitals require additional Jz blocks")
    args = dict(regulator=regulator, total_pplus_GeV=pp, total_transverse_GeV=tuple(pt))
    v = validate_source_vectors(pilot, vectors)
    if kind in ("qqqgg", "qqqq-qbar"):
        image = build_excluded_source(pilot, kind, out_cutoff=out_cutoff, _retained_vectors=v,
                                      _orbital_kernel=partial(cubic_field_orbital, **args))
    elif kind in ("IF", "IG", "combined"):
        kernels = {"IF":pair_mass_kernel, "IG":adjoint_pair_mass_kernel,
                   "combined":lambda *x: pair_mass_kernel(*x)+adjoint_pair_mass_kernel(*x)}
        image = _build_pair_source(pilot, out_cutoff, kernels[kind], kind, _retained_vectors=v,
            _orbital_kernel=partial(pair_field_orbital, sector="qqqgg", **args))
    elif kind == "sea":
        image = build_instantaneous_sea_source(pilot, out_cutoff=out_cutoff, _retained_vectors=v,
            _orbital_kernel=partial(pair_field_orbital, **args))
    elif kind in COMPONENTS:
        image = build_direct_source(pilot, out_cutoff=out_cutoff, component=kind, _retained_vectors=v,
            _orbital_kernel=partial(direct_field_orbital, **args))
    else:
        raise ValueError("explicit cubic, IF/IG/combined, sea or one of seven direct-Q6 kinds required")
    return FieldRegulatedSourceImage(image.retained_pilot, image.space, image.coupling_power,
        image.values, image.components, regulator, pp, tuple(pt))
