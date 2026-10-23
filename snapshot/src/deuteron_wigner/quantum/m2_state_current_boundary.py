"""Fail-closed M2 K9 invariant-subspace to spin-one-current boundary.

This is a representation audit, not a current construction. The accepted M2
object is a six-dimensional, open-color q-sector projector in the K9 compact
finite basis. C47 retains fundamental q color and projects every retained qg
color factor onto its fundamental summand. Thus the complete K9 M2 space is a
direct sum of fundamental triplets, not a space containing a color-singlet
target. Repository light-front and LPS adapters instead consume spin-one
target-helicity current data. This module makes that color-intertwiner
obstruction, the remaining target-composition gap, and the missing finite-K
current operator explicit.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

import numpy as np

from deuteron_wigner.bridge.basis1 import core as c47
from deuteron_wigner.bridge.c401_c396_mass_directions import resolution_record
from deuteron_wigner.bridge.c405_c117_i2_current_topology_embedding import (
    completion_record as c405_completion_record,
)
from deuteron_wigner.bridge.c405_c117_i2_current_topology_embedding import (
    direct_sum_axis_record,
)
from deuteron_wigner.bridge.icurrent import (
    current_block_ancestry,
    current_block_completeness_decision,
)
from deuteron_wigner.bridge.modes.core import color_triplet_projector
from deuteron_wigner.lf_current import SpinOnePlusCurrent

from .m2_k9_exploratory_state import ExploratoryK9LowEigenspace


K9 = "K9"
_SPIN_ONE_HELICITIES = (+1, 0, -1)
_LF_AMPLITUDES = ("I++", "I+0", "I+-", "I00")
_PROJECTOR_Q_BLOCK_IDENTITY_TOLERANCE = 1.0e-10
_PROJECTOR_QG_LEAKAGE_TOLERANCE = 1.0e-10
_TRIPLET_ISOMETRY_TOLERANCE = 1.0e-12
_TRIPLET_PROJECTOR_IMAGE_TOLERANCE = 1.0e-12
_TRIPLET_INTERTWINING_TOLERANCE = 1.0e-12
_FUNDAMENTAL_CASIMIR_TOLERANCE = 1.0e-12


class M2StateCurrentInterfaceError(RuntimeError):
    """Raised if unavailable source physics is used as an M2 current map."""


def _require_k9_subspace(eigenspace: ExploratoryK9LowEigenspace) -> None:
    if not isinstance(eigenspace, ExploratoryK9LowEigenspace):
        raise TypeError("eigenspace must be an ExploratoryK9LowEigenspace")
    if eigenspace.assembly.bundle.resolution != K9:
        raise ValueError("the M2 state-to-current audit is defined only for K9")
    if eigenspace.multiplicity != 6 or not eigenspace.degenerate:
        raise ValueError("the accepted M2 boundary requires the sixfold degenerate K9 subspace")
    if eigenspace.assembly.point.physical or eigenspace.assembly.bundle.physical:
        raise ValueError("the M2 state-to-current boundary is exploratory and nonphysical")


def _projector_support(eigenspace: ExploratoryK9LowEigenspace) -> Mapping[str, Any]:
    """Return basis-independent support facts for the complete eigenspace."""

    record = resolution_record(K9)
    q_dimension = int(record["q_dimension"])
    labels = eigenspace.assembly.h0_supply.target_basis_labels
    projector = eigenspace.projector
    q_block = projector[:q_dimension, :q_dimension]
    qg_rows = projector[q_dimension:, :]
    q_labels = tuple(labels[:q_dimension])
    if len(q_labels) != q_dimension or any(label[0] != "q" for label in q_labels):
        raise RuntimeError("K9 target q labels no longer match the declared direct-sum basis")
    if q_dimension != eigenspace.multiplicity:
        raise RuntimeError("the accepted K9 subspace no longer closes the q coordinate block")
    q_block_identity_residual = float(np.linalg.norm(q_block - np.eye(q_dimension)))
    qg_row_frobenius_norm = float(np.linalg.norm(qg_rows))
    if q_block_identity_residual > _PROJECTOR_Q_BLOCK_IDENTITY_TOLERANCE:
        raise RuntimeError(
            "M2 K9 projector q-block identity residual "
            f"{q_block_identity_residual:.3e} exceeds "
            f"{_PROJECTOR_Q_BLOCK_IDENTITY_TOLERANCE:.1e}"
        )
    if qg_row_frobenius_norm > _PROJECTOR_QG_LEAKAGE_TOLERANCE:
        raise RuntimeError(
            "M2 K9 projector qg leakage norm "
            f"{qg_row_frobenius_norm:.3e} exceeds "
            f"{_PROJECTOR_QG_LEAKAGE_TOLERANCE:.1e}"
        )
    return {
        "projector_rank": eigenspace.multiplicity,
        "projector_trace": float(np.trace(projector).real),
        "q_block_identity_residual": q_block_identity_residual,
        "qg_row_frobenius_norm": qg_row_frobenius_norm,
        "q_labels": q_labels,
        "open_quark_helicities": tuple(sorted({int(label[1]) for label in q_labels})),
        "open_quark_Jz": tuple(sorted({float(label[1]) / 2.0 for label in q_labels})),
        "open_triplet_colors": tuple(sorted({int(label[2]) for label in q_labels})),
        "fock_sector": "q only",
        "charge_or_flavor_selected": False,
        "color_singlet_selected": False,
        "deuteron_Jz_selected": False,
        "parity_selected": False,
    }


def _color_multiplets(
    labels: tuple[tuple[object, ...], ...],
    *,
    q_dimension: int,
) -> tuple[Mapping[int, tuple[int, ...]], Mapping[tuple[object, ...], tuple[int, ...]]]:
    """Group live M2 labels into the C47 retained three-color modules."""

    q_modules: dict[int, list[int]] = {}
    qg_modules: dict[tuple[object, ...], list[int]] = {}
    for label in labels[:q_dimension]:
        if label[0] != "q":
            raise RuntimeError("the M2 q block no longer carries C47 q labels")
        q_modules.setdefault(int(label[1]), []).append(int(label[2]))
    for label in labels[q_dimension:]:
        if label[0] != "qg":
            raise RuntimeError("the M2 qg block no longer carries C47 qg labels")
        # (partition,n,m,h_q,h_g) distinguishes all noncolor qg labels.
        qg_modules.setdefault(tuple(label[1:6]), []).append(int(label[6]))
    return (
        {key: tuple(sorted(colors)) for key, colors in q_modules.items()},
        {key: tuple(sorted(colors)) for key, colors in qg_modules.items()},
    )


def _c47_color_decomposition(
    eigenspace: ExploratoryK9LowEigenspace,
) -> Mapping[str, Any]:
    """Establish C47's fundamental-only color decomposition at live K9.

    ``q_basis`` retains open fundamental colors. ``qg_basis`` stores the
    output color label of C47's ``U_{3 <- 3 x 8}=T^b/sqrt(C_F)`` isometry, so
    every noncolor qg tuple is another fundamental module. The source
    Gell--Mann action verifies the isometry covariance and the nonzero
    fundamental Casimir. Therefore the direct sum has no SU(3) invariant
    vector: an invariant would have zero quadratic Casimir, whereas every
    retained summand has ``C_2=4/3``.
    """

    basis = resolution_record(K9)
    q_dimension = int(basis["q_dimension"])
    full_resolution = str(basis["full_resolution_id"])
    c47_resolution = next(
        item for item in c47.RESOLUTIONS if item.label == full_resolution
    )
    source_q = c47.q_basis(c47_resolution)
    source_qg, _, _ = c47.qg_basis(c47_resolution)
    labels = tuple(eigenspace.assembly.h0_supply.target_basis_labels)
    q_modules, qg_modules = _color_multiplets(labels, q_dimension=q_dimension)

    source_q_modules: dict[int, list[int]] = {}
    source_qg_modules: dict[tuple[object, ...], list[int]] = {}
    for row in source_q:
        source_q_modules.setdefault(int(row[3]), []).append(int(row[4]))
    for row in source_qg:
        source_qg_modules.setdefault(
            (int(row[0]), int(row[5]), int(row[6]), int(row[9]), int(row[10])), []
        ).append(int(row[11]))
    source_q_color_sets = {
        key: tuple(sorted(colors)) for key, colors in source_q_modules.items()
    }
    source_qg_color_sets = {
        key: tuple(sorted(colors)) for key, colors in source_qg_modules.items()
    }

    triplet_isometry = c47.triplet_isometry()
    triplet_projector, product_generators, fundamental_generators = color_triplet_projector()
    identity_triplet = np.eye(3, dtype=np.complex128)
    isometry_residual = float(
        np.linalg.norm(triplet_isometry.conj().T @ triplet_isometry - identity_triplet)
    )
    image_residual = float(np.linalg.norm(triplet_projector @ triplet_isometry - triplet_isometry))
    intertwining_residual = float(
        max(
            np.linalg.norm(
                product_generators[generator] @ triplet_isometry
                - triplet_isometry @ fundamental_generators[generator]
            )
            for generator in range(8)
        )
    )
    fundamental_casimir = sum(
        generator @ generator for generator in fundamental_generators
    )
    casimir_residual = float(
        np.linalg.norm(fundamental_casimir - (4.0 / 3.0) * identity_triplet)
    )
    if isometry_residual > _TRIPLET_ISOMETRY_TOLERANCE:
        raise RuntimeError(
            "C47 triplet isometry residual "
            f"{isometry_residual:.3e} exceeds {_TRIPLET_ISOMETRY_TOLERANCE:.1e}"
        )
    if image_residual > _TRIPLET_PROJECTOR_IMAGE_TOLERANCE:
        raise RuntimeError(
            "C47 triplet-projector image residual "
            f"{image_residual:.3e} exceeds {_TRIPLET_PROJECTOR_IMAGE_TOLERANCE:.1e}"
        )
    if intertwining_residual > _TRIPLET_INTERTWINING_TOLERANCE:
        raise RuntimeError(
            "C47 triplet eight-generator intertwining residual "
            f"{intertwining_residual:.3e} exceeds {_TRIPLET_INTERTWINING_TOLERANCE:.1e}"
        )
    if casimir_residual > _FUNDAMENTAL_CASIMIR_TOLERANCE:
        raise RuntimeError(
            "C47 fundamental Casimir residual "
            f"{casimir_residual:.3e} exceeds {_FUNDAMENTAL_CASIMIR_TOLERANCE:.1e}"
        )
    all_color_sets_are_triplets = (
        set(q_modules.values()) == {(0, 1, 2)}
        and set(qg_modules.values()) == {(0, 1, 2)}
        and source_q_color_sets == q_modules
        and source_qg_color_sets == qg_modules
    )
    q_triplet_count = len(q_modules)
    qg_triplet_count = len(qg_modules)
    total_triplet_count = q_triplet_count + qg_triplet_count
    direct_sum_dimension = int(basis["direct_sum_dimension"])
    if not all_color_sets_are_triplets:
        raise RuntimeError("the live M2 color labels no longer preserve the C47 triplet modules")
    if total_triplet_count * 3 != direct_sum_dimension:
        raise RuntimeError("the live M2 color-module count no longer closes its direct-sum dimension")

    return {
        "source": "C47 q_basis open fundamental colors plus C47 U_(3<-3x8)=T^b/sqrt(C_F)",
        "q_fundamental_triplet_count": q_triplet_count,
        "qg_fundamental_triplet_count": qg_triplet_count,
        "total_fundamental_triplet_count": total_triplet_count,
        "full_space_decomposition": f"H_M2,K9 = ({total_triplet_count}) * 3",
        "direct_sum_dimension": direct_sum_dimension,
        "qg_triplet_isometry_shape": tuple(triplet_isometry.shape),
        "qg_triplet_isometry_residual": isometry_residual,
        "qg_triplet_image_residual": image_residual,
        "qg_triplet_intertwining_residual": intertwining_residual,
        "fundamental_casimir": "4/3",
        "fundamental_casimir_residual": casimir_residual,
        "color_singlet_subrepresentation_present": False,
        "singlet_intertwiner_space": "Hom_SU(3)(1, H_M2,K9) = {0}",
        "proof": (
            "every retained q/qg color module is fundamental with C2=4/3; "
            "an SU(3)-invariant vector would have C2=0"
        ),
        "M2_basis_map_preserves_C47_color_modules": True,
    }


def m2_k9_state_current_interface_audit(
    eigenspace: ExploratoryK9LowEigenspace,
) -> Mapping[str, Any]:
    """Audit the fail-closed M2 boundary to a color-singlet target current.

    A future target composition has type
    ``C_{i/f}: C^3_spin-one tensor 1_color -> H_D,K`` on an enlarged
    many-body/hadronic color-singlet space, followed by
    ``J_D,K^mu: H_D,K -> H_D,K`` and
    ``J_target^mu=C_f^dagger J_D,K^mu C_i``. It cannot land in the present
    fundamental-only ``H_M2,K9``. Any expression involving ``P_K9 J_K9 P_K9``
    is consequently only a colored-subsystem diagnostic; it is never a
    deuteron target-current matrix element.
    """

    _require_k9_subspace(eigenspace)
    basis = resolution_record(K9)
    support = _projector_support(eigenspace)
    color = _c47_color_decomposition(eigenspace)
    full_resolution = str(basis["full_resolution_id"])
    c405_axis = direct_sum_axis_record(K9)
    c405 = c405_completion_record()
    c114_ancestry = current_block_ancestry(full_resolution)
    c114 = current_block_completeness_decision()
    source_dimension = int(support["projector_rank"])
    target_dimension = len(_SPIN_ONE_HELICITIES)
    if source_dimension != 6 or target_dimension != 3:
        raise RuntimeError("this audit expects the accepted K9 six-dimensional projector and spin-one target")

    return {
        "claim_tier": "EXPLORATORY",
        "physical": False,
        "state_selection": "INVARIANT_PROJECTOR_ONLY_NO_EIGENVECTOR_SELECTED",
        "resolution": K9,
        "domain": {
            "ambient_space": f"H_M2,K9 = C^{int(basis['direct_sum_dimension'])}",
            "basis_order": "q followed by qg; qg=(partition,n,m,quark helicity,gluon helicity,open triplet color)",
            "state_object": "Ran(P_K9)",
            "state_dimension": source_dimension,
            "finite_basis_inner_product": "Euclidean compact-coordinate inner product after the C47/C401 permutation",
            "normalization_owner": "C47 x-scaled CM-ground basis and M2 permutation; not an external deuteron normalization",
            "state_coordinate_units": "dimensionless normalized finite-basis amplitudes; the M2 Hamiltonian has GeV^2 units",
            "support": support,
        },
        "required_codomain": {
            "target_spin_space": "C^3 with canonical target helicities (+1,0,-1)",
            "light_front_input": {
                "type": SpinOnePlusCurrent.__name__,
                "fields": tuple(field.name for field in fields(SpinOnePlusCurrent)),
                "amplitude_order": _LF_AMPLITUDES,
                "normalization": "I=J+/(2P+) in Drell-Yan q+=0",
            },
            "lps_input": {
                "shape": (4, 3, 3),
                "component_order": ("+", "-", "x", "y"),
                "spin_order": _SPIN_ONE_HELICITIES,
                "normalization": "unnormalized LPS free current in longitudinal Breit kinematics",
            },
            "required_external_labels": (
                "initial/final target helicity",
                "initial/final four-momentum and transfer kinematics",
                "target charge/flavor",
                "color-singlet and Fock-sector composition",
                "parity and orbital convention",
            ),
            "initial_final_status": "M2 supplies one invariant subspace P_K9, not labeled initial/final target states P_i/P_f",
        },
        "required_defining_equations": {
            "color_singlet_target_composition": "C_i,C_f: C^3_spin-one tensor 1_color -> H_D,K (not H_M2,K9)",
            "finite_K_current_after_enlargement": "J_D,K^mu: H_D,K -> H_D,K",
            "target_current_after_enlargement": "J_target^mu = C_f^dagger J_D,K^mu C_i in C^(3x3)",
            "colored_subsystem_diagnostic": (
                "(1/6) Tr[P_K9 J_K9^mu P_K9] only after J_K9^mu is source-qualified; "
                "colored-subsystem diagnostic, never a deuteron target current"
            ),
            "colored_subsystem_transition": (
                "P_f J_K9^mu P_i only after colored-subsystem initial/final projectors and transfer conventions are defined; "
                "never a deuteron target current"
            ),
        },
        "map_classification": {
            "existing_map": "NONE_SOURCE_QUALIFIED",
            "primary_obstruction": "C47_COLOR_SINGLET_INTERTWINER_ZERO",
            "color_singlet_map_into_H_M2,K9": "ZERO_BY_SU3_REPRESENTATION_CONTENT",
            "dimension_note": (
                "six versus three rules out an isomorphism only; an abstract C^3 -> C^6 embedding exists, "
                "but no nonzero SU(3)-equivariant color-singlet target embedding can land in H_M2,K9"
            ),
            "projective": "NO: a numerical three-dimensional choice would violate the proven color-singlet obstruction and select unowned combinations",
            "density_matrix_valued": "UNAVAILABLE_NOT_ZERO: P_K9/6 is basis-invariant but cannot acquire target-spin/current indices without an enlarged singlet Hilbert space and current",
        },
        "finite_K_coordinate_correspondence": {
            "C405_axis_matches": "same K9 q(6) plus qg(1344) direct-sum coordinate axis",
            "C405_to_target_spin_map": "NOT_SUPPLIED",
            "M2_to_light_front_amplitudes": "NOT_SUPPLIED",
            "M2_to_LPS_spin_matrix": "NOT_SUPPLIED",
            "transfer_kinematics": "NOT_SUPPLIED_NOT_ZERO",
        },
        "representation_obstruction": {
            "primary": "Hom_SU(3)(1, H_M2,K9) = {0}",
            "color_decomposition": color,
            "dimension_note": (source_dimension, target_dimension),
            "m2_open_Jz": support["open_quark_Jz"],
            "required_target_helicities": _SPIN_ONE_HELICITIES,
            "missing_target_helicity_zero": 0.0 not in support["open_quark_Jz"],
            "open_color_not_color_singlet": support["open_triplet_colors"],
            "missing_information_not_zero": (
                "spin-1 target composition/intertwiner",
                "color-singlet projection",
                "nucleon/proton-neutron and charge-flavor composition",
                "orbital/parity assembly",
                "initial/final transfer-kinematics map",
                "external-state normalization and physical mass",
            ),
        },
        "current_operator_obstruction": {
            "C405_same_direct_sum_axis": {
                "dimension_matches_M2": int(c405_axis["direct_sum_dimension"])
                == int(basis["direct_sum_dimension"]),
                "q_diagonal_block": c405_axis["q_diagonal_block_status"],
                "qg_diagonal_block": c405_axis["qg_diagonal_block_status"],
                "cross_sector_blocks": c405_axis["cross_sector_blocks"],
            },
            "C405_complete_C117_action": bool(c405["full_C117_I2_action_ready"]),
            "C405_missing_object": c405["smallest_missing_object"],
            "C114_complete_instantaneous_current": bool(c114["complete_block"]),
            "C114_missing_products": tuple(c114["unavailable_products"]),
            "C114_basis_dimensions": dict(c114_ancestry["basis_dimensions"]),
            "interpretation": "C405/C114 are C117 instantaneous-current topology/projection structures, not a completed external electromagnetic spin-one current",
        },
        "factor_ownership": {
            "M2_kinetic": "C47 x-scaled q_rel^2 convention with M2 sparse recurrence; no C128 fractions/free matrix",
            "M2_color_content": "C47 owns open q fundamental colors and the qg 3<-3x8 triplet isometry; no singlet resides in the retained M2 direct sum",
            "mass_directions": "C401/C396, separate from H0",
            "interaction": "C411 exploratory C117 action with unresolved normalization/mixing explicit",
            "finite_K_current": "C114/C115/C405 current factors and finite-HO projection remain incomplete, not zero",
            "target_current_normalization": "light-front/LPS adapter convention is defined, but no M2-to-target composition owns its conversion",
        },
        "separate_historical_current": {
            "source": "microscopic/h1/current.py",
            "domain": "C8 H1 qqq valence basis only, with its own H1 Hamiltonian identity",
            "tower_dimensions": (4, 7, 10),
            "M2_basis_map": "NOT_DECLARED; C7/C8 dimensions are not assumptions for M2",
            "usable_for_M2": False,
        },
        "adapter_use": "NOT_STARTED_NO_LAWFUL_INPUT_OBJECT",
        "current_response": "NOT_EVALUATED_UNAVAILABLE_NOT_ZERO",
        "next_executable_construction": (
            "first introduce or bind an enlarged many-body/hadronic finite-K Hilbert space H_D,K with a color-singlet "
            "spin-one deuteron composition, including charge/flavor, Fock, orbital/parity, normalization, and transfer labels; "
            "only then derive finite-K current intertwiners J_D,K^mu rather than treating the incomplete C405/C114 blocks as zero"
        ),
    }


def require_lawful_m2_k9_current_interface(
    eigenspace: ExploratoryK9LowEigenspace,
) -> None:
    """Fail closed instead of fabricating a spin-one current from ``Ran(P_K9)``."""

    audit = m2_k9_state_current_interface_audit(eigenspace)
    raise M2StateCurrentInterfaceError(
        "No lawful M2 K9-to-spin-one current map exists: C47 proves "
        "Hom_SU(3)(1, H_M2,K9) = {0}, so a nonzero color-singlet target composition cannot land "
        "inside the present M2 space. An enlarged color-singlet spin-one Hilbert space and its finite-K "
        "current intertwiners are unavailable, not zero."
    )


__all__ = [
    "M2StateCurrentInterfaceError",
    "m2_k9_state_current_interface_audit",
    "require_lawful_m2_k9_current_interface",
]
