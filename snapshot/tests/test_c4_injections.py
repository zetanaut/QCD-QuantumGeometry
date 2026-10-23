"""Every mandatory C4 injected fault has a stable fail-closed diagnostic."""

import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.pilot.c4_injections import (
    INJECTIONS, detect_injected_violation,
)


@pytest.mark.parametrize("stable_id,description,diagnostic", INJECTIONS)
def test_mandatory_injection_is_detected(stable_id, description, diagnostic):
    with pytest.raises(ArchitectureError) as caught:
        detect_injected_violation(stable_id)
    assert caught.value.requirement_id == diagnostic
    assert stable_id in str(caught.value)


def test_injection_ledger_is_complete_ordered_and_unique():
    assert len(INJECTIONS) == 40
    assert [item[0] for item in INJECTIONS] == [
        f"C4.INJECT.{index:02d}" for index in range(1, 41)
    ]
    assert len({item[0] for item in INJECTIONS}) == 40
