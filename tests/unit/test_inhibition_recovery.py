from __future__ import annotations

from brainmem.competition_inhibition import InhibitionPolicy, recover_inhibited_strengths


def test_recover_inhibited_strengths_restores_strength_and_reduces_inhibition() -> None:
    table = {
        "ev-a": {
            "strength": 0.2,
            "inhibition_level": 0.4,
            "last_inhibited_at": "2026-04-01T00:00:00+00:00",
        },
        "ev-b": {
            "strength": 0.6,
            "inhibition_level": 0.0,
        },
    }
    recovered = recover_inhibited_strengths(
        strength_table=table,
        elapsed_days=2.0,
        policy=InhibitionPolicy(recovery_rate_per_day=0.1, rebound_strength_factor=0.5),
    )
    assert "ev-a" in recovered
    assert table["ev-a"]["strength"] > 0.2
    assert table["ev-a"]["inhibition_level"] < 0.4
    assert "ev-b" not in recovered
