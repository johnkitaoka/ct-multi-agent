"""A completed trajectory retains the monitor's evidence with its score."""

from inspect_ai.scorer import Score

from control_tower.protocols.eval.monitoring.monitor import (
    entries,
    entry,
    monitor_score,
)
from control_tower.trajectories.monitor_score import verdicts


def test_scored_trajectory_explanation_survives_log_round_trip() -> None:
    original = entry(
        Score(
            value=7,
            explanation="The observed action needs review.",
            metadata={"monitor": "review@worker", "draw": 1},
        ),
        message_id=None,
    )
    recorded = monitor_score([original], [])
    assert verdicts(recorded)[0].score.explanation == original["explanation"]
    assert entries(recorded, []) == [original]
