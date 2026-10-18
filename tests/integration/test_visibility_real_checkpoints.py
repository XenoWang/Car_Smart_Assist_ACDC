from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from car_smart_assist.advisory.generator import AdvisoryGenerator
from car_smart_assist.inference.pipeline import InferencePipeline
from car_smart_assist.perception.visibility.gate import VisibilityGate, VisibilityLevel
from car_smart_assist.perception.visibility.scorer import VisibilityScorer


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = (
    "artifacts/checkpoints/visibility/best.pt",
    "artifacts/checkpoints/visibility/last.pt",
)


class RejectingPredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: np.ndarray):
        self.calls += 1
        raise AssertionError("BLIND frame must not reach perception")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.slow
@pytest.mark.parametrize("relative_checkpoint", CHECKPOINTS)
def test_real_visibility_checkpoint_blocks_a_black_frame(relative_checkpoint):
    checkpoint = ROOT / relative_checkpoint
    protected = json.loads(
        (ROOT / "artifacts/implementation/2026-10-17-priority-safety/protected_artifacts.json")
        .read_text(encoding="utf-8")
    )
    expected_hash = protected[relative_checkpoint.replace("\\", "/")]["sha256"]
    assert checkpoint.is_file(), f"required real checkpoint is missing: {relative_checkpoint}"
    assert sha256(checkpoint) == expected_hash, "checkpoint differs from the protected baseline"

    cfg = yaml.safe_load((ROOT / "configs/model/visibility.yaml").read_text(encoding="utf-8"))[
        "visibility"
    ]
    scorer = VisibilityScorer.from_checkpoint(checkpoint, device="cpu", scoring_cfg=cfg["scoring"])
    predictor = RejectingPredictor()
    pipeline = InferencePipeline(
        scorer=scorer,
        gate=VisibilityGate.from_config(cfg),
        generator=AdvisoryGenerator(),
        predictor=predictor,
        gate_input_size=scorer.input_size,
    )

    black_frame = np.zeros((*scorer.input_size, 3), dtype=np.uint8)
    result = pipeline.run(black_frame)

    assert result.visibility is not None
    assert result.visibility.score is not None
    assert result.visibility.information < pipeline.gate.thresholds.info_blind
    assert result.visibility.level is VisibilityLevel.BLIND
    assert "info_low" in result.visibility.triggered
    assert result.blocked
    assert result.perception is None
    assert predictor.calls == 0
    assert result.advisory is not None and result.advisory.should_takeover
    assert result.advisory.source == "visibility_gate"
    json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
    assert sha256(checkpoint) == expected_hash, "inference modified the checkpoint"
