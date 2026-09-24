"""端到端管线与数据契约的单元测试。

职责:
    - 锁住门控作为**第一级判断**的行为：BLIND 必须阻断感知并直接出接管请求
    - 锁住数据契约：文本不可为空、未知距离用 None 而非 0、置信度统一降级
    - 锁住两个阶段用**不同分辨率**（门控降采样，感知用原图）
    - 全部用桩对象，不需要真实模型或数据集

这些测试守的是安全属性而非数值精度：
「看不见却不出声」和「把看不见的帧当成看得见继续跑感知」
是这套系统最不能接受的两种失败，两条都要测。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from car_smart_assist.advisory.generator import AdvisoryGenerator
from car_smart_assist.advisory.prompt.templates import (
    FALLBACK_TEXT,
    MAX_TEXT_CHARS,
    VISIBILITY_TEXT,
    check_wording,
    render_visibility,
)
from car_smart_assist.advisory.schema import (
    AdvisoryResult,
    PerceptionResult,
    RiskLevel,
    TargetDirection,
    TargetObject,
)
from car_smart_assist.inference.pipeline import InferencePipeline, PipelineResult
from car_smart_assist.perception.visibility.gate import (
    GateThresholds,
    VisibilityGate,
    VisibilityLevel,
)
from car_smart_assist.perception.visibility.scorer import VisibilityScore

# ---------------------------------------------------------------------------
# 桩对象
# ---------------------------------------------------------------------------


class StubScorer:
    """返回预设打分的打分器桩。记录收到的图像尺寸，用于验证分辨率。"""

    def __init__(self, information: float = 0.9, recon_z: float = 0.0) -> None:
        self.information = information
        self.recon_z = recon_z
        self.seen_shapes: list[tuple[int, int]] = []

    def score_arrays(self, images, paths=None):
        from car_smart_assist.perception.visibility.scorer import InformationFeatures

        out = []
        for i, im in enumerate(images):
            self.seen_shapes.append(im.shape[:2])
            out.append(
                VisibilityScore(
                    path=f"<stub:{i}>",
                    recon_mean=0.01,
                    recon_p90_block=0.01,
                    recon_z=self.recon_z,
                    features=InformationFeatures(1.0, 1.0, 1.0, 1.0),
                    information=self.information,
                )
            )
        return out


class ShortScorer(StubScorer):
    """模拟评分器漏返回某些帧的分数。"""

    def score_arrays(self, images, paths=None):
        return super().score_arrays(images, paths)[:-1]


class BrokenScorer(StubScorer):
    def score_arrays(self, images, paths=None):
        raise RuntimeError("synthetic scorer failure")


class StubPredictor:
    """记录收到的图像尺寸，用于验证感知阶段拿到的是原图。"""

    def __init__(self) -> None:
        self.seen_shapes: list[tuple[int, int]] = []

    def predict(self, image: np.ndarray) -> PerceptionResult:
        self.seen_shapes.append(image.shape[:2])
        return PerceptionResult(visibility_level="visible", road_condition="fog")


def make_pipeline(information: float = 0.9, predictor=None, **kw) -> InferencePipeline:
    return InferencePipeline(
        scorer=StubScorer(information=information),
        gate=VisibilityGate(GateThresholds(), require_calibration=False),
        generator=AdvisoryGenerator(),
        predictor=predictor,
        gate_input_size=(144, 256),
        **kw,
    )


def frame(h: int = 720, w: int = 1280) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 门控作为第一级判断
# ---------------------------------------------------------------------------


class TestGateBlocksPerception:
    """BLIND 必须阻断感知 —— 这是管线最重要的行为。"""

    def test_blind_skips_perception(self):
        predictor = StubPredictor()
        r = make_pipeline(information=0.01, predictor=predictor).run(frame())
        assert r.blocked is True
        assert r.perception is None
        assert predictor.seen_shapes == [], "BLIND 时不该调用感知模型"
        assert "perception" in r.skipped

    def test_blind_emits_takeover_request(self):
        r = make_pipeline(information=0.01).run(frame())
        a = r.advisory
        assert a.should_takeover is True
        assert a.risk_level is RiskLevel.CRITICAL
        assert a.source == "visibility_gate"
        assert a.text  # 绝不能为空

    def test_blind_evidence_is_traceable(self):
        r = make_pipeline(information=0.01).run(frame())
        joined = " ".join(r.advisory.evidence)
        assert "能见度判定" in joined
        assert "信息量" in joined

    def test_visible_runs_perception(self):
        predictor = StubPredictor()
        r = make_pipeline(information=0.95, predictor=predictor).run(frame())
        assert r.blocked is False
        assert r.perception is not None
        assert len(predictor.seen_shapes) == 1

    def test_degraded_still_runs_perception(self):
        """DEGRADED 只是降低置信度，不该阻断 —— 误报会让用户学会忽略它。"""
        predictor = StubPredictor()
        r = make_pipeline(information=0.40, predictor=predictor).run(frame())
        assert r.blocked is False
        assert r.perception is not None
        assert r.perception.visibility_confidence_multiplier == pytest.approx(0.6)
        assert r.perception.visibility_level == "degraded"

    def test_missing_predictor_recorded_not_faked(self):
        """感知未接入时要记进 skipped，而不是假装跑过。"""
        r = make_pipeline(information=0.95).run(frame())
        assert r.perception is None
        assert "perception" in r.skipped
        assert "尚未接入" in r.skipped["perception"]
        assert r.advisory.should_takeover is True
        assert r.advisory.risk_level is RiskLevel.CRITICAL

    def test_scorer_failure_emits_takeover_fallback(self):
        predictor = StubPredictor()
        pipe = InferencePipeline(
            scorer=BrokenScorer(),
            gate=VisibilityGate(GateThresholds(), require_calibration=False),
            predictor=predictor,
        )
        result = pipe.run(frame())
        assert "gate" in result.skipped
        assert "synthetic scorer failure" in result.skipped["gate"]
        assert predictor.seen_shapes == []
        assert result.perception is None
        assert result.advisory.should_takeover is True
        assert result.advisory.source == "fallback"

    def test_short_batch_scores_fail_closed_for_every_frame(self):
        predictor = StubPredictor()
        pipe = InferencePipeline(
            scorer=ShortScorer(),
            gate=VisibilityGate(GateThresholds(), require_calibration=False),
            predictor=predictor,
        )
        results = pipe.run_batch([frame(), frame()])
        assert all("gate" in result.skipped for result in results)
        assert all(result.advisory.should_takeover for result in results)
        assert predictor.seen_shapes == []


# ---------------------------------------------------------------------------
# 分辨率
# ---------------------------------------------------------------------------


class TestResolutionSeparation:
    """门控用降采样图，感知用原图 —— 混用会让距离估计整体偏掉。"""

    def test_gate_receives_downsampled(self):
        scorer = StubScorer(information=0.95)
        pipe = InferencePipeline(
            scorer=scorer,
            gate=VisibilityGate(GateThresholds(), require_calibration=False),
            gate_input_size=(144, 256),
        )
        pipe.run(frame(1080, 1920))
        assert scorer.seen_shapes == [(144, 256)]

    def test_predictor_receives_original(self):
        predictor = StubPredictor()
        pipe = make_pipeline(information=0.95, predictor=predictor)
        pipe.run(frame(1080, 1920))
        assert predictor.seen_shapes == [(1080, 1920)], "感知必须拿到原始分辨率"

    def test_both_stages_in_one_run(self):
        scorer, predictor = StubScorer(information=0.95), StubPredictor()
        pipe = InferencePipeline(
            scorer=scorer,
            gate=VisibilityGate(GateThresholds(), require_calibration=False),
            predictor=predictor,
            gate_input_size=(144, 256),
        )
        pipe.run(frame(1080, 1920))
        assert scorer.seen_shapes == [(144, 256)]
        assert predictor.seen_shapes == [(1080, 1920)]

    def test_already_correct_size_not_resized(self):
        scorer = StubScorer(information=0.95)
        pipe = InferencePipeline(
            scorer=scorer, gate=VisibilityGate(GateThresholds(), require_calibration=False),
            gate_input_size=(144, 256),
        )
        pipe.run(frame(144, 256))
        assert scorer.seen_shapes == [(144, 256)]


# ---------------------------------------------------------------------------
# 输入格式
# ---------------------------------------------------------------------------


class TestInputHandling:
    @pytest.mark.parametrize("h,w", [(720, 1280), (144, 256)])
    def test_accepts_ndarray(self, h, w):
        assert make_pipeline().run(frame(h, w)).advisory is not None

    def test_accepts_pil(self):
        assert make_pipeline().run(Image.fromarray(frame())).advisory is not None

    def test_accepts_path(self, tmp_path: Path):
        p = tmp_path / "a.png"
        Image.fromarray(frame()).save(p)
        assert make_pipeline().run(p).advisory is not None
        assert make_pipeline().run(str(p)).advisory is not None

    def test_grayscale_expanded_to_rgb(self):
        gray = np.random.default_rng(0).integers(0, 256, (720, 1280), dtype=np.uint8)
        assert make_pipeline().run(gray).advisory is not None

    def test_rgba_drops_alpha(self):
        rgba = np.random.default_rng(0).integers(0, 256, (720, 1280, 4), dtype=np.uint8)
        assert make_pipeline().run(rgba).advisory is not None

    def test_float_input_clipped(self):
        f = np.random.default_rng(0).random((720, 1280, 3)) * 255.0
        assert make_pipeline().run(f).advisory is not None

    def test_batch_preserves_order_and_count(self):
        pipe = make_pipeline(information=0.95)
        results = pipe.run_batch([frame(720, 1280) for _ in range(3)])
        assert len(results) == 3
        assert all(isinstance(r, PipelineResult) for r in results)


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


class TestSchemaContract:
    def test_advisory_text_cannot_be_empty(self):
        """空文本等于静默失声，必须被拦在最外层。"""
        with pytest.raises(ValueError, match="不能为空"):
            AdvisoryResult(text="")

    def test_unknown_distance_is_none_not_zero(self):
        o = TargetObject(category="car")
        assert o.distance_m is None
        assert o.direction is TargetDirection.UNKNOWN

    def test_confidence_multiplier_applied(self):
        p = PerceptionResult(visibility_confidence_multiplier=0.6)
        assert p.effective_confidence(1.0) == pytest.approx(0.6)

    def test_blind_multiplier_is_zero(self):
        p = PerceptionResult(visibility_confidence_multiplier=0.0)
        assert p.effective_confidence(0.9) == 0.0

    def test_nearest_leading_ignores_unknown_distance(self):
        p = PerceptionResult(objects=[
            TargetObject("car", TargetDirection.LEADING, None),
            TargetObject("car", TargetDirection.LEADING, 30.0),
            TargetObject("car", TargetDirection.LEADING, 45.0),
        ])
        assert p.nearest_leading.distance_m == 30.0

    def test_nearest_oncoming_separate_from_leading(self):
        p = PerceptionResult(objects=[
            TargetObject("car", TargetDirection.LEADING, 10.0),
            TargetObject("car", TargetDirection.ONCOMING, 80.0),
        ])
        assert p.nearest_leading.distance_m == 10.0
        assert p.nearest_oncoming.distance_m == 80.0

    def test_direction_unknown_never_counted_as_leading(self):
        """方向未知不能默认成前车 —— 两者驾驶建议完全相反。"""
        p = PerceptionResult(objects=[TargetObject("car", TargetDirection.UNKNOWN, 5.0)])
        assert p.nearest_leading is None
        assert p.nearest_oncoming is None

    def test_results_are_json_serializable(self):
        import json

        r = make_pipeline(information=0.95).run(frame())
        json.dumps(r.to_dict())  # 不抛异常即可

    def test_perception_result_serializes_enums(self):
        import json

        p = PerceptionResult(objects=[TargetObject("car", TargetDirection.ONCOMING, 20.0)])
        d = p.to_dict()
        assert d["objects"][0]["direction"] == "oncoming"
        json.dumps(d)


# ---------------------------------------------------------------------------
# 建议生成
# ---------------------------------------------------------------------------


class TestAdvisoryGenerator:
    def test_fallback_when_nothing_available(self):
        a = AdvisoryGenerator().generate()
        assert a.text == FALLBACK_TEXT
        assert a.source == "fallback"
        assert a.text  # 不静默失声
        assert a.should_takeover is True
        assert a.risk_level is RiskLevel.CRITICAL

    def test_perception_path_not_implemented_but_does_not_silence(self):
        """感知路径未实现时必须明确，而不是返回假数据或空。"""
        g = AdvisoryGenerator()
        p = PerceptionResult(road_condition="fog")
        assert hasattr(g, "from_perception")
        with pytest.raises(NotImplementedError, match="尚未实现"):
            g.from_perception(p)
        # 但统一入口不能因此失声
        a = g.generate(perception=p)
        assert a.text
        assert a.should_takeover is True

    def test_blind_takes_priority_over_perception(self):
        from car_smart_assist.perception.visibility.gate import VisibilityVerdict

        g = AdvisoryGenerator()
        v = VisibilityVerdict(level=VisibilityLevel.BLIND, reason="x", information=0.01)
        a = g.generate(visibility=v, perception=PerceptionResult(road_condition="fog"))
        assert a.should_takeover is True
        assert a.source == "visibility_gate"


# ---------------------------------------------------------------------------
# 文案约束
# ---------------------------------------------------------------------------


class TestWordingConstraints:
    def test_all_visibility_texts_pass_own_check(self):
        for level in VisibilityLevel:
            text, _ = render_visibility(level)
            assert check_wording(text) == [], f"{level} 的文案不合格: {text}"

    def test_visible_means_image_visibility_not_safe_road(self):
        text, risk = render_visibility(VisibilityLevel.VISIBLE)
        assert "路况正常" not in text
        assert "观察路况" in text
        assert risk is RiskLevel.NONE

    def test_blind_text_contains_action(self):
        """必须含明确动作，不能只有描述。"""
        text = VISIBILITY_TEXT[VisibilityLevel.BLIND]
        assert "接管" in text

    def test_banned_words_detected(self):
        assert check_wording("绝对安全，无需注意")
        assert not check_wording("能见度下降，请注意")

    def test_overlong_text_detected(self):
        assert check_wording("很" * (MAX_TEXT_CHARS + 1))

    def test_unknown_level_falls_back_to_conservative_text(self):
        text, risk = render_visibility("banana")  # type: ignore[arg-type]
        assert text
        assert risk is RiskLevel.NOTICE


# ---------------------------------------------------------------------------
# 构建与配置
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_missing_checkpoint_raises_by_default(self, tmp_path: Path):
        """门控是安全网，缺了它必须响亮地失败，而不是静默降级。"""
        cfg = {
            "data": {"acdc_root": "data/raw/acdc"},
            "train": {"checkpoint_dir": str(tmp_path / "nope")},
            "model": {"input_size": [144, 256]},
            "device": "cpu",
        }
        with pytest.raises(FileNotFoundError, match="门控 checkpoint 不存在"):
            InferencePipeline.from_config(cfg, project_root=tmp_path)

    def test_error_message_points_at_the_fix(self, tmp_path: Path):
        cfg = {
            "data": {}, "train": {"checkpoint_dir": str(tmp_path / "nope")},
            "model": {"input_size": [144, 256]}, "device": "cpu",
        }
        with pytest.raises(FileNotFoundError) as ei:
            InferencePipeline.from_config(cfg, project_root=tmp_path)
        msg = str(ei.value)
        assert "train_visibility" in msg
        assert "require_visibility=False" in msg

    def test_can_opt_out_explicitly(self, tmp_path: Path):
        cfg = {
            "data": {}, "train": {"checkpoint_dir": str(tmp_path / "nope")},
            "model": {"input_size": [144, 256]}, "device": "cpu",
        }
        pipe = InferencePipeline.from_config(
            cfg, project_root=tmp_path, require_visibility=False
        )
        assert pipe.scorer is None
        # 没有门控时仍要能出文案，且记录跳过原因
        r = pipe.run(frame())
        assert r.advisory.text
        assert "gate" in r.skipped


class TestPipelineResult:
    def test_timings_recorded_per_stage(self):
        r = make_pipeline().run(frame())
        assert "gate" in r.timings_ms
        assert "advisory" in r.timings_ms
        assert "total" in r.timings_ms

    def test_blocked_property(self):
        assert make_pipeline(information=0.01).run(frame()).blocked is True
        assert make_pipeline(information=0.95).run(frame()).blocked is False
