"""端到端推理管线：图像 -> 门控 -> 感知 -> 司机提示。

职责:
    - 串起 门控 -> Stage1 感知 -> postprocess -> Stage2 生成
    - 管理模型常驻与批处理，避免每帧重复加载
    - 记录每帧耗时（各阶段分开计），供性能说明用
    - 提供 CLI 与最小 demo 接口，方便录演示视频

管线结构（门控是第一级判断，这决定了整体形状）:
    ┌─────────────┐
    │ 输入图像     │
    └──────┬──────┘
           ▼
    ┌─────────────────────────────┐
    │ ① 能见度门控（已实现）        │  ← 降采样到 144×256 跑
    │   VISIBLE / DEGRADED / BLIND │
    └──────┬──────────────────────┘
           │
      ┌────┴─────┐
   BLIND      DEGRADED / VISIBLE
      │            │
      │            ▼
      │     ┌─────────────────────┐
      │     │ ② 感知识别（待实现）  │  ← 用**原始分辨率**
      │     │   路况/接管边界/距离  │
      │     └──────┬──────────────┘
      │            ▼
      │     ┌─────────────────────┐
      │     │ ③ 建议生成（部分实现）│
      │     └──────┬──────────────┘
      │            │
      └────────────┴──► 给司机的提示
      （BLIND 直接出接管请求，不跑 ②③）

    两条设计约束:
    1. **门控必须能独立工作**。它在 ②③ 未实现、模型文件缺失、
       甚至没有 GPU 的情况下都要能给出结论 —— 它是整套系统的安全网，
       不能依赖任何别的模块才生效。
    2. **两个阶段用不同分辨率**。门控跑在 144×256 的降采样图上（能见度是
       全局属性，不需要细节，且 AE 就是按这个尺寸训练的）；而感知必须用
       **原始分辨率** —— 检测框和单目测距都依赖原始像素尺度与相机内参，
       用降采样图会让距离估计整体偏掉。混用这两个输入是本项目最容易犯的错。

当前实现状态:
    ① 已实现且已验证（召回/误报/单调性见 artifacts/reports/visibility/）
    ② 未实现 —— Stage 1 多任务模型尚未训练
    ③ 已实现 —— 门控接管与结构化感知规则路径可用（advisory/generator）

    因此现在跑管线会得到「路况未知」类的提示而不是真实识别结果。
    这是如实反映实现进度，不是 bug —— 管线会把这些阶段记进 `skipped`。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from car_smart_assist.advisory.generator import AdvisoryGenerator
from car_smart_assist.advisory.schema import AdvisoryResult, PerceptionResult
from car_smart_assist.config.visibility import MODEL_DEFAULTS
from car_smart_assist.config.weather import load_weather_config
from car_smart_assist.perception.visibility.dataset import read_rgb_image
from car_smart_assist.perception.visibility.gate import (
    VisibilityGate,
    VisibilityLevel,
    VisibilityVerdict,
)
from car_smart_assist.perception.visibility.scorer import VisibilityScorer
from car_smart_assist.perception.weather import WeatherPrediction, WeatherPredictor

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """一帧的完整管线输出。"""

    visibility: VisibilityVerdict | None = None
    perception: PerceptionResult | None = None
    advisory: AdvisoryResult | None = None
    weather: WeatherPrediction | None = None
    # 各阶段耗时（毫秒）。分开计是因为三者的优化手段完全不同，
    # 只给一个总耗时说明不了任何问题。
    timings_ms: dict[str, float] = field(default_factory=dict)
    # 被跳过的阶段及原因。空字典表示全链路都跑了。
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        """管线是否因能见度不足而阻断了感知。"""
        return self.visibility is not None and self.visibility.level is VisibilityLevel.BLIND

    def to_dict(self) -> dict[str, Any]:
        return {
            "visibility": self.visibility.to_dict() if self.visibility else None,
            "weather": self.weather.to_dict() if self.weather else None,
            "perception": self.perception.to_dict() if self.perception else None,
            "advisory": self.advisory.to_dict() if self.advisory else None,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "skipped": dict(self.skipped),
            "blocked": self.blocked,
        }


def _to_array(image: np.ndarray | Image.Image | str | Path) -> np.ndarray:
    """把各种输入统一成 (H, W, 3) uint8 数组。"""
    if isinstance(image, np.ndarray):
        a = image
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        if a.shape[-1] == 4:
            a = a[..., :3]
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(a)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    return read_rgb_image(image)


class InferencePipeline:
    """端到端推理管线。

    Args:
        scorer: 能见度打分器。为 None 则跳过门控（**不推荐**，见 from_config）
        gate: 判定器
        generator: Stage 2 建议生成器
        predictor: Stage 1 感知模型封装。为 None 表示感知阶段尚未接入，
            管线会把它记进 skipped 而不是假装跑过
        gate_input_size: 门控的输入尺寸，必须与 AE 训练时一致
        cfg: 完整配置，供各阶段读取自己的参数
    """

    def __init__(
        self,
        scorer: VisibilityScorer | None = None,
        gate: VisibilityGate | None = None,
        generator: AdvisoryGenerator | None = None,
        predictor: Any | None = None,
        gate_input_size: tuple[int, int] = MODEL_DEFAULTS["input_size"],
        cfg: dict[str, Any] | None = None,
        weather_predictor: WeatherPredictor | None = None,
    ) -> None:
        self.scorer = scorer
        self.gate = gate or VisibilityGate()
        self.generator = generator or AdvisoryGenerator()
        self.predictor = predictor
        self.weather_predictor = weather_predictor
        self.gate_input_size = tuple(gate_input_size)
        self.cfg = cfg or {}

        if self.scorer is None:
            logger.warning(
                "管线未接入能见度门控 —— 这一帧不会经过任何能见度判断。安全网失效，仅应用于调试。"
            )

    # --- 构建 ---

    @classmethod
    def from_config(
        cls,
        visibility_cfg: dict[str, Any],
        project_root: str | Path = ".",
        checkpoint: str | Path | None = None,
        device: str | None = None,
        advisory_cfg: dict[str, Any] | None = None,
        predictor: Any | None = None,
        require_visibility: bool = True,
        weather_checkpoint: str | Path | None = None,
    ) -> InferencePipeline:
        """按配置构建管线。

        Args:
            require_visibility: 为 True（默认）时，门控 checkpoint 不存在就报错。
                改成 False 会得到一条**没有安全网**的管线 —— 只在调试时用，
                因为「看不见却不出声」正是这套系统最不能接受的失败模式。
        """
        root = Path(project_root)
        tcfg = visibility_cfg.get("train", {})
        model_cfg = visibility_cfg.get("model", {})
        size = tuple(model_cfg.get("input_size", MODEL_DEFAULTS["input_size"]))

        ckpt = (
            Path(checkpoint)
            if checkpoint is not None
            else root / tcfg.get("checkpoint_dir", "artifacts/checkpoints/visibility") / "best.pt"
        )

        scorer: VisibilityScorer | None = None
        if ckpt.exists():
            from car_smart_assist.perception.visibility.trainer import resolve_device

            scorer = VisibilityScorer.from_checkpoint(
                ckpt,
                device=resolve_device(device or str(visibility_cfg.get("device", "auto"))),
                cfg=model_cfg,
                scoring_cfg=visibility_cfg.get("scoring"),
            )
            size = scorer.input_size
            logger.info("能见度门控已接入: %s", ckpt)
        elif require_visibility:
            raise FileNotFoundError(
                f"能见度门控 checkpoint 不存在: {ckpt}\n"
                "这套管线把门控当作安全网，缺了它就不是完整的系统。\n"
                "  先训练: python scripts/train_visibility.py\n"
                "  或显式关闭: InferencePipeline.from_config(..., require_visibility=False)\n"
                "  后者会得到一条没有能见度判断的管线，只应用于调试。"
            )
        else:
            logger.warning("未找到门控 checkpoint（%s），管线将不带能见度判断运行", ckpt)

        weather_predictor = None
        weather_config_path = root / "configs/model/weather_classifier.yaml"
        if weather_config_path.is_file():
            weather_cfg = load_weather_config(weather_config_path)
            weather_ckpt = (
                Path(weather_checkpoint) if weather_checkpoint else Path(weather_cfg.checkpoint)
            )
            if not weather_ckpt.is_absolute():
                weather_ckpt = root / weather_ckpt
            if weather_ckpt.is_file():
                weather_predictor = WeatherPredictor.from_checkpoint(weather_ckpt, weather_cfg)
                logger.info("天气小模型已接入: %s", weather_ckpt)
        elif weather_checkpoint is not None:
            raise FileNotFoundError(f"天气模型配置不存在: {weather_config_path}")

        return cls(
            scorer=scorer,
            gate=VisibilityGate.from_config(visibility_cfg),
            generator=AdvisoryGenerator(advisory_cfg),
            predictor=predictor,
            weather_predictor=weather_predictor,
            gate_input_size=size,
            cfg={"visibility": visibility_cfg, "advisory": advisory_cfg or {}},
        )

    # --- 内部阶段 ---

    def _resize_for_gate(self, img: np.ndarray) -> np.ndarray:
        """缩放到门控输入尺寸。

        用 BILINEAR 而非 LANCZOS：与训练时的缓存构建保持一致。
        两者不一致会让输入的空间频率分布出现系统性差异，
        而高频能量正是判「糊没糊」的关键量 —— 这类不一致不会报错，
        只会让分数整体漂移。
        """
        h, w = self.gate_input_size
        if img.shape[0] == h and img.shape[1] == w:
            return img
        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR), dtype=np.uint8)

    def _run_gate(self, images: Sequence[np.ndarray]) -> list[VisibilityVerdict]:
        if self.scorer is None:
            return []
        small = [self._resize_for_gate(im) for im in images]
        scores = self.scorer.score_arrays(small)
        if len(scores) != len(images):
            raise RuntimeError(
                f"门控评分数量不匹配：输入 {len(images)} 帧，得到 {len(scores)} 个分数"
            )
        return self.gate.judge_many(scores)

    # --- 主流程 ---

    def run(self, image: np.ndarray | Image.Image | str | Path) -> PipelineResult:
        """单帧推理。"""
        return self.run_batch([image])[0]

    def run_batch(
        self, images: Sequence[np.ndarray | Image.Image | str | Path]
    ) -> list[PipelineResult]:
        """批量推理。门控阶段真正批量执行；感知阶段按 predictor 的接口走。"""
        if len(images) == 0:
            return []
        arrays = [_to_array(im) for im in images]
        results = [PipelineResult() for _ in arrays]

        # --- ① 门控 ---
        t0 = time.perf_counter()
        try:
            verdicts = self._run_gate(arrays)
        except Exception as exc:  # noqa: BLE001
            # 门控失败不能被解释成「可见」；记录故障并在建议阶段请求接管。
            logger.exception("能见度门控失败")
            verdicts = []
            for r in results:
                r.skipped["gate"] = f"门控运行失败: {type(exc).__name__}: {exc}"
        gate_ms = (time.perf_counter() - t0) * 1000.0
        if verdicts:
            for r, v in zip(results, verdicts, strict=True):
                r.visibility = v
                r.timings_ms["gate"] = gate_ms / max(len(arrays), 1)
        if self.scorer is None:
            for r in results:
                r.skipped["gate"] = "未接入能见度门控（scorer 为 None）"
        elif not verdicts:
            for r in results:
                r.skipped.setdefault("gate", "门控未产出结果")

        # --- ② 感知（仅对未被阻断的帧）---
        for i, r in enumerate(results):
            if self.scorer is not None and "gate" in r.skipped:
                r.skipped["perception"] = "能见度门控失败，已跳过感知"
                continue
            if r.blocked:
                r.skipped["perception"] = "能见度判定为 BLIND，感知结果不可信，已跳过"
                r.skipped["advisory_perception_path"] = "同上"
                continue

            if self.weather_predictor is not None:
                weather_started = time.perf_counter()
                try:
                    weather = self.weather_predictor.predict(arrays[i])
                    if not isinstance(weather, WeatherPrediction):
                        raise TypeError("weather_predictor.predict() 必须返回 WeatherPrediction")
                    r.weather = weather
                except Exception as exc:  # noqa: BLE001
                    logger.exception("天气识别失败")
                    r.skipped["weather"] = f"天气识别异常: {type(exc).__name__}: {exc}"
                r.timings_ms["weather"] = (time.perf_counter() - weather_started) * 1000.0

            if self.predictor is None:
                r.skipped["perception"] = (
                    "Stage 1 感知模型尚未接入（perception/models/multitask.py 待训练）"
                )
                if r.weather is not None:
                    r.perception = PerceptionResult(
                        visibility_level=r.visibility.level.value if r.visibility else "unknown",
                        visibility_confidence_multiplier=(
                            r.visibility.confidence_multiplier if r.visibility else 1.0
                        ),
                        visibility_reasons=list(r.visibility.triggered) if r.visibility else [],
                        road_condition=r.weather.condition,
                        road_condition_confidence=r.weather.confidence,
                    )
                continue

            t1 = time.perf_counter()
            try:
                # 注意：传**原始分辨率**的图，不是门控用的降采样图。
                # 检测框与单目测距都依赖原始像素尺度。
                perception = self.predictor.predict(arrays[i])
                if not isinstance(perception, PerceptionResult):
                    raise TypeError(
                        "predictor.predict() 必须返回 PerceptionResult，"
                        f"实际为 {type(perception).__name__}"
                    )
                if r.visibility is not None:
                    perception.visibility_level = r.visibility.level.value
                    perception.visibility_confidence_multiplier = r.visibility.confidence_multiplier
                    perception.visibility_reasons = list(r.visibility.triggered)
                if r.weather is not None and r.weather.accepted:
                    if perception.road_condition is None:
                        perception.road_condition = r.weather.condition
                        perception.road_condition_confidence = r.weather.confidence
                    elif perception.road_condition != r.weather.condition:
                        perception.road_condition = None
                        perception.road_condition_confidence = 0.0
                        r.skipped["weather_fusion"] = "两个识别器的路况类别冲突，改为无法判断"
                r.perception = perception
            except Exception as exc:  # noqa: BLE001
                # 感知失败不能让整条管线失声 —— 记下来，后面走降级文案
                logger.exception("感知阶段失败")
                r.skipped["perception"] = f"感知阶段异常: {type(exc).__name__}: {exc}"
            r.timings_ms["perception"] = (time.perf_counter() - t1) * 1000.0

        # --- ③ 建议 ---
        t2 = time.perf_counter()
        for r in results:
            if r.blocked:
                r.advisory = self.generator.generate(visibility=r.visibility)
            elif "gate" in r.skipped:
                r.advisory = self.generator.from_unavailable(r.skipped["gate"])
            elif r.perception is None and "perception" in r.skipped:
                r.advisory = self.generator.from_unavailable(r.skipped["perception"])
            else:
                r.advisory = self.generator.generate(
                    visibility=r.visibility, perception=r.perception
                )
        advisory_ms = (time.perf_counter() - t2) * 1000.0
        for r in results:
            r.timings_ms["advisory"] = advisory_ms / max(len(results), 1)
            r.timings_ms["total"] = sum(r.timings_ms.values())

        return results
