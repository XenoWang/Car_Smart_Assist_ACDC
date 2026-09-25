"""端到端管线演示：图像 -> 能见度门控 -> 司机提示。

职责:
    - 按配置构建 InferencePipeline，在样例图上跑一遍并打印结果
    - 输出各阶段耗时与跳过原因，便于确认链路状态
    - 本身不含任何管线逻辑（全在 inference/pipeline.py）

用法:
    python scripts/run_pipeline.py                       # 每个天气子集抽一张
    python scripts/run_pipeline.py --image path/to.png   # 指定单张
    python scripts/run_pipeline.py --synthetic           # 额外跑合成退化对照
    python scripts/run_pipeline.py --json                # 输出 json

当前实现状态:
    门控（第一级判断）已实现且可用；Stage 1 感知尚未训练接入，
    因此输出里的 `skipped` 会标明感知阶段被跳过 —— 这是如实反映进度，
    不是异常。被门控判为 BLIND 的帧则连感知都不需要，直接出接管请求。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.inference.pipeline import InferencePipeline  # noqa: E402
from car_smart_assist.perception.visibility.dataset import (  # noqa: E402
    list_adverse_images,
    list_ref_images,
    read_rgb_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="端到端管线演示")
    p.add_argument("--config", default="configs/model/visibility.yaml")
    p.add_argument("--checkpoint", default=None, help="门控权重，默认取配置里的路径")
    p.add_argument("--image", action="append", default=None, help="指定图像，可重复")
    p.add_argument("--synthetic", action="store_true", help="额外跑合成退化对照")
    p.add_argument("--device", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args()


def build_samples(root: Path, cfg: dict) -> list[tuple[str, np.ndarray]]:
    """每个天气子集抽一张 + 一张正常天气参考图。"""
    acdc = root / cfg.get("data", {}).get("acdc_root", "data/raw/acdc")
    samples: list[tuple[str, np.ndarray]] = []

    refs = list_ref_images(acdc)
    if refs:
        samples.append(("正常天气参考图", read_rgb_image(refs[0])))

    for cond in ("fog", "night", "rain", "snow"):
        paths = list_adverse_images(acdc, [cond])
        if paths:
            samples.append((f"真实 {cond}", read_rgb_image(paths[0])))
    return samples


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))["visibility"]

    pipe = InferencePipeline.from_config(
        cfg, project_root=root, checkpoint=args.checkpoint, device=args.device
    )

    if args.image:
        samples = [(Path(p).name, read_rgb_image(p)) for p in args.image]
    else:
        samples = build_samples(root, cfg)

    if args.synthetic:
        from car_smart_assist.perception.visibility import degrade

        base = samples[0][1] if samples else np.full((1080, 1920, 3), 128, dtype=np.uint8)
        for kind in ("fog", "darkness", "occlusion", "blur"):
            samples.append((f"合成 {kind} @1.0", degrade(base, kind, 1.0, seed=0)[0]))

    # 全黑帧：不属于任何数据集，但是「看不见」最纯粹的形式，值得作为对照
    samples.append(("全黑帧（对照）", np.zeros((1080, 1920, 3), dtype=np.uint8)))

    results = pipe.run_batch([img for _, img in samples])

    if args.json:
        print(json.dumps(
            [{"name": n, **r.to_dict()} for (n, _), r in zip(samples, results, strict=True)],
            ensure_ascii=False, indent=2,
        ))
        return 0

    print("=" * 92)
    print(f"{'样本':<22}{'能见度':<10}{'信息量':>8}{'接管':>6}{'风险':>10}  提示")
    print("-" * 92)
    for (name, _), r in zip(samples, results, strict=True):
        v = r.visibility
        a = r.advisory
        info = f"{v.information:.3f}" if v else "n/a"
        lvl = v.level.value if v else "n/a"
        print(
            f"{name:<22}{lvl:<10}{info:>8}"
            f"{('是' if a.should_takeover else '否'):>6}{a.risk_level.value:>10}  {a.text}"
        )

    print("-" * 92)
    t = results[-1].timings_ms
    print(f"耗时: 门控 {t.get('gate', 0):.1f}ms / 帧（首次含模型预热，后续约 10ms）")
    skipped = results[0].skipped
    if skipped:
        print("\n本次运行中被跳过的阶段（如实反映实现进度，不是异常）:")
        for k, why in skipped.items():
            print(f"  - {k}: {why}")
    print("\n提示文案措辞自检（不应有越权承诺）:")
    from car_smart_assist.advisory.prompt.templates import check_wording

    bad = 0
    for r in results:
        probs = check_wording(r.advisory.text)
        if probs:
            bad += 1
            print(f"  ✗ {r.advisory.text}: {probs}")
    print("  全部通过" if not bad else f"  {bad} 条不合格")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
