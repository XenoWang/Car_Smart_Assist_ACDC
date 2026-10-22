"端到端管线演示：图像 -> 能见度门控 -> 司机提示。"

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.inference.pipeline import InferencePipeline, _to_array  # noqa: E402
from car_smart_assist.perception.visibility.dataset import (  # noqa: E402
    list_adverse_images,
    list_ref_images,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="端到端管线演示")
    p.add_argument("--config", default="configs/model/visibility.yaml")
    p.add_argument("--checkpoint", default=None, help="门控权重，默认取配置里的路径")
    p.add_argument("--image", action="append", default=None, help="指定图像，可重复")
    p.add_argument("--synthetic", action="store_true", help="额外跑合成退化对照")
    p.add_argument("--device", default=None)
    p.add_argument("--weather-config", default=None, help="新天气模型配置；省略时沿用当前默认天气模型")
    p.add_argument("--weather-loss", choices=("focal", "asl"), default="focal", help="用于金字塔天气候选；注意力候选固定使用 ASL")
    p.add_argument("--weather-checkpoint", default=None, help="指定天气权重，默认从所选配置读取")
    p.add_argument("--json", action="store_true")
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args()


def build_samples(root: Path, cfg: dict) -> list[tuple[str, np.ndarray | Path]]:
    """每个天气子集抽一张 + 一张正常天气参考图。"""
    dcfg = cfg.get("data", {})
    acdc = root / dcfg.get("acdc_root", "data/raw/acdc")
    manifest = dcfg.get("cleaning_manifest")
    samples: list[tuple[str, np.ndarray | Path]] = []

    refs = list_ref_images(acdc, cleaning_manifest=manifest, project_root=root)
    if refs:
        samples.append(("正常天气参考图", refs[0]))

    for cond in ("fog", "night", "rain", "snow"):
        paths = list_adverse_images(
            acdc, [cond], cleaning_manifest=manifest, project_root=root
        )
        if paths:
            samples.append((f"真实 {cond}", paths[0]))
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
        cfg, project_root=root, checkpoint=args.checkpoint, device=args.device,
        weather_config=args.weather_config, weather_loss=args.weather_loss,
        weather_checkpoint=args.weather_checkpoint,
    )

    if args.image:
        samples = [(Path(p).name, Path(p)) for p in args.image]
    else:
        samples = build_samples(root, cfg)

    if args.synthetic:
        from car_smart_assist.perception.visibility import degrade

        base = None
        for name, image in samples:
            try:
                base = _to_array(image)
            except Exception as exc:  # noqa: BLE001
                logging.warning("合成对照底图不可用（%s）: %s: %s", name, type(exc).__name__, exc)
                continue
            break
        if base is None:
            logging.warning("没有可解码的样本，已跳过合成退化对照")
        else:
            for kind in ("fog", "darkness", "occlusion", "blur"):
                samples.append((f"合成 {kind} @1.0", degrade(base, kind, 1.0, seed=0)[0]))

    # 全黑帧：不属于任何数据集，但它是「看不见」最直接的形式，值得做对照
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
        if "input" in r.skipped:
            print(f"  {r.skipped['input']}")
        if r.weather_warning:
            print(f"  天气提醒：{r.weather_warning}")

    print("-" * 92)
    t = results[-1].timings_ms
    gate_ms = t.get("gate")
    if gate_ms is not None:
        print(f"末帧门控内部诊断: {gate_ms:.1f}ms（批内均摊，非端到端性能量测）")
    skipped = results[0].skipped
    if skipped:
        print("\n本次运行中被跳过的阶段（如实反映实现进度，不是异常）:")
        for k, why in skipped.items():
            print(f"  - {k}: {why}")
    print("\n提示文案措辞自检（不应有越权承诺）:")
    from car_smart_assist.advisory.prompt.templates import check_wording

    bad = 0
    for r in results:
        texts = [r.advisory.text]
        if r.weather_warning:
            texts.append(r.weather_warning)
        for text in texts:
            probs = check_wording(text)
            if probs:
                bad += 1
                print(f"  ✗ {text}: {probs}")
    print("  全部通过" if not bad else f"  {bad} 条不合格")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
