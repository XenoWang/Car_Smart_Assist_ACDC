"""保留300轮训练状态，追加600轮；完成后统一验证及选择默认模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_weather_attention import main as train_main  # noqa: E402
from car_smart_assist.utils.stop import STOP_EXIT_CODE, StopController  # noqa: E402

from car_smart_assist.config.weather_attention import load_weather_attention_config  # noqa: E402
from car_smart_assist.perception.weather_extension import finish_extension  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/weather_attention_swad_extend_600.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument(
        "--fresh", action="store_true", help="从固定300轮来源重开追加阶段，先保存本阶段历史"
    )
    modes.add_argument("--stop", action="store_true", help="只写停止请求")
    modes.add_argument("--continue", dest="continue_training", action="store_true", help="从本阶段 last 继续原目标")
    parser.add_argument("--stop-file", type=Path, default=None)
    args = parser.parse_args(argv)
    path = ROOT / args.config
    cfg = load_weather_attention_config(path)
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))["weather_extension"]
    if args.fresh:
        route = yaml.safe_load(
            (ROOT / "configs/model/weather_default.yaml").read_text(encoding="utf-8")
        )["weather_default"]
        if (ROOT / route["checkpoint"]).resolve().parent == (ROOT / cfg.checkpoint_root).resolve():
            raise ValueError("当前默认模型引用本阶段输出；重开实验须先使用新的独立输出目录")
    with StopController(args.stop_file, default_directory=ROOT / cfg.checkpoint_root) as controller:
        if args.stop:
            controller.request_stop_file()
            print(f"已请求轮末停止：{controller.path}")
            return 0
        if args.fresh or args.continue_training:
            controller.resume()
        controller.require_running()
        arguments = ["--config", args.config, "--stop-file", str(controller.path)]
        if args.preflight:
            arguments.append("--preflight")
        if args.fresh:
            arguments.append("--fresh")
        if args.continue_training:
            arguments.append("--continue")
        folder = ROOT / cfg.report_dir
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(folder / "extension.lock", timeout=0):
                code = train_main(arguments, stop_controller=controller)
                if code or args.preflight:
                    return code
                with FileLock(ROOT / cfg.checkpoint_root / "run.lock", timeout=0):
                    controller.require_running()
                    verdict = finish_extension(cfg, ROOT, path, policy, stop_controller=controller)
                    controller.require_running()
                print(json.dumps(verdict, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"追加阶段未完成收尾：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
