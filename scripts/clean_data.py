"""数据清洗入口：校验 ACDC 与 KITTI 的完整性与一致性。

职责:
    - 解析 configs/data/cleaning.yaml，调用 car_smart_assist.data.preprocessing.clean
    - 把汇总结果打到控制台，明细写进 artifacts/reports/cleaning/
    - 按严重度决定退出码，便于在流水线里判定是否该阻断
    - 本身不含任何清洗逻辑（逻辑全在 preprocessing.py）

用法:
    python scripts/clean_data.py                          # 全量清洗
    python scripts/clean_data.py --only acdc              # 只洗 ACDC
    python scripts/clean_data.py --only kitti --workers 8

产物（**不修改任何原始文件**）:
    artifacts/reports/cleaning/report.json    完整明细
    artifacts/reports/cleaning/report.md      可读摘要
    data/processed/manifests/{acdc,kitti}.json  有效样本清单，供数据集类过滤

退出码:
    0  无 ERROR
    1  存在 ERROR（有样本须剔除）
    2  执行失败
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# 让脚本能直接运行而无需先 pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.data import preprocessing as pp  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="数据清洗：完整性、一致性与重复性校验")
    p.add_argument("--config", default="configs/data/cleaning.yaml", help="清洗配置文件")
    p.add_argument(
        "--only", nargs="*", choices=["acdc", "kitti"], default=None,
        help="只清洗指定数据集，默认全部",
    )
    p.add_argument("--workers", type=int, default=None, help="覆盖配置中的并行度")
    p.add_argument("--log-level", default="INFO", help="日志级别")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / args.config
    if not cfg_path.exists():
        print(f"配置文件不存在: {cfg_path}", file=sys.stderr)
        return 2

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    # NOTE: 这里直接读 yaml 是因为 config/loader.py 尚未实现。
    #       待配置加载器落地后，改为 from car_smart_assist.config import load_config，
    #       以便自动处理 base 继承与 schema 校验。
    cfg = raw.get("cleaning", raw)
    if args.workers is not None:
        cfg["num_workers"] = args.workers

    print(f"开始清洗（配置文件 {cfg_path.relative_to(root)}）...")
    report = pp.clean(cfg, project_root=root, only=args.only)

    sev = report.counts_by_severity()
    print("\n" + "=" * 62)
    print("清洗完成")
    print("=" * 62)
    print(f"  问题总数    : {len(report.issues)}")
    print(f"    ERROR     : {sev.get('error', 0)}   ← 必须剔除")
    print(f"    WARNING   : {sev.get('warning', 0)}   ← 需人工确认")
    print(f"    INFO      : {sev.get('info', 0)}   ← 仅记录")

    by_check = report.counts_by_check()
    if by_check:
        print("\n  按检查项:")
        for name, d in sorted(by_check.items()):
            badge = "!" if d.get("error") else ("~" if d.get("warning") else ".")
            print(f"    [{badge}] {name:34s} e={d.get('error', 0):<5} "
                  f"w={d.get('warning', 0):<5} i={d.get('info', 0)}")

    if report.skipped:
        print("\n  已跳过（不代表没问题）:")
        for name, reason in report.skipped.items():
            print(f"    - {name}: {reason}")

    out_dir = root / cfg.get("output", {}).get("report_dir", "artifacts/reports/cleaning")
    man_dir = root / cfg.get("output", {}).get("manifest_dir", "data/processed/manifests")
    print(f"\n  报告   : {out_dir}")
    print(f"  清单   : {man_dir}")
    print("\n  原始数据未被修改；清单是过滤依据，由数据集类消费。")

    return 1 if sev.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
