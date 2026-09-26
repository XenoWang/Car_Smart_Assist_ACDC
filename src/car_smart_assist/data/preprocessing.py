"""数据清洗：对 ACDC、KITTI 与 Pixel Accurate Benchmark 做完整性检查。

职责:
    - 图像完整性：能否解码、是否截断、尺寸是否异常、是否退化（纯色/极低方差）
    - ACDC 配对：每张图与它的 5 个标注变体是否配套；掩码是否有效（非全 ignore、类别在范围内）
    - 检测标注：bbox 是否越界/零面积、是否引用了不存在的图像、是否有孤儿标注
    - KITTI 三元组：image / label / calib 是否齐全，每帧内参是否可解析
    - Pixel Accurate Benchmark：直接检查子 ZIP 中图片能否完整解码
    - 重复检测：精确重复（内容哈希）与近重复（dHash 汉明距离）
    - 统计离群（可选）：抓「能正常解码但统计特征异常」的图，这是完整性检查抓不到的
    - 产出清洗报告与「有效样本清单」，供数据集类过滤

设计原则（重要，改这个文件前先读）:
    1. **绝不修改原始数据。**
       清洗的产物是清单与报告，不是被删掉的文件。原始数据保持只读。
       理由：raw 层是不可再生资产 —— ACDC 是审批制获取的，误删的代价
       不是「重新下载一次」能弥补的。对训练而言，按清单过滤与物理删除等价。
    2. **只有「数据不可用」才排除；「图像不寻常」一律不排除。**（最重要的一条）
       ERROR   （→ 进 invalid 清单，被数据集类过滤）
           · 图像无法解码 / 截断
           · 必需的配套文件缺失（ACDC 掩码、KITTI 的 calib）
           · 结构不一致（掩码尺寸 ≠ 图像尺寸、类别 ID 越界、标注引用不存在的图像）
           判据是「这条数据**无法**用于训练」。
       WARNING （→ 进 suspect 清单，仍然参与训练，仅提示复核）
           · 灰度方差偏低、尺寸偏小、宽高比异常、疑似重复、统计离群、bbox 过小
           判据是「这张图**看起来**不寻常」，决定权必须留给人。
       INFO    记录备查，不影响使用（如 KITTI 尺寸天然不统一）

       ⚠️ 这条区分的必要性：ACDC 的核心内容就是大雾、夜路、暴雨、雪天。
       浓雾画面接近均匀灰白、夜路画面整体偏暗，都会压低灰度方差 ——
       如果按方差阈值自动排除，会**系统性删掉最该被学会的那部分数据**，
       而且报告上只会显示「排除 N 张退化图」，看起来像个正常结论，没人会察觉。
       实测全量 23011 张图：ACDC 最低灰度方差 12.96、KITTI 46.61，
       与默认阈值 2.0 相距甚远 —— 但这是这两份数据碰巧没有「糊成一片」的帧，
       换数据集/加自采数据后不成立。所以默认按 WARNING 处理，不赌运气。

       内容类检查的严重度可在 cleaning.yaml 的 image_statistics.severity 里改，
       但改之前请先看报告里的实际分布并人工抽查 —— 不要直接调阈值硬排除。
    3. 阈值一律从 configs/data/cleaning.yaml 读，代码里不出现魔数。
    4. 报告里必须同时写明「检查了什么」和「跳过了什么」，
       避免「跑过了 = 数据干净」的错觉。跳过的项要给出跳过原因。

依赖: Pillow, numpy（均为已有依赖）。统计离群项额外需要 scikit-learn。
      RLE 解码校验需要 pycocotools —— 未安装时该项自动跳过并记 INFO，不报错。

被谁调用: scripts/prepare_acdc.py, car_smart_assist.cli.main
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ACDC 官方标注的 5 种变体后缀，顺序即失败时的排查优先级
# （labelIds 是训练用的，其余是可视化或辅助用的）
ACDC_MASK_SUFFIXES: tuple[str, ...] = (
    "labelIds",        # 训练用：Cityscapes 类别 ID
    "labelTrainIds",   # 训练用：Cityscapes trainID
    "labelColor",      # 仅可视化
    "invIds",          # 无效区域掩码，无效=1
    "invGray",         # 无效区域掩码，无效=255，仅可视化
)

# 有标注的 split。test 的标注官方不公开；*_ref 的标注在单独的 zip 里
# （gt_trainval_ref.zip），当前未下载 —— 这不是数据损坏，是范围选择。
ACDC_ANNOTATED_SPLITS: frozenset[str] = frozenset({"train", "val"})


class Severity(str, Enum):
    """问题严重度。决定该项是否影响样本可用性。"""

    ERROR = "error"      # 必须剔除
    WARNING = "warning"  # 可疑，需人工确认
    INFO = "info"        # 仅记录


class SampleStatus(str, Enum):
    """单个样本的清洗结论。"""

    VALID = "valid"
    SUSPECT = "suspect"    # 有 WARNING，可用但需留意
    INVALID = "invalid"    # 有 ERROR，不可用


# severity 字符串 -> 枚举。配置里用字符串写，代码里用枚举判断。
_SEVERITY_BY_NAME: dict[str, Severity] = {
    "error": Severity.ERROR,
    "warning": Severity.WARNING,
    "info": Severity.INFO,
}


def _severity(cfg: dict[str, Any], key: str, default: Severity) -> Severity:
    """从配置读严重度，非法值回退到 default。"""
    return _SEVERITY_BY_NAME.get(str(cfg.get(key, default.value)).lower(), default)


# =============================================================================
# 报告容器
# =============================================================================


@dataclass(frozen=True)
class Issue:
    """一条清洗问题。"""

    check: str
    severity: Severity
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "path": self.path,
            "message": self.message,
        }


@dataclass
class CleaningReport:
    """清洗结果的汇总容器。

    累积 issue、统计量与「检查了什么/跳过了什么」，最后落盘为 json + markdown。
    """

    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    checked: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    _status: dict[str, SampleStatus] = field(default_factory=dict, repr=False)

    # --- 记录 ---

    def add(self, check: str, severity: Severity, path: str | Path, message: str) -> None:
        """记录一条问题，并同步更新该样本的状态。"""
        p = str(path)
        self.issues.append(Issue(check, severity, p, message))
        cur = self._status.get(p, SampleStatus.VALID)
        if severity is Severity.ERROR:
            self._status[p] = SampleStatus.INVALID
        elif severity is Severity.WARNING and cur is SampleStatus.VALID:
            self._status[p] = SampleStatus.SUSPECT

    def error(self, check: str, path: str | Path, message: str) -> None:
        self.add(check, Severity.ERROR, path, message)

    def warn(self, check: str, path: str | Path, message: str) -> None:
        self.add(check, Severity.WARNING, path, message)

    def info(self, check: str, path: str | Path, message: str) -> None:
        self.add(check, Severity.INFO, path, message)

    def mark_checked(self, name: str, n: int) -> None:
        self.checked[name] = n

    def mark_skipped(self, name: str, reason: str) -> None:
        self.skipped[name] = reason
        logger.info("跳过检查项 %s：%s", name, reason)

    # --- 查询 ---

    def status_of(self, path: str | Path) -> SampleStatus:
        return self._status.get(str(path), SampleStatus.VALID)

    def invalid_paths(self) -> set[str]:
        return {p for p, s in self._status.items() if s is SampleStatus.INVALID}

    def counts_by_severity(self) -> dict[str, int]:
        c = Counter(i.severity.value for i in self.issues)
        return {s.value: c.get(s.value, 0) for s in Severity}

    def counts_by_check(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in self.issues:
            out[i.check][i.severity.value] += 1
        return {k: dict(v) for k, v in out.items()}

    # --- 落盘 ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_issues": len(self.issues),
                "by_severity": self.counts_by_severity(),
                "by_check": self.counts_by_check(),
                "samples": {
                    "valid": sum(1 for s in self._status.values() if s is SampleStatus.VALID),
                    "suspect": sum(1 for s in self._status.values() if s is SampleStatus.SUSPECT),
                    "invalid": sum(1 for s in self._status.values() if s is SampleStatus.INVALID),
                },
            },
            "checked": self.checked,
            "skipped": self.skipped,
            "stats": self.stats,
            "issues": [i.to_dict() for i in self.issues],
        }

    def write(self, out_dir: str | Path, max_samples_per_issue: int = 50) -> Path:
        """写出 report.json 与 report.md，返回 json 路径。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "report.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        lines: list[str] = ["# 数据清洗报告", ""]
        lines += ["## 汇总", "", "| 严重度 | 数量 |", "|--------|------|"]
        for sev, n in self.counts_by_severity().items():
            lines.append(f"| {sev} | {n} |")
        lines += ["", "## 按检查项", "", "| 检查项 | error | warning | info |", "|--------|-------|---------|------|"]
        for check, d in sorted(self.counts_by_check().items()):
            lines.append(
                f"| {check} | {d.get('error', 0)} | {d.get('warning', 0)} | {d.get('info', 0)} |"
            )

        lines += ["", "## 已检查", "", "| 项目 | 数量 |", "|------|------|"]
        for k, v in self.checked.items():
            lines.append(f"| {k} | {v} |")

        if self.skipped:
            lines += [
                "",
                "## 已跳过（未能检查，不代表没问题）",
                "",
                "| 项目 | 原因 |",
                "|------|------|",
            ]
            for k, v in self.skipped.items():
                lines.append(f"| {k} | {v} |")

        if self.stats:
            lines += ["", "## 统计", "", "```json",
                      json.dumps(self.stats, ensure_ascii=False, indent=2), "```"]

        lines += ["", "## 问题明细", ""]
        grouped: dict[tuple[str, str], list[Issue]] = defaultdict(list)
        for i in self.issues:
            grouped[(i.check, i.severity.value)].append(i)
        for (check, sev), items in sorted(grouped.items()):
            lines.append(f"### {check} / {sev}（共 {len(items)}）")
            lines.append("")
            for i in items[:max_samples_per_issue]:
                lines.append(f"- `{i.path}` — {i.message}")
            if len(items) > max_samples_per_issue:
                lines.append(f"- …另有 {len(items) - max_samples_per_issue} 条，见 report.json")
            lines.append("")

        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        logger.info("清洗报告已写出: %s", out)
        return out / "report.json"


# =============================================================================
# 图像探测（多进程 worker，必须是模块级函数以便 pickle）
# =============================================================================


def _probe_image(task: tuple[str, bool, bool, int]) -> dict[str, Any]:
    """探测单张图像的可读性与统计特征。

    Args:
        task: (路径, 是否强制解码像素, 是否算统计量, 灰度降采样系数)

    Returns:
        含 ok / error / size / mode / 统计量的 dict。
        这个函数在子进程里运行，**不能抛异常** —— 所有异常转成 error 字段。
    """
    path, force_load, want_stats, downsample = task
    r: dict[str, Any] = {
        "path": path, "ok": False, "error": None,
        "width": None, "height": None, "mode": None,
        "brightness_mean": None, "brightness_std": None,
        "entropy": None, "edge_density": None, "colorfulness": None,
    }
    try:
        # verify() 只查文件结构，不做像素解码，能抓到大部分截断文件
        with Image.open(path) as im:
            im.verify()
        # verify() 之后必须重开，否则文件句柄状态不可用
        with Image.open(path) as im:
            if force_load:
                im.load()
            r["width"], r["height"] = im.size
            r["mode"] = im.mode
            if want_stats:
                r.update(_image_stats(im, downsample))
        r["ok"] = True
    except Exception as exc:  # noqa: BLE001 —— 子进程里必须吞掉一切异常
        r["error"] = f"{type(exc).__name__}: {exc}"
    return r


def _image_stats(im: Image.Image, downsample: int) -> dict[str, float]:
    """计算用于退化与离群检测的图像统计量。

    全部在灰度降采样图上算，兼顾速度与稳定性。
    返回值范围固定，便于跨数据集比较。
    """
    small = im.convert("L")
    if downsample > 1:
        small = small.resize(
            (max(1, small.width // downsample), max(1, small.height // downsample)),
            Image.BILINEAR,
        )
    a = np.asarray(small, dtype=np.float32)

    # 直方图熵：退化图（纯色）熵接近 0
    hist, _ = np.histogram(a, bins=256, range=(0, 255))
    p = hist.astype(np.float64) / max(1, hist.sum())
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    # 边缘密度：退化或模糊图接近 0
    gy, gx = np.gradient(a)
    mag = np.hypot(gx, gy)
    edge_density = float((mag > 10.0).mean())

    # 色彩丰富度（Hasler-Süsstrunk 简化式）：灰度化后失去的色彩信息量
    rgb = np.asarray(
        im.convert("RGB").resize((max(1, im.width // downsample), max(1, im.height // downsample)))
        if downsample > 1 else im.convert("RGB"),
        dtype=np.float32,
    )
    rr, gg, bb = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = rr - gg, 0.5 * (rr + gg) - bb
    colorfulness = float(math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    return {
        "brightness_mean": float(a.mean()),
        "brightness_std": float(a.std()),
        "entropy": entropy,
        "edge_density": edge_density,
        "colorfulness": colorfulness,
    }


def _probe_batch(tasks: Sequence[tuple]) -> list[dict[str, Any]]:
    """批量探测，供 ProcessPoolExecutor.map 使用（模块级以便 pickle）。"""
    return [_probe_image(t) for t in tasks]


def _run_probes(
    paths: Sequence[Path],
    report: CleaningReport,
    *,
    check: str,
    force_load: bool,
    want_stats: bool,
    downsample: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    """并行探测一批图像，把完整性/尺寸/退化问题直接记入 report。

    Returns:
        全部探测结果（含正常的），供后续统计与离群检测复用，避免重复 IO。
    """
    tasks = [(str(p), force_load, want_stats, downsample) for p in paths]
    results: list[dict[str, Any]] = []

    if num_workers > 1 and len(tasks) > 64:
        # 分块提交，避免一次性 pickl 几万个 tuple
        chunk = max(64, len(tasks) // (num_workers * 4))
        chunks = [tasks[i : i + chunk] for i in range(0, len(tasks), chunk)]
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for out in ex.map(_probe_batch, chunks):
                results.extend(out)
    else:
        results.extend(_probe_batch(tasks))

    for r in results:
        if not r["ok"]:
            report.error(check, r["path"], f"无法解码: {r['error']}")
    return results


def check_zip_image_integrity(
    archive_paths: Sequence[Path],
    report: CleaningReport,
    *,
    check: str,
    project_root: str | Path,
) -> list[str]:
    """完整解码 ZIP 内图片；原归档不解包、不修改。"""
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    checked_paths: list[str] = []
    checked = 0
    invalid = 0
    base = Path(project_root).resolve()

    for archive_path in sorted(archive_paths):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    if info.is_dir() or Path(info.filename).suffix.lower() not in image_extensions:
                        continue
                    member_path = f"{archive_path.resolve().relative_to(base)}::{info.filename}"
                    checked_paths.append(member_path)
                    checked += 1
                    try:
                        image_bytes = archive.read(info)
                        with Image.open(io.BytesIO(image_bytes)) as image:
                            image.verify()
                        with Image.open(io.BytesIO(image_bytes)) as image:
                            image.load()
                    except Exception as exc:  # noqa: BLE001 — 一个损坏成员不能中断整包扫描
                        invalid += 1
                        report.error(
                            check,
                            member_path,
                            f"无法完整解码: {type(exc).__name__}: {exc}",
                        )
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            report.error(check, archive_path, f"无法读取图像归档: {type(exc).__name__}: {exc}")

    report.mark_checked(f"{check.rsplit('/', 1)[0]}.images_probed", checked)
    report.stats[f"{check.rsplit('/', 1)[0]}_total_images"] = checked
    report.stats[f"{check.rsplit('/', 1)[0]}_corrupt_images"] = invalid
    return checked_paths


# =============================================================================
# 尺寸与退化检查
# =============================================================================


def _check_dimensions_and_degeneracy(
    results: Sequence[dict[str, Any]],
    report: CleaningReport,
    cfg: dict[str, Any],
    *,
    check: str,
    label: str,
) -> None:
    """基于探测结果检查尺寸异常与退化图，并把尺寸分布记入 stats。

    ⚠️ 本函数的三项检查**全部是内容判断**，默认严重度都是 WARNING，
    即默认**不会**把任何样本标为 INVALID。

    原因：这些指标衡量的是「图像看起来是否寻常」，不是「数据是否可用」。
    大雾天画面接近均匀灰白、夜路画面整体偏暗，都会压低灰度标准差 ——
    而这两种场景恰恰是 ACDC 的核心内容，把它们当作「退化图」剔除会
    直接破坏数据集的分布，让模型在恶劣天气上的评估失去意义。

    实测参考（全量 23011 张）：ACDC 最低灰度标准差 12.96、KITTI 46.61，
    与默认阈值 2.0 相距甚远，但那是这两份数据恰好没有「糊成一片」的帧，
    换数据集后不一定成立。因此这里把决定权交给配置与人，而不是默认排除。

    需要更激进的策略时，在 cleaning.yaml 的 image_statistics.severity 里
    把对应项改成 error —— 但改之前请先看报告里的实际分布，并人工抽查被标出的图。
    """
    min_side = cfg.get("min_side_px", 64)
    lo, hi = cfg.get("aspect_ratio_range", [0.5, 6.0])
    min_std = cfg.get("min_pixel_std", 2.0)

    sev_cfg = cfg.get("severity", {})
    sev_min_side = _severity(sev_cfg, "min_side", Severity.WARNING)
    sev_aspect = _severity(sev_cfg, "aspect_ratio", Severity.WARNING)
    sev_low_std = _severity(sev_cfg, "low_std", Severity.WARNING)

    size_dist: Counter[tuple[int, int]] = Counter()
    n_small = n_aspect = n_flat = 0
    for r in results:
        if not r["ok"]:
            continue
        w, h = r["width"], r["height"]
        size_dist[(w, h)] += 1

        if min(w, h) < min_side:
            n_small += 1
            report.add(
                check, sev_min_side, r["path"],
                f"尺寸过小: {w}x{h}（下限 {min_side}）。"
                "注意：尺寸小不等于损坏，需人工确认是否为原始采集缺陷",
            )

        ar = w / h if h else 0.0
        if not (lo <= ar <= hi):
            n_aspect += 1
            report.add(check, sev_aspect, r["path"], f"宽高比异常: {ar:.2f}（允许 {lo}–{hi}）")

        std = r.get("brightness_std")
        if std is not None and std < min_std:
            n_flat += 1
            report.add(
                check, sev_low_std, r["path"],
                f"灰度标准差偏低: {std:.3f} < {min_std}。"
                "⚠️ 低方差**不等于**损坏 —— 浓雾、纯雪地、极暗夜路都会如此。"
                "默认只告警不排除；若要排除请确认人工看过该图",
            )

    report.stats[f"{label}_low_variance_candidates"] = n_flat
    report.stats[f"{label}_small_image_candidates"] = n_small
    report.stats[f"{label}_odd_aspect_candidates"] = n_aspect
    if n_flat:
        logger.warning(
            "%s: %d 张图灰度方差偏低。这些**未必**是坏图（浓雾/雪地/夜路均会如此），"
            "当前按 WARNING 处理、不排除。建议人工抽查后再决定。",
            label, n_flat,
        )

    top = size_dist.most_common(8)
    report.stats[f"{label}_size_distribution"] = {
        f"{w}x{h}": n for (w, h), n in top
    }
    report.stats[f"{label}_distinct_sizes"] = len(size_dist)
    if len(size_dist) > 1:
        # 尺寸不统一本身不一定是缺陷（KITTI 天然如此），记 INFO 供人判断
        report.info(
            check,
            label,
            f"存在 {len(size_dist)} 种图像尺寸: "
            + ", ".join(f"{w}x{h}({n})" for (w, h), n in top[:4]),
        )


# =============================================================================
# ACDC 检查
# =============================================================================


def _acdc_mask_path(image_path: Path, acdc_root: Path, suffix: str) -> Path:
    """由 rgb_anon 图像路径推出对应的 gt 掩码路径。

    命名约定（已实测核实）:
        rgb_anon/{cond}/{split}/{seq}/{seq}_frame_{n:06d}_rgb_anon.png
        gt/{cond}/{split}/{seq}/{seq}_frame_{n:06d}_gt_{suffix}.png
    """
    rel = image_path.relative_to(acdc_root / "rgb_anon")
    stem = image_path.name.replace("_rgb_anon.png", "")
    return acdc_root / "gt" / rel.parent / f"{stem}_gt_{suffix}.png"


def check_acdc(
    acdc_root: str | Path,
    report: CleaningReport,
    cfg: dict[str, Any],
    checks: dict[str, bool],
    *,
    num_workers: int = 4,
    max_samples_per_issue: int = 50,
) -> None:
    """ACDC 全量清洗。"""
    root = Path(acdc_root)
    if not root.exists():
        report.mark_skipped("acdc", f"路径不存在: {root}")
        return

    all_images = sorted((root / "rgb_anon").rglob("*.png"))
    if not all_images:
        report.mark_skipped("acdc", f"未找到图像: {root / 'rgb_anon'}")
        return

    # --- 1. 图像完整性 ---
    if checks.get("image_integrity", True) or checks.get("image_statistics", True):
        integ = cfg.get("integrity", {})
        stat_cfg = cfg.get("image_statistics", {})
        results = _run_probes(
            all_images,
            report,
            check="acdc/image_integrity",
            force_load=bool(integ.get("full_decode", True) and integ.get("force_load", True)),
            want_stats=bool(checks.get("image_statistics", True)),
            downsample=int(stat_cfg.get("std_check_downsample", 4)),
            num_workers=num_workers,
        )
        report.mark_checked("acdc.images_probed", len(results))
        report.stats["acdc_total_images"] = len(all_images)

        if checks.get("image_statistics", True):
            _check_dimensions_and_degeneracy(
                results, report, stat_cfg, check="acdc/image_statistics", label="acdc"
            )
    else:
        results = []
        report.mark_skipped("acdc/image_integrity", "配置中已关闭")

    # --- 2. 图像与掩码配对 ---
    if checks.get("acdc_image_mask_pairing", True):
        n_paired = n_missing = 0
        for img in all_images:
            rel = img.relative_to(root / "rgb_anon")
            split = rel.parts[1] if len(rel.parts) > 2 else ""
            if split not in ACDC_ANNOTATED_SPLITS:
                continue  # test 与 *_ref 无标注属预期，见模块头说明
            n_paired += 1
            for suffix in ACDC_MASK_SUFFIXES:
                mp = _acdc_mask_path(img, root, suffix)
                if not mp.exists():
                    n_missing += 1
                    report.error(
                        "acdc/mask_pairing", img,
                        f"缺少标注变体 {suffix}（期望 {mp.name}）",
                    )
        report.mark_checked("acdc.annotated_images", n_paired)
        report.stats["acdc_missing_masks"] = n_missing
        report.stats["acdc_unannotated_images"] = len(all_images) - n_paired

        # 反向：是否有孤儿掩码（有标注却没有对应图像）
        n_orphan = 0
        for mp in (root / "gt").rglob("*.png"):
            if not mp.name.endswith("_gt_labelIds.png"):
                continue
            rel = mp.relative_to(root / "gt")
            stem = mp.name.replace("_gt_labelIds.png", "")
            if not (root / "rgb_anon" / rel.parent / f"{stem}_rgb_anon.png").exists():
                n_orphan += 1
                report.warn("acdc/mask_pairing", mp, "孤儿标注：找不到对应图像")
        if n_orphan:
            report.stats["acdc_orphan_masks"] = n_orphan
    else:
        report.mark_skipped("acdc_image_mask_pairing", "配置中已关闭")

    # --- 3. 掩码有效性 ---
    if checks.get("acdc_mask_validity", True):
        mask_cfg = cfg.get("mask_validity", {})
        ignore_index = int(mask_cfg.get("ignore_index", 255))
        min_valid = float(mask_cfg.get("min_valid_pixel_ratio", 0.05))
        single_ratio = float(mask_cfg.get("single_class_ratio", 0.995))
        max_id = int(mask_cfg.get("max_class_id", 33))

        label_ids = sorted((root / "gt").rglob("*_gt_labelIds.png"))
        n_bad = 0
        for mp in label_ids:
            try:
                m = np.asarray(Image.open(mp))
            except Exception as exc:  # noqa: BLE001
                report.error("acdc/mask_validity", mp, f"掩码无法读取: {exc}")
                n_bad += 1
                continue

            vals, counts = np.unique(m, return_counts=True)
            total = m.size or 1
            valid = total - int(counts[vals == ignore_index].sum())
            if valid / total < min_valid:
                report.error(
                    "acdc/mask_validity", mp,
                    f"有效像素仅 {valid / total:.1%}（下限 {min_valid:.0%}），几乎全为 ignore",
                )
                n_bad += 1
                continue

            over = [int(v) for v in vals if v != ignore_index and (v > max_id or v < 0)]
            if over:
                report.error("acdc/mask_validity", mp, f"类别 ID 越界: {sorted(over)[:5]}")
                n_bad += 1
                continue

            top = counts.max() / total
            if top > single_ratio:
                only = int(vals[counts.argmax()])
                report.warn(
                    "acdc/mask_validity", mp,
                    f"{top:.1%} 像素为单一类别 id={only}，真实驾驶场景不太可能",
                )
        report.mark_checked("acdc.masks_validated", len(label_ids))
        report.stats["acdc_masks_suspect"] = n_bad
    else:
        report.mark_skipped("acdc_mask_validity", "配置中已关闭")

    # --- 4. 检测标注 ---
    if checks.get("detection_annotations", True):
        _check_detection_jsons(root / "gt_detection", root / "rgb_anon", report, cfg)
    else:
        report.mark_skipped("detection_annotations", "配置中已关闭")


def _check_detection_jsons(
    det_root: Path, rgb_root: Path, report: CleaningReport, cfg: dict[str, Any]
) -> None:
    """校验 ACDC 检测标注（COCO 格式）。"""
    if not det_root.exists():
        report.mark_skipped("detection_annotations", f"路径不存在: {det_root}")
        return

    dcfg = cfg.get("detection", {})
    oob_ratio = float(dcfg.get("max_out_of_bounds_ratio", 0.5))
    min_area = float(dcfg.get("min_bbox_area_px", 4))
    min_side = float(dcfg.get("min_bbox_side_px", 2))
    validate_rle = bool(dcfg.get("validate_rle", False))

    if validate_rle:
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            report.mark_skipped(
                "detection/rle",
                "需要 pycocotools，未安装。bbox 检查不受影响；"
                "若要校验实例掩码的 RLE 可解码性，请安装 pycocotools",
            )
            validate_rle = False

    jsons = sorted(det_root.rglob("*_gt_detection.json"))
    if not jsons:
        report.mark_skipped("detection_annotations", f"未找到 *_gt_detection.json: {det_root}")
        return

    total_ann = 0
    for jp in jsons:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            report.error("detection/json", jp, f"JSON 解析失败: {exc}")
            continue

        images = data.get("images", [])
        anns = data.get("annotations", [])
        cats = {c["id"] for c in data.get("categories", [])}
        img_by_id = {im["id"]: im for im in images}
        total_ann += len(anns)

        n_oob = n_tiny = n_orphan = n_badcat = 0
        n_rle_fail = 0
        for a in anns:
            im = img_by_id.get(a.get("image_id"))
            if im is None:
                n_orphan += 1
                continue
            if a.get("category_id") not in cats:
                n_badcat += 1

            x, y, w, h = a.get("bbox", [0, 0, 0, 0])
            W, H = im.get("width", 0), im.get("height", 0)
            if w < min_side or h < min_side or w * h < min_area:
                n_tiny += 1
            if W and H:
                # 框完全落在图像外的面积占比
                ix = max(0.0, min(x + w, W) - max(x, 0.0))
                iy = max(0.0, min(y + h, H) - max(y, 0.0))
                inside = ix * iy
                if (w * h) > 0 and 1.0 - inside / (w * h) > oob_ratio:
                    n_oob += 1

            if validate_rle:
                try:
                    from pycocotools import mask as mask_utils

                    seg = a.get("segmentation")
                    if isinstance(seg, dict):
                        mask_utils.decode(seg)
                except Exception:  # noqa: BLE001
                    n_rle_fail += 1

        key = str(jp.relative_to(det_root.parent))
        if n_oob:
            report.warn("detection/bbox_bounds", jp, f"{n_oob} 个框超出图像边界比例 > {oob_ratio:.0%}")
        if n_tiny:
            report.warn("detection/bbox_size", jp, f"{n_tiny} 个框面积或边长过小（噪声标注）")
        if n_orphan:
            report.error("detection/orphan", jp, f"{n_orphan} 条标注引用了不存在的 image_id")
        if n_badcat:
            report.error("detection/category", jp, f"{n_badcat} 条标注的 category_id 不在 categories 中")
        if validate_rle and n_rle_fail:
            report.error("detection/rle", jp, f"{n_rle_fail} 条 segmentation RLE 无法解码")

        report.stats.setdefault("detection_files", {})[key] = {
            "images": len(images), "annotations": len(anns),
            "oob": n_oob, "tiny": n_tiny, "orphan": n_orphan,
        }

        # 反向：标注里声明的图像在磁盘上是否存在
        n_missing_img = 0
        for im in images:
            if not (rgb_root / im["file_name"]).exists():
                n_missing_img += 1
        if n_missing_img:
            report.error(
                "detection/missing_image", jp,
                f"{n_missing_img} 张被标注引用的图像在 rgb_anon/ 下不存在",
            )

    report.mark_checked("detection.annotations", total_ann)
    report.mark_checked("detection.files", len(jsons))


# =============================================================================
# KITTI 检查
# =============================================================================


def check_kitti(
    kitti_root: str | Path,
    report: CleaningReport,
    cfg: dict[str, Any],
    checks: dict[str, bool],
    *,
    num_workers: int = 4,
) -> None:
    """KITTI 全量清洗。

    重点与 ACDC 不同：KITTI 的图像尺寸逐帧不同、内参逐帧不同，
    所以这里不做「尺寸必须统一」的判断，而是校验 image/label/calib 三元组齐全、
    以及内参与图像尺寸是否自洽。
    """
    root = Path(kitti_root)
    if not root.exists():
        report.mark_skipped("kitti", f"路径不存在: {root}")
        return

    kcfg = cfg.get("kitti", {})
    require_calib = bool(kcfg.get("require_calib", True))
    pp_tol = float(kcfg.get("principal_point_tolerance", 0.08))

    for split in ("training", "testing"):
        img_dir = root / split / "image_2"
        if not img_dir.exists():
            continue
        images = sorted(img_dir.glob("*.png"))
        if not images:
            report.mark_skipped(f"kitti/{split}", f"目录下无 png: {img_dir}")
            continue

        if checks.get("image_integrity", True) or checks.get("image_statistics", True):
            integ = cfg.get("integrity", {})
            stat_cfg = cfg.get("image_statistics", {})
            results = _run_probes(
                images, report,
                check=f"kitti/{split}/image_integrity",
                force_load=bool(integ.get("full_decode", True) and integ.get("force_load", True)),
                want_stats=bool(checks.get("image_statistics", True)),
                downsample=int(stat_cfg.get("std_check_downsample", 4)),
                num_workers=num_workers,
            )
            report.mark_checked(f"kitti.{split}.images_probed", len(results))
            if checks.get("image_statistics", True):
                _check_dimensions_and_degeneracy(
                    results, report, stat_cfg,
                    check=f"kitti/{split}/image_statistics", label=f"kitti_{split}",
                )

        if not checks.get("kitti_triplets", True):
            continue

        # --- 三元组齐全性与内参自洽 ---
        label_dir, calib_dir = root / split / "label_2", root / split / "calib"
        n_missing_label = n_missing_calib = n_bad_calib = n_pp_off = 0

        for img in images:
            stem = img.stem
            lp = label_dir / f"{stem}.txt"
            cp = calib_dir / f"{stem}.txt"
            if label_dir.exists() and not lp.exists():
                n_missing_label += 1
            if not cp.exists():
                n_missing_calib += 1
                if require_calib:
                    report.error(f"kitti/{split}/triplet", img, "缺少对应 calib 文件")
                continue

            try:
                p2 = None
                for line in cp.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("P2:"):
                        p2 = [float(v) for v in line.split()[1:]]
                        break
                if p2 is None or len(p2) < 12:
                    n_bad_calib += 1
                    report.error(f"kitti/{split}/calib", cp, "P2 缺失或字段数不足 12")
                    continue
                with Image.open(img) as im:
                    W, H = im.size
                cx, cy = p2[2], p2[6]
                if abs(cx / W - 0.5) > pp_tol or abs(cy / H - 0.5) > pp_tol:
                    n_pp_off += 1
                    report.warn(
                        f"kitti/{split}/calib", cp,
                        f"主点偏离图像中心较大: cx/W={cx / W:.3f} cy/H={cy / H:.3f}（容差 {pp_tol}）",
                    )
            except Exception as exc:  # noqa: BLE001
                n_bad_calib += 1
                report.error(f"kitti/{split}/calib", cp, f"内参解析失败: {exc}")

        report.stats[f"kitti_{split}"] = {
            "images": len(images),
            "missing_label": n_missing_label,
            "missing_calib": n_missing_calib,
            "bad_calib": n_bad_calib,
            "principal_point_off": n_pp_off,
        }
        if n_missing_label:
            report.warn(
                f"kitti/{split}/triplet", img_dir,
                f"{n_missing_label} 张图缺少对应的 label_2 标注"
                + ("（testing 集官方不提供标注，属预期）" if split == "testing" else ""),
            )

        if kcfg.get("report_size_variation", True):
            report.info(
                f"kitti/{split}/image_statistics", img_dir,
                "KITTI 图像尺寸逐帧不同、内参逐帧不同，这是官方数据集的固有特性，非缺陷。"
                "测距时必须逐帧读取该帧自己的 calib，不可使用统一内参。"
                "详见 docs/label_spec.md 4.3",
            )


# =============================================================================
# 重复检测
# =============================================================================


def _file_digest(path: Path, nbytes: int) -> str:
    """取文件前 nbytes 字节的 sha1。

    只用前 256KB 而非全文哈希：全量哈希 16GB 需要数分钟且收益有限，
    前 256KB 足以区分不同照片（图像头部包含尺寸、调色板与起始像素数据）。
    """
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def _dhash(path: Path, hash_size: int = 8) -> np.ndarray | None:
    """差分感知哈希。返回 bool 数组，None 表示读取失败。

    自行实现而非引入 imagehash：逻辑只有几行，
    避免为一个函数增加一个依赖。
    """
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
            a = np.asarray(g, dtype=np.int16)
        return (a[:, 1:] > a[:, :-1]).flatten()
    except Exception:  # noqa: BLE001
        return None


def check_duplicates(
    paths: Sequence[Path],
    report: CleaningReport,
    cfg: dict[str, Any],
    checks: dict[str, bool],
    *,
    label: str,
) -> None:
    """精确重复与近重复检测。"""
    if checks.get("duplicates_exact", True):
        nbytes = int(cfg.get("exact_hash_bytes", 262144))
        seen: dict[str, list[str]] = defaultdict(list)
        for p in paths:
            try:
                seen[_file_digest(p, nbytes)].append(str(p))
            except Exception as exc:  # noqa: BLE001
                report.error(f"{label}/duplicates_exact", p, f"读取失败: {exc}")
        dups = {k: v for k, v in seen.items() if len(v) > 1}
        report.mark_checked(f"{label}.exact_hashed", len(paths))
        report.stats[f"{label}_exact_duplicate_groups"] = len(dups)
        for k, v in dups.items():
            for extra in v[1:]:
                report.warn(
                    f"{label}/duplicates_exact", extra,
                    f"与 {v[0]} 内容重复（前 {nbytes // 1024}KB 哈希相同）",
                )
    else:
        report.mark_skipped(f"{label}/duplicates_exact", "配置中已关闭")

    if checks.get("duplicates_near", False):
        hs = int(cfg.get("near_hash_size", 8))
        th = int(cfg.get("near_hamming_threshold", 5))
        hashes: list[tuple[str, np.ndarray]] = []
        for p in paths:
            h = _dhash(p, hs)
            if h is not None:
                hashes.append((str(p), h))
        n_pairs = 0
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if int(np.count_nonzero(hashes[i][1] != hashes[j][1])) <= th:
                    n_pairs += 1
                    report.warn(
                        f"{label}/duplicates_near", hashes[j][0],
                        f"与 {hashes[i][0]} 高度相似（dHash 汉明距离 <= {th}）",
                    )
        report.mark_checked(f"{label}.near_hashed", len(hashes))
        report.stats[f"{label}_near_duplicate_pairs"] = n_pairs
    else:
        report.mark_skipped(
            f"{label}/duplicates_near",
            "配置中默认关闭（O(n²) 比较，15k 图量级开销大）。需要时在 cleaning.yaml 开启",
        )


# =============================================================================
# 统计离群（可选）
# =============================================================================


def check_statistical_outliers(
    results: Sequence[dict[str, Any]],
    report: CleaningReport,
    cfg: dict[str, Any],
    *,
    label: str,
) -> None:
    """用统计方法找「能正常解码但特征异常」的图像。

    这是完整性检查之外的一层：损坏文件 Pillow 会报错，但曝光异常、
    内容错帧这类问题只能从统计特征看出来。
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        report.mark_skipped(
            f"{label}/statistical_outliers", "需要 scikit-learn，未安装"
        )
        return

    feats = cfg.get("features", ["brightness_mean", "brightness_std"])
    rows, paths = [], []
    for r in results:
        if not r["ok"]:
            continue
        vals = [r.get(f) for f in feats]
        if any(v is None for v in vals):
            continue
        rows.append(vals)
        paths.append(r["path"])

    if len(rows) < 32:
        report.mark_skipped(f"{label}/statistical_outliers", f"有效样本仅 {len(rows)}，不足以做离群检测")
        return

    X = np.asarray(rows, dtype=np.float64)
    model = IsolationForest(
        contamination=float(cfg.get("contamination", 0.01)),
        random_state=int(cfg.get("random_state", 42)),
        n_estimators=200,
    )
    pred = model.fit_predict(X)
    n_out = int((pred == -1).sum())
    for p, flag in zip(paths, pred):
        if flag == -1:
            report.warn(
                f"{label}/statistical_outliers", p,
                "统计特征离群（IsolationForest）。多数情况下是正常的极端场景，"
                "建议人工看一眼再决定，不要直接剔除",
            )
    report.mark_checked(f"{label}.outliers_scanned", len(paths))
    report.stats[f"{label}_statistical_outliers"] = n_out


# =============================================================================
# 顶层入口
# =============================================================================


def _write_manifest(
    report: CleaningReport, all_paths: Sequence[str | Path], out_dir: str | Path, name: str
) -> Path:
    """写出有效样本清单 —— 清洗的实际交付物，数据集类据此过滤。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    valid = [str(p) for p in all_paths if report.status_of(p) is SampleStatus.VALID]
    suspect = [str(p) for p in all_paths if report.status_of(p) is SampleStatus.SUSPECT]
    invalid = [str(p) for p in all_paths if report.status_of(p) is SampleStatus.INVALID]

    payload = {
        "name": name,
        "note": "raw 数据未被修改；本清单是过滤依据，由数据集类消费",
        "counts": {"valid": len(valid), "suspect": len(suspect), "invalid": len(invalid)},
        "suspect": suspect,
        "invalid": invalid,
    }
    path = out / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "%s 清单已写出: valid=%d suspect=%d invalid=%d -> %s",
        name, len(valid), len(suspect), len(invalid), path,
    )
    return path


def clean(
    cfg: dict[str, Any],
    *,
    project_root: str | Path = ".",
    only: Iterable[str] | None = None,
    corrupt_images_only: bool = False,
) -> CleaningReport:
    """执行数据清洗。

    Args:
        cfg: cleaning 配置段（configs/data/cleaning.yaml 的 ``cleaning`` 键）
        project_root: 项目根目录，配置里的相对路径以此为基准
        only: 只跑指定的数据集，如 ``{"acdc"}``；None 表示全部
        corrupt_images_only: 只检查完整图像解码，不检查配对、统计特征或重复项

    Returns:
        填充好的 CleaningReport

    Note:
        本函数**不修改任何原始文件**。产物是 artifacts/ 下的报告与
        data/processed/manifests/ 下的有效样本清单。
    """
    root = Path(project_root)
    report = CleaningReport()
    checks = {} if corrupt_images_only else cfg.get("checks", {})
    workers = int(cfg.get("num_workers", 4))
    out_cfg = cfg.get("output", {})
    max_samp = int(out_cfg.get("max_samples_per_issue", 50))

    targets = set(only) if only else {"acdc", "kitti"}

    if "acdc" in targets:
        acdc_root = root / "data" / "raw" / "acdc"
        if not corrupt_images_only:
            check_acdc(acdc_root, report, cfg, checks, num_workers=workers,
                       max_samples_per_issue=max_samp)
        acdc_images = sorted((acdc_root / "rgb_anon").rglob("*.png")) if acdc_root.exists() else []
        if acdc_images:
            if corrupt_images_only:
                results = _run_probes(
                    acdc_images,
                    report,
                    check="acdc/image_integrity",
                    force_load=True,
                    want_stats=False,
                    downsample=1,
                    num_workers=workers,
                )
                report.mark_checked("acdc.images_probed", len(results))
                report.stats["acdc_total_images"] = len(acdc_images)
            else:
                check_duplicates(
                    acdc_images, report, cfg.get("duplicates", {}), checks, label="acdc"
                )
            if checks.get("statistical_outliers", False):
                integ = cfg.get("integrity", {})
                stat_cfg = cfg.get("image_statistics", {})
                res = _run_probes(
                    acdc_images, report, check="acdc/stats_probe",
                    force_load=bool(integ.get("force_load", True)), want_stats=True,
                    downsample=int(stat_cfg.get("std_check_downsample", 4)),
                    num_workers=workers,
                )
                check_statistical_outliers(
                    res, report, cfg.get("statistical_outliers", {}), label="acdc"
                )
            _write_manifest(report, acdc_images, root / out_cfg.get(
                "manifest_dir", "data/processed/manifests"), "acdc")

    if "kitti" in targets:
        kitti_root = root / "data" / "external" / "kitti"
        if not corrupt_images_only:
            check_kitti(kitti_root, report, cfg, checks, num_workers=workers)
        kitti_images = sorted(kitti_root.rglob("image_2/*.png")) if kitti_root.exists() else []
        if kitti_images:
            if not corrupt_images_only:
                check_duplicates(
                    kitti_images, report, cfg.get("duplicates", {}), checks, label="kitti"
                )
            else:
                results = _run_probes(
                    kitti_images, report, check="kitti/image_integrity",
                    force_load=True, want_stats=False, downsample=1, num_workers=workers,
                )
                report.mark_checked("kitti.images_probed", len(results))
            _write_manifest(report, kitti_images, root / out_cfg.get(
                "manifest_dir", "data/processed/manifests"), "kitti")

    if "pixel_accurate_benchmark" in targets:
        benchmark_cfg = cfg.get("pixel_accurate_benchmark", {})
        benchmark_root = root / benchmark_cfg.get(
            "root", "data/external/pixel_accurate_depth_benchmark/pixel_accurate_depth_benchmark"
        )
        if not benchmark_root.exists():
            report.mark_skipped("pixel_accurate_benchmark", f"路径不存在: {benchmark_root}")
        else:
            image_archives = sorted(benchmark_root.glob("*.zip"))
            image_paths = check_zip_image_integrity(
                image_archives,
                report,
                check="pixel_accurate_benchmark/image_integrity",
                project_root=root,
            )
            if image_paths:
                _write_manifest(
                    report,
                    image_paths,
                    root / out_cfg.get("manifest_dir", "data/processed/manifests"),
                    "pixel_accurate_benchmark",
                )

    report.write(root / out_cfg.get("report_dir", "artifacts/reports/cleaning"), max_samp)
    return report
