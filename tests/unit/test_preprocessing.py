"""数据清洗模块的单元测试。

职责:
    - 验证损坏文件、退化图、尺寸异常能被正确识别
    - 验证 ACDC 掩码路径推导与配对逻辑
    - 验证报告容器的严重度提升与清单写出
    - 验证检测标注（COCO）的越界/孤儿/类别检查

测试全部使用临时目录中的合成图像，**不依赖真实的 ACDC / KITTI 数据**，
因此在没有数据的 CI 环境里也能跑。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from car_smart_assist.data import preprocessing as pp


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_image(tmp_path: Path):
    """生成一张正常图像并返回路径。"""

    def _make(name: str = "ok.png", size: tuple[int, int] = (320, 240), uniform: bool = False) -> Path:
        p = tmp_path / name
        if uniform:
            arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
        else:
            rng = np.random.default_rng(0)
            arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(arr).save(p)
        return p

    return _make


# ---------------------------------------------------------------------------
# 图像探测
# ---------------------------------------------------------------------------


class TestProbeImage:
    def test_valid_image(self, tmp_image):
        p = tmp_image("a.png", (320, 240))
        r = pp._probe_image((str(p), True, True, 1))
        assert r["ok"] is True
        assert r["error"] is None
        assert (r["width"], r["height"]) == (320, 240)
        assert r["mode"] == "RGB"
        # 随机噪声图的统计量应落在合理区间
        assert r["entropy"] > 5.0
        assert r["edge_density"] > 0.0

    def test_truncated_png_is_caught(self, tmp_path: Path):
        """截断的 PNG 必须被判为不可用 —— 这是清洗的核心能力。"""
        p = tmp_path / "truncated.png"
        good = tmp_path / "good.png"
        Image.fromarray(
            np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        ).save(good)
        data = good.read_bytes()
        p.write_bytes(data[: len(data) // 2])  # 砍掉一半

        r = pp._probe_image((str(p), True, False, 1))
        assert r["ok"] is False
        assert r["error"] is not None

    def test_nonexistent_file_is_caught(self, tmp_path: Path):
        r = pp._probe_image((str(tmp_path / "nope.png"), True, False, 1))
        assert r["ok"] is False
        assert "FileNotFoundError" in r["error"]

    def test_non_image_file_is_caught(self, tmp_path: Path):
        p = tmp_path / "fake.png"
        p.write_text("this is not an image")
        r = pp._probe_image((str(p), True, False, 1))
        assert r["ok"] is False

    def test_probe_never_raises(self, tmp_path: Path):
        """worker 在子进程里跑，任何输入都不允许抛异常。"""
        for bad in ["", str(tmp_path), str(tmp_path / "x.png")]:
            r = pp._probe_image((bad, True, True, 1))
            assert r["ok"] is False


# ---------------------------------------------------------------------------
# 尺寸与退化
# ---------------------------------------------------------------------------


class TestDimensionsAndDegeneracy:
    CFG = {
        "min_side_px": 64,
        "aspect_ratio_range": [0.5, 6.0],
        "min_pixel_std": 2.0,
    }

    def _run(self, results):
        rep = pp.CleaningReport()
        pp._check_dimensions_and_degeneracy(
            results, rep, self.CFG, check="t", label="t"
        )
        return rep

    def test_normal_image_passes(self, tmp_image):
        r = pp._probe_image((str(tmp_image("n.png", (320, 240))), True, True, 1))
        rep = self._run([r])
        assert not [i for i in rep.issues if i.severity is pp.Severity.ERROR]

    def test_uniform_image_flagged_degenerate(self, tmp_image):
        r = pp._probe_image((str(tmp_image("u.png", (320, 240), uniform=True)), True, True, 1))
        rep = self._run([r])
        assert any("退化图" in i.message for i in rep.issues)
        assert rep.status_of(r["path"]) is pp.SampleStatus.INVALID

    def test_tiny_image_flagged(self, tmp_image):
        r = pp._probe_image((str(tmp_image("tiny.png", (16, 16))), True, True, 1))
        rep = self._run([r])
        assert any("尺寸过小" in i.message for i in rep.issues)

    def test_extreme_aspect_ratio_flagged(self, tmp_image):
        r = pp._probe_image((str(tmp_image("wide.png", (800, 80))), True, True, 1))
        rep = self._run([r])
        assert any("宽高比异常" in i.message for i in rep.issues)

    def test_size_variation_is_info_not_error(self, tmp_image):
        """KITTI 天然多尺寸 —— 必须记 INFO，不能报成错误。"""
        rs = [
            pp._probe_image((str(tmp_image(f"s{i}.png", (320 + i, 240))), True, True, 1))
            for i in range(3)
        ]
        rep = self._run(rs)
        info = [i for i in rep.issues if i.severity is pp.Severity.INFO]
        assert any("种图像尺寸" in i.message for i in info)
        assert not [i for i in rep.issues if i.severity is pp.Severity.ERROR]


# ---------------------------------------------------------------------------
# ACDC 路径推导
# ---------------------------------------------------------------------------


class TestAcdcMaskPath:
    def test_derivation_matches_official_convention(self, tmp_path: Path):
        root = tmp_path / "acdc"
        img = root / "rgb_anon" / "fog" / "val" / "GOPR0476" / "GOPR0476_frame_000761_rgb_anon.png"
        img.parent.mkdir(parents=True)

        got = pp._acdc_mask_path(img, root, "labelIds")
        want = root / "gt" / "fog" / "val" / "GOPR0476" / "GOPR0476_frame_000761_gt_labelIds.png"
        assert got == want

    def test_all_five_suffixes_derived(self, tmp_path: Path):
        root = tmp_path / "acdc"
        img = root / "rgb_anon" / "snow" / "train" / "GP010475" / "GP010475_frame_001043_rgb_anon.png"
        for suffix in pp.ACDC_MASK_SUFFIXES:
            p = pp._acdc_mask_path(img, root, suffix)
            assert p.name.endswith(f"_gt_{suffix}.png")


# ---------------------------------------------------------------------------
# 哈希
# ---------------------------------------------------------------------------


class TestHashing:
    def test_identical_content_same_digest(self, tmp_image):
        a = tmp_image("a.png")
        b = tmp_image("b.png")
        b.write_bytes(a.read_bytes())
        assert pp._file_digest(a, 262144) == pp._file_digest(b, 262144)

    def test_different_content_different_digest(self, tmp_image):
        a = tmp_image("a.png")
        c = tmp_image("c.png", uniform=True)
        assert pp._file_digest(a, 262144) != pp._file_digest(c, 262144)

    def test_dhash_identical_is_zero_distance(self, tmp_image):
        a = tmp_image("a.png")
        b = tmp_image("b.png")
        b.write_bytes(a.read_bytes())
        assert int((pp._dhash(a) != pp._dhash(b)).sum()) == 0

    def test_dhash_returns_none_on_bad_file(self, tmp_path: Path):
        p = tmp_path / "bad.png"
        p.write_text("nope")
        assert pp._dhash(p) is None


# ---------------------------------------------------------------------------
# 报告容器
# ---------------------------------------------------------------------------


class TestCleaningReport:
    def test_error_dominates_warning(self):
        """同一路径同时有 WARNING 与 ERROR 时，状态必须是 INVALID。"""
        rep = pp.CleaningReport()
        rep.warn("c", "/a.png", "先警告")
        rep.error("c", "/a.png", "后报错")
        assert rep.status_of("/a.png") is pp.SampleStatus.INVALID

    def test_warning_after_error_does_not_downgrade(self):
        rep = pp.CleaningReport()
        rep.error("c", "/a.png", "报错")
        rep.warn("c", "/a.png", "再警告")
        assert rep.status_of("/a.png") is pp.SampleStatus.INVALID

    def test_info_does_not_mark_sample(self):
        rep = pp.CleaningReport()
        rep.info("c", "/a.png", "备查")
        assert rep.status_of("/a.png") is pp.SampleStatus.VALID

    def test_counts(self):
        rep = pp.CleaningReport()
        rep.error("x", "/1", "e")
        rep.warn("y", "/2", "w")
        rep.info("y", "/3", "i")
        assert rep.counts_by_severity() == {"error": 1, "warning": 1, "info": 1}
        assert rep.counts_by_check()["y"] == {"warning": 1, "info": 1}

    def test_skipped_recorded_with_reason(self):
        rep = pp.CleaningReport()
        rep.mark_skipped("foo", "因为缺库")
        assert rep.skipped["foo"] == "因为缺库"

    def test_write_produces_json_and_markdown(self, tmp_path: Path):
        rep = pp.CleaningReport()
        rep.error("c", "/bad.png", "坏了")
        rep.mark_checked("c", 10)
        rep.mark_skipped("d", "未安装某库")
        rep.stats["k"] = 1
        p = rep.write(tmp_path, max_samples_per_issue=5)

        assert p.exists()
        assert (tmp_path / "report.md").exists()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["summary"]["by_severity"]["error"] == 1
        assert data["checked"] == {"c": 10}
        # 跳过的项必须出现在报告里 —— 否则会造成「跑过了就没问题」的错觉
        assert data["skipped"] == {"d": "未安装某库"}

    def test_write_truncates_long_issue_lists(self, tmp_path: Path):
        rep = pp.CleaningReport()
        for i in range(50):
            rep.error("c", f"/bad{i}.png", "坏")
        rep.write(tmp_path, max_samples_per_issue=5)
        md = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "另有" in md


# ---------------------------------------------------------------------------
# 检测标注（COCO）
# ---------------------------------------------------------------------------


class TestDetectionJsonChecks:
    def _make_coco(self, tmp_path: Path, *, oob: bool = False, orphan: bool = False,
                   bad_cat: bool = False, tiny: bool = False,
                   create_image: bool = True) -> Path:
        det = tmp_path / "gt_detection"
        det.mkdir(parents=True, exist_ok=True)
        rel_name = "fog/train/GP/frame.png"

        # 造出被引用的图像文件，否则 missing_image 检查会（正确地）报错
        if create_image:
            img_path = tmp_path / "rgb_anon" / rel_name
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
            ).save(img_path)

        ann = {
            "id": 1, "image_id": 0, "category_id": 26,
            "bbox": [10, 10, 50, 40], "area": 2000, "iscrowd": 0,
            "segmentation": [],
        }
        if oob:
            ann["bbox"] = [-900, -900, 20, 20]      # 几乎全在图像外
        if tiny:
            ann["bbox"] = [10, 10, 1, 1]            # 噪声标注
        if orphan:
            ann["image_id"] = 999                   # 不存在的图像
        if bad_cat:
            ann["category_id"] = 777                # 不在 categories 中

        payload = {
            "images": [{"id": 0, "file_name": rel_name,
                        "width": 1920, "height": 1080}],
            "categories": [{"id": 26, "name": "car", "supercategory": "vehicle"}],
            "annotations": [ann],
        }
        p = det / "instancesonly_train_gt_detection.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    CFG = {"detection": {"max_out_of_bounds_ratio": 0.5, "min_bbox_area_px": 4,
                         "min_bbox_side_px": 2, "validate_rle": False}}

    def _run(self, tmp_path: Path):
        rep = pp.CleaningReport()
        # 不建 rgb_anon 目录：被标注引用的图像不存在，这本身也应被检出
        pp._check_detection_jsons(tmp_path / "gt_detection", tmp_path / "rgb_anon", rep, self.CFG)
        return rep

    def test_clean_annotation(self, tmp_path: Path):
        self._make_coco(tmp_path)
        rep = self._run(tmp_path)
        assert not [i for i in rep.issues if i.severity is pp.Severity.ERROR]
        assert rep.checked["detection.annotations"] == 1

    def test_out_of_bounds_bbox_warns(self, tmp_path: Path):
        self._make_coco(tmp_path, oob=True)
        rep = self._run(tmp_path)
        assert any("超出图像边界" in i.message for i in rep.issues)
        assert any(i.severity is pp.Severity.WARNING for i in rep.issues)

    def test_orphan_annotation_is_error(self, tmp_path: Path):
        self._make_coco(tmp_path, orphan=True)
        rep = self._run(tmp_path)
        assert any("不存在的 image_id" in i.message for i in rep.issues)
        assert any(i.severity is pp.Severity.ERROR for i in rep.issues)

    def test_unknown_category_is_error(self, tmp_path: Path):
        self._make_coco(tmp_path, bad_cat=True)
        rep = self._run(tmp_path)
        assert any("category_id 不在" in i.message for i in rep.issues)

    def test_missing_image_on_disk_is_error(self, tmp_path: Path):
        # 故意不创建被引用的图像文件
        self._make_coco(tmp_path, create_image=False)
        rep = self._run(tmp_path)
        assert any("在 rgb_anon/ 下不存在" in i.message for i in rep.issues)
        assert any(i.severity is pp.Severity.ERROR for i in rep.issues)

    def test_rle_skip_recorded_when_pycocotools_missing(self, tmp_path: Path):
        """pycocotools 未安装时必须记 skipped 并说明原因，而不是静默跳过。"""
        self._make_coco(tmp_path)
        cfg = {"detection": {**self.CFG["detection"], "validate_rle": True}}
        rep = pp.CleaningReport()
        pp._check_detection_jsons(tmp_path / "gt_detection", tmp_path / "rgb_anon", rep, cfg)
        try:
            import pycocotools  # noqa: F401
            has = True
        except ImportError:
            has = False
        if not has:
            assert "detection/rle" in rep.skipped
            assert "pycocotools" in rep.skipped["detection/rle"]


# ---------------------------------------------------------------------------
# 清单
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_lists_invalid_and_suspect(self, tmp_path: Path):
        # 各状态的数量必须**互不相同**。
        # 早期版本用 4 条路径（2 valid / 1 suspect / 1 invalid），
        # 而 valid 的过滤条件被误写成 `is not`，反转后数量恰好也等于 2，
        # 测试因此侥幸通过、没能发现这个 bug。7 = 4+2+1 让巧合不可能发生。
        rep = pp.CleaningReport()
        paths = [tmp_path / f"{i}.png" for i in range(7)]
        rep.error("c", paths[0], "坏")
        rep.warn("c", paths[1], "可疑")
        rep.warn("c", paths[2], "可疑")
        p = pp._write_manifest(rep, paths, tmp_path / "man", "t")

        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["counts"] == {"valid": 4, "suspect": 2, "invalid": 1}
        assert str(paths[0]) in data["invalid"]
        assert str(paths[1]) in data["suspect"]
        assert str(paths[2]) in data["suspect"]
        # valid 列表不落盘（体积考虑），但它的计数必须与真实有效数一致
        assert data["counts"]["valid"] == len(paths) - 2 - 1

    def test_manifest_valid_count_is_not_inverted(self, tmp_path: Path):
        """回归测试：valid 计数绝不能是「非 valid」的数量。

        构造一个极端分布 —— 只有 1 张有问题、其余全部正常 ——
        反转 bug 会把 valid 报成 1。
        """
        rep = pp.CleaningReport()
        paths = [tmp_path / f"{i}.png" for i in range(50)]
        rep.error("c", paths[0], "坏")
        p = pp._write_manifest(rep, paths, tmp_path / "man", "t")
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["counts"]["valid"] == 49, "valid 计数被反转了"
        assert data["counts"]["invalid"] == 1
