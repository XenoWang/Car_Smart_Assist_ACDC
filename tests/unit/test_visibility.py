"""能见度门控的单元测试。

职责:
    - 锁住多尺度编码器的结构契约（三条分支真的分别用了 3/5/7 的核）
    - 锁住判定契约：**高信息量的帧永远不判 BLIND**
    - 验证信息量特征与合成退化的行为方向正确
    - 全部用合成图像，不依赖真实 ACDC 数据与 GPU

这些测试守的是安全属性，不是数值精度：
门控判错的两个方向代价不对称 —— 漏报（看不见却说没事）是安全问题，
误报（看得见却说看不见）会让用户学会忽略它。两条都要测。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from car_smart_assist.perception.visibility import (
    GateThresholds,
    InformationFeatures,
    MultiScaleBlock,
    VisibilityGate,
    VisibilityLevel,
    VisibilityScore,
    build_autoencoder,
    compute_information_features,
    degrade,
    information_score,
    reconstruction_error,
)
from car_smart_assist.perception.visibility.autoencoder import (
    DEFAULT_KERNEL_SIZES,
    effective_kernel_size,
    reconstruction_mean,
)
from car_smart_assist.perception.visibility.dataset import (
    VisibilityImageDataset,
    sequence_of,
    split_ref_indices,
)
from car_smart_assist.perception.visibility.scorer import CalibrationStats
from car_smart_assist.perception.visibility.trainer import (
    VisibilityTrainer,
    _atomic_save,
    resolve_resume_spec,
)

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def structured_image(h: int = 144, w: int = 256, seed: int = 0) -> np.ndarray:
    """造一张有明确结构的合成图（强边缘 + 多个灰度层次）。

    用它而不是随机噪声：随机噪声的高频能量拉满，
    无法代表「清晰场景」，会让特征测试失去意义。
    """
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 90, dtype=np.float32)
    # 大面积梯度（模拟天空-道路的亮度分层）
    img += np.linspace(-60, 60, h, dtype=np.float32)[:, None, None]
    # 若干强边缘（模拟车道线、车辆轮廓）
    for i in range(6):
        y = int(h * (0.15 + 0.12 * i))
        img[y : y + 3, :, :] += 110
    for j in range(5):
        x = int(w * (0.1 + 0.18 * j))
        img[:, x : x + 4, :] += 80
    img += rng.normal(0, 6.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def uniform_image(value: int = 0, h: int = 144, w: int = 256) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 多尺度块
# ---------------------------------------------------------------------------


class TestMultiScaleBlock:
    def test_uses_all_three_kernel_branches(self):
        """三条分支必须真的分别用 3/5/7 的核 —— 这是本次改动的核心。"""
        blk = MultiScaleBlock(16, 24, kernel_sizes=(3, 5, 7), stride=2)
        ks = [c.kernel_size[0] for c in blk.branches]
        assert ks == [3, 5, 7], f"分支核尺寸不对: {ks}"

    def test_padding_keeps_branch_sizes_aligned(self):
        """各分支 padding = d*(k-1)//2，输出空间尺寸必须一致，否则无法在通道维拼接。"""
        blk = MultiScaleBlock(8, 12, kernel_sizes=(3, 5, 7), stride=2)
        x = torch.randn(2, 8, 64, 64)
        h = blk.reduce(x)
        outs = [conv(h) for conv in blk.branches]
        shapes = {tuple(o.shape) for o in outs}
        assert len(shapes) == 1, f"分支输出尺寸不一致: {shapes}"
        assert tuple(outs[0].shape) == (2, 4, 32, 32)  # 12//3 = 4 通道, stride 2

    def test_output_shape_and_channels(self):
        blk = MultiScaleBlock(16, 30, kernel_sizes=(3, 5, 7), stride=2)
        y = blk(torch.randn(3, 16, 48, 48))
        assert y.shape == (3, 30, 24, 24)

    def test_output_channels_not_divisible_by_branches(self):
        """out_channels 不能被分支数整除时也要能跑（通道数是向上取整的）。"""
        blk = MultiScaleBlock(8, 32, kernel_sizes=(3, 5, 7), stride=2)
        y = blk(torch.randn(2, 8, 32, 32))
        assert y.shape[1] == 32

    def test_bottleneck_reduces_parameters(self):
        """1×1 降维应当显著减少参数量 —— 这是它存在的理由。

        这里全部用关键字参数：先前用位置参数写 `MultiScaleBlock(64,128,(3,5,7),2,True)`，
        后来在 stride 前面插入了 dilations，位置 2 就被当成空洞率，
        报出与参数数量毫无关系的 TypeError。多参数构造一律用关键字。
        """
        kw = {"kernel_sizes": (3, 5, 7), "stride": 2}
        with_b = sum(
            p.numel()
            for p in MultiScaleBlock(64, 128, use_bottleneck=True, **kw).parameters()
        )
        without = sum(
            p.numel()
            for p in MultiScaleBlock(64, 128, use_bottleneck=False, **kw).parameters()
        )
        assert with_b < without * 0.3, f"降维后 {with_b} 未显著少于 {without}"

    def test_no_bottleneck_variant_works(self):
        blk = MultiScaleBlock(8, 12, kernel_sizes=(3, 5, 7), stride=2, use_bottleneck=False)
        assert blk(torch.randn(2, 8, 32, 32)).shape == (2, 12, 16, 16)

    def test_default_kernels(self):
        assert DEFAULT_KERNEL_SIZES == (3, 5, 7)

    def test_custom_kernel_sizes(self):
        blk = MultiScaleBlock(8, 9, kernel_sizes=(3, 5), stride=2)
        assert [c.kernel_size[0] for c in blk.branches] == [3, 5]
        assert blk(torch.randn(2, 8, 32, 32)).shape == (2, 9, 16, 16)


class TestCheckpointResume:
    """检查点保存与续训。

    这里锁的是「每次训练都存档、有存档就接着训」这条要求。
    测试不碰真实数据 —— VisibilityTrainer.__init__ 只建模型、不建数据集，
    所以可以用临时目录完整验证存档/载入逻辑。
    """

    def _cfg(self, tmp_path, **model_over):
        model = {"input_size": (32, 32), "base_channels": 4, "latent_dim": 8, **model_over}
        return {
            "device": "cpu",
            "model": model,
            "train": {"checkpoint_dir": str(tmp_path / "ckpt")},
        }

    # --- resolve_resume_spec ---

    def test_resolve_none_when_empty(self, tmp_path):
        assert resolve_resume_spec("auto", tmp_path) is None

    def test_resolve_prefers_last_over_best(self, tmp_path):
        (tmp_path / "best.pt").write_bytes(b"x")
        (tmp_path / "last.pt").write_bytes(b"y")
        assert resolve_resume_spec("auto", tmp_path).name == "last.pt"

    def test_resolve_falls_back_to_best(self, tmp_path):
        (tmp_path / "best.pt").write_bytes(b"x")
        assert resolve_resume_spec("auto", tmp_path).name == "best.pt"

    def test_resolve_none_forces_fresh(self, tmp_path):
        (tmp_path / "last.pt").write_bytes(b"x")
        assert resolve_resume_spec("none", tmp_path) is None
        assert resolve_resume_spec(False, tmp_path) is None

    def test_resolve_explicit_path(self, tmp_path):
        p = tmp_path / "custom.pt"
        p.write_bytes(b"x")
        assert resolve_resume_spec(p, tmp_path) == p
        assert resolve_resume_spec(str(p), tmp_path) == p

    def test_resolve_missing_explicit_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="检查点不存在"):
            resolve_resume_spec(tmp_path / "nope.pt", tmp_path)

    # --- 原子写 ---

    def test_atomic_save_leaves_no_tmp_file(self, tmp_path):
        target = tmp_path / "sub" / "x.pt"
        _atomic_save({"a": 1}, target)
        assert target.exists()
        assert not list(target.parent.glob("*.tmp")), "残留了临时文件"

    def test_atomic_save_overwrites_safely(self, tmp_path):
        target = tmp_path / "x.pt"
        _atomic_save({"v": 1}, target)
        _atomic_save({"v": 2}, target)
        assert torch.load(target, weights_only=False)["v"] == 2

    # --- 续训 vs 全新 ---

    def test_fresh_training_ignores_existing_checkpoint(self, tmp_path):
        cfg = self._cfg(tmp_path)
        t1 = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
        _atomic_save(t1._make_payload(7, None, None, 0.5), tmp_path / "ckpt" / "last.pt")

        t2 = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
        assert t2.resume_path is None, "--fresh 不该捡起已有检查点"

    def test_auto_resume_picks_up_existing_checkpoint(self, tmp_path):
        cfg = self._cfg(tmp_path)
        t1 = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
        _atomic_save(t1._make_payload(7, None, None, 0.5), tmp_path / "ckpt" / "last.pt")

        t2 = VisibilityTrainer(cfg, project_root=tmp_path, resume="auto")
        assert t2.resume_path is not None
        assert t2.resume_path.name == "last.pt"

    def test_checkpoint_model_cfg_overrides_current_config(self, tmp_path):
        """续训时以检查点里的结构为准，否则权重加载会形状不匹配。

        这里构造一个「配置被改过」的场景，验证结构确实来自检查点。
        """
        cfg_a = self._cfg(tmp_path, base_channels=4)
        t = VisibilityTrainer(cfg_a, project_root=tmp_path, resume="none")
        _atomic_save(t._make_payload(3, None, None, 0.4), tmp_path / "ckpt" / "last.pt")

        # 用户改了配置，但检查点是旧结构
        cfg_b = self._cfg(tmp_path, base_channels=16)
        t2 = VisibilityTrainer(cfg_b, project_root=tmp_path, resume="auto")
        assert t2.model_cfg["base_channels"] == 4, "应以检查点的结构为准"

    # --- _restore 恢复完整状态 ---

    def test_restore_brings_back_weights_optimizer_and_epoch(self, tmp_path):
        cfg = self._cfg(tmp_path)
        t1 = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
        opt1 = torch.optim.AdamW(t1.model.parameters(), lr=1e-3)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=10)

        # 把权重改成一个可识别的值，再额外走一步让优化器留下状态
        with torch.no_grad():
            for p in t1.model.parameters():
                p.add_(0.123)
        loss = t1.model(torch.rand(2, 3, 32, 32)).sum()
        loss.backward()
        opt1.step()

        t1.history.train_loss = [0.9, 0.8, 0.7]
        t1.history.val_loss = [0.95, 0.85, 0.75]
        t1.history.best_val_loss = 0.75
        t1.history.best_epoch = 3
        t1.calibration = CalibrationStats(mean=0.01, std=0.003, p95=0.02, p99=0.03, n=10)
        expected = {k: v.clone() for k, v in t1.model.state_dict().items()}
        _atomic_save(t1._make_payload(3, opt1, sched1, 0.75), tmp_path / "ckpt" / "last.pt")

        # 新实例：结构随机初始化，随后应被检查点覆盖
        t2 = VisibilityTrainer(cfg, project_root=tmp_path, resume="auto")
        opt2 = torch.optim.AdamW(t2.model.parameters(), lr=1e-3)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
        epoch = t2._restore(opt2, sched2)

        assert epoch == 3
        assert t2.history.start_epoch == 3
        assert t2.history.train_loss == [0.9, 0.8, 0.7]
        assert t2.history.best_epoch == 3
        assert t2.history.best_val_loss == pytest.approx(0.75)
        assert t2.history.resumed_from is not None
        assert t2.calibration is not None and t2.calibration.n == 10
        for k, v in expected.items():
            assert torch.allclose(t2.model.state_dict()[k], v), f"{k} 权重没恢复"
        # 优化器状态也要回来，否则续训初期动量从零重建
        assert opt2.state_dict()["state"], "优化器状态为空，没恢复"

    def test_payload_records_config_snapshot(self, tmp_path):
        cfg = self._cfg(tmp_path)
        cfg["scoring"] = {"aggregation": {"power": -2.0}}
        t = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
        payload = t._make_payload(1, None, None, 0.1)
        assert payload["format_version"] == 2
        assert payload["model_cfg"]["latent_dim"] == 8
        assert payload["scoring_cfg"] == cfg["scoring"]
        assert "config_snapshot" in payload

    def test_scorer_prefers_explicit_scoring_config_then_checkpoint(self, tmp_path):
        from car_smart_assist.perception.visibility.scorer import VisibilityScorer

        cfg = self._cfg(tmp_path)
        cfg["scoring"] = {"aggregation": {"power": -2.0}}
        trainer = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
        checkpoint = tmp_path / "scorer.pt"
        _atomic_save(trainer._make_payload(1, None, None, 0.1), checkpoint)

        active_cfg = {"aggregation": {"power": -0.5}}
        scorer = VisibilityScorer.from_checkpoint(
            checkpoint, cfg=trainer.model_cfg, scoring_cfg=active_cfg
        )
        assert scorer.scoring_cfg == active_cfg

        fallback_scorer = VisibilityScorer.from_checkpoint(
            checkpoint, cfg=trainer.model_cfg
        )
        assert fallback_scorer.scoring_cfg == cfg["scoring"]


class TestAblationPresetsAreValid:
    """配置里的消融预设必须都能真的建出模型。

    加这组测试的直接原因：`kernels_3_5` 预设只改了 kernel_sizes 没改 dilations，
    长度对不上，消融跑到第 4 组才崩 —— 前面三组各花 7 分钟训练，
    等于白跑了 25 分钟。配置错误应该在跑之前、用一次 import 的代价发现。
    """

    @staticmethod
    def _presets():
        import yaml

        from car_smart_assist.perception.visibility.trainer import apply_config_overrides

        root = Path(__file__).resolve().parents[2]
        cfg_path = root / "configs/model/visibility.yaml"
        if not cfg_path.exists():
            pytest.skip("配置文件不存在")
        base = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["visibility"]
        presets = [{"tag": "baseline", "set": {}}] + list(base.get("ablations", []))
        return base, presets, apply_config_overrides

    def test_every_preset_builds_a_model(self):
        base, presets, apply_overrides = self._presets()
        assert presets, "没有读到任何预设"

        for p in presets:
            cfg = apply_overrides(base, p.get("set", {}))
            cfg.setdefault("model", {})["input_size"] = (144, 256)
            try:
                m = build_autoencoder(cfg["model"])
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"预设 {p['tag']} 建不出模型: {type(exc).__name__}: {exc}")
            assert m is not None

    def test_every_preset_does_a_forward_pass(self):
        base, presets, apply_overrides = self._presets()
        x = torch.rand(1, 3, 144, 256)
        for p in presets:
            cfg = apply_overrides(base, p.get("set", {}))
            cfg.setdefault("model", {})["input_size"] = (144, 256)
            m = build_autoencoder(cfg["model"])
            assert m(x).shape == x.shape, f"预设 {p['tag']} 输出形状不对"

    def test_presets_have_unique_tags(self):
        _, presets, _ = self._presets()
        tags = [p["tag"] for p in presets]
        assert len(tags) == len(set(tags)), f"预设 tag 有重名: {tags}"

    def test_every_preset_declares_its_overrides(self):
        _, presets, _ = self._presets()
        for p in presets:
            assert "tag" in p
            assert isinstance(p.get("set", {}), dict)


class TestSplitRefIndices:
    """train / val / calib / test 四划分。

    这里守两个属性：
    1. 四分互不重叠、并集为全集 —— 重叠会造成训练集泄漏
    2. fixed 模式可复现、random 模式每次不同 —— 前者是可比性的前提
    """

    def test_four_way_partition_is_disjoint_and_complete(self):
        tr, va, ca, te = split_ref_indices(1000, 0.8, 0.05, 0.05, 0.1, seed=42)
        all_idx = np.concatenate([tr, va, ca, te])
        assert len(all_idx) == len(set(all_idx.tolist())) == 1000
        assert set(all_idx.tolist()) == set(range(1000))

    def test_ratios_respected(self):
        tr, va, ca, te = split_ref_indices(1000, 0.8, 0.05, 0.05, 0.1, seed=0)
        assert len(tr) == 800
        assert len(va) == 50
        assert len(ca) == 50
        assert len(te) == 100

    def test_indices_sorted(self):
        """排序让 memmap 访问局部性更好，也便于与缓存行对齐。"""
        tr, va, ca, te = split_ref_indices(500, 0.8, 0.05, 0.05, 0.1, seed=1)
        for arr in (tr, va, ca, te):
            assert np.all(np.diff(arr) > 0)

    def test_fixed_mode_is_reproducible(self):
        a = split_ref_indices(300, seed=7, mode="fixed")
        b = split_ref_indices(300, seed=7, mode="fixed")
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(x, y)

    def test_different_seeds_give_different_splits(self):
        a = split_ref_indices(300, seed=1, mode="fixed")
        b = split_ref_indices(300, seed=2, mode="fixed")
        assert not np.array_equal(a[0], b[0])

    def test_random_mode_varies(self):
        """random 模式每次换种子 —— 这是被明确要求的行为，但代价是不可比。"""
        a = split_ref_indices(300, seed=42, mode="random")[0]
        b = split_ref_indices(300, seed=42, mode="random")[0]
        assert not np.array_equal(a, b), "random 模式应当每次不同"
        # 但划分本身仍然完整
        assert len(set(a.tolist())) == len(a)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="split_mode"):
            split_ref_indices(100, mode="banana")

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="比例之和"):
            split_ref_indices(100, 0.8, 0.1, 0.3)

    def test_test_split_is_disjoint_from_calib_used_for_thresholds(self):
        """关键安全属性：测试集绝不能与校准集重叠。

        校准集参与阈值标定，若测试集与它重叠，报出的泛化指标就是乐观的。
        """
        tr, va, ca, te = split_ref_indices(1000, 0.8, 0.05, 0.05, 0.1, seed=42)
        assert not (set(va.tolist()) & set(ca.tolist()))
        assert not (set(va.tolist()) & set(te.tolist()))
        assert not (set(va.tolist()) & set(tr.tolist()))
        assert not (set(ca.tolist()) & set(te.tolist()))
        assert not (set(tr.tolist()) & set(te.tolist()))

    # --- 按序列整组划分 ---

    def test_groups_never_cross_splits(self):
        """**核心属性**：同一序列的帧不能出现在两个集合里。

        ACDC 参考图来自视频，相邻帧近乎重复。一旦跨集合，
        校准集里就有训练样本的复制品，泛化指标虚高。
        """
        groups = [f"seq{i // 5}" for i in range(100)]  # 20 个序列，每组 5 帧
        tr, va, ca, te = split_ref_indices(
            100, 0.8, 0.05, 0.05, 0.1, seed=0, groups=groups
        )

        g_tr = {groups[i] for i in tr}
        g_va = {groups[i] for i in va}
        g_ca = {groups[i] for i in ca}
        g_te = {groups[i] for i in te}
        assert not (g_tr & g_va), "序列同时出现在训练与验证集"
        assert not (g_va & g_ca), "序列同时出现在验证与校准集"
        assert not (g_va & g_te), "序列同时出现在验证与测试集"
        assert not (g_tr & g_ca), "序列同时出现在训练与校准集"
        assert not (g_tr & g_te), "序列同时出现在训练与测试集"
        assert not (g_ca & g_te), "序列同时出现在校准与测试集"
        assert g_tr | g_va | g_ca | g_te == {f"seq{i}" for i in range(20)}

    def test_group_split_is_disjoint_and_complete(self):
        groups = [f"s{i // 5}" for i in range(200)]
        tr, va, ca, te = split_ref_indices(
            200, 0.8, 0.05, 0.05, 0.1, seed=3, groups=groups
        )
        all_idx = np.concatenate([tr, va, ca, te])
        assert len(all_idx) == len(set(all_idx.tolist())) == 200
        assert set(all_idx.tolist()) == set(range(200))

    def test_group_split_approximates_target_ratios(self):
        groups = [f"s{i // 20}" for i in range(1000)]  # 50 个等大组
        tr, va, ca, te = split_ref_indices(
            1000, 0.8, 0.05, 0.05, 0.1, seed=5, groups=groups
        )
        for got, want in ((len(tr), 800), (len(va), 50), (len(ca), 50), (len(te), 100)):
            assert abs(got - want) <= 40, f"划分大小偏离较多: {got} vs {want}"

    def test_group_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="groups 长度"):
            split_ref_indices(10, groups=["a", "b"])

    def test_group_split_requires_four_nonempty_partitions(self):
        with pytest.raises(ValueError, match="至少需要 4 个不同序列"):
            split_ref_indices(10, groups=["a", "a", "b", "b", "c", "c", "c", "c", "c", "c"])

    def test_image_split_rejects_too_few_samples_for_ratios(self):
        with pytest.raises(ValueError, match="不足以按当前比例"):
            split_ref_indices(10, 0.8, 0.05, 0.05, 0.1)

    def test_group_split_reproducible(self):
        groups = [f"s{i // 5}" for i in range(100)]
        a = split_ref_indices(100, 0.8, 0.05, 0.05, 0.1, seed=9, groups=groups)
        b = split_ref_indices(100, 0.8, 0.05, 0.05, 0.1, seed=9, groups=groups)
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(x, y)

    def test_sequence_of_extracts_sequence_name(self):
        p = "data/raw/acdc/rgb_anon/fog/train_ref/GOPR0475/GOPR0475_frame_000041_rgb_ref_anon.png"
        assert sequence_of(p) == "GOPR0475"


class TestVisibilityDataset:
    """数据集与缓存。

    这里锁的是一个真踩过的坑：DataLoader 用 spawn 启动 worker 时要 pickle 数据集，
    若把 memmap 数组挂在数据集上，pickle 会内联整个数组（443MB），
    在 Windows 上直接 `pickle data was truncated`。所以数据集只能持有路径。
    """

    def _make_cache(self, tmp_path, n=6, size=(16, 16)):
        arr = np.stack([
            np.full((*size, 3), i * 40 % 256, dtype=np.uint8) for i in range(n)
        ])
        p = tmp_path / "cache.npy"
        np.save(p, arr)
        return p

    def test_dataset_holds_only_path_not_array(self, tmp_path):
        cache = self._make_cache(tmp_path)
        paths = [tmp_path / f"{i}.png" for i in range(6)]
        ds = VisibilityImageDataset(paths, (16, 16), cache_path=cache)
        # 关键：不能有已打开的数组，否则 pickle 会内联数据
        assert ds._cache is None
        assert isinstance(ds.cache_path, type(tmp_path))

    def test_dataset_is_picklable_and_small(self, tmp_path):
        """pickle 出来的体积必须很小 —— 证明没有把数组带进去。"""
        import pickle

        cache = self._make_cache(tmp_path, n=200, size=(64, 64))
        paths = [tmp_path / f"{i}.png" for i in range(200)]
        ds = VisibilityImageDataset(paths, (64, 64), cache_path=cache)
        blob = pickle.dumps(ds)
        raw = 200 * 64 * 64 * 3
        assert len(blob) < raw * 0.05, (
            f"pickle 后 {len(blob)} 字节，接近原始数组 {raw} —— "
            "说明数组被内联进 pickle 了，spawn worker 时会截断"
        )

    def test_lazy_open_and_read(self, tmp_path):
        cache = self._make_cache(tmp_path, n=6)
        paths = [tmp_path / f"{i}.png" for i in range(6)]
        ds = VisibilityImageDataset(paths, (16, 16), cache_path=cache)
        x = ds[0]
        assert x.shape == (3, 16, 16)
        assert ds._cache is not None  # 首次访问后才打开

    def test_indices_select_rows(self, tmp_path):
        cache = self._make_cache(tmp_path, n=6)
        paths = [tmp_path / f"{i}.png" for i in range(6)]
        ds = VisibilityImageDataset(paths, (16, 16), cache_path=cache, indices=[2, 4])
        assert len(ds) == 2
        # 第 2 行填充值是 2*40=80 -> 80/255
        assert x_close(ds[0][0, 0, 0].item(), 80 / 255)

    def test_path_of_follows_indices(self, tmp_path):
        cache = self._make_cache(tmp_path, n=6)
        paths = [tmp_path / f"{i}.png" for i in range(6)]
        ds = VisibilityImageDataset(paths, (16, 16), cache_path=cache, indices=[3])
        assert ds.path_of(0) == paths[3]

    def test_missing_cache_raises_on_access(self, tmp_path):
        paths = [tmp_path / "0.png"]
        ds = VisibilityImageDataset(paths, (16, 16), cache_path=tmp_path / "nope.npy")
        with pytest.raises(FileNotFoundError, match="缓存不存在"):
            _ = ds[0]

    def test_works_without_cache(self, tmp_path):
        """不传缓存时实时解码 —— 小规模调试路径必须可用。"""
        img = structured_image(16, 16)
        p = tmp_path / "a.png"
        from PIL import Image

        Image.fromarray(img).save(p)
        ds = VisibilityImageDataset([p], (16, 16), cache_path=None)
        assert ds[0].shape == (3, 16, 16)


def x_close(a: float, b: float, tol: float = 0.02) -> bool:
    return abs(a - b) < tol


class TestDilatedBranches:
    """空洞卷积：扩大感受野的可选手段。"""

    def test_dilation_applied_to_branches(self):
        blk = MultiScaleBlock(8, 9, kernel_sizes=(3, 3, 3), dilations=(1, 2, 3), stride=2)
        assert [c.dilation[0] for c in blk.branches] == [1, 2, 3]

    def test_dilation_keeps_branch_sizes_aligned(self):
        """padding = d*(k-1)//2 让任意 (k,d) 组合输出同尺寸 —— 拼接的前提。

        若沿用常见的 k//2，dilation>1 时各分支尺寸会各不相同、拼接直接报错。
        """
        blk = MultiScaleBlock(8, 12, kernel_sizes=(3, 5, 7), dilations=(1, 2, 3), stride=2)
        h = blk.reduce(torch.randn(2, 8, 64, 64))
        shapes = {tuple(c(h).shape) for c in blk.branches}
        assert len(shapes) == 1, f"不同 (k,d) 分支输出尺寸不一致: {shapes}"

    def test_mixed_kernel_and_dilation_sizes_align(self):
        """k 与 d 都不同也要对齐。"""
        blk = MultiScaleBlock(8, 12, kernel_sizes=(3, 5, 7), dilations=(2, 1, 3), stride=2)
        y = blk(torch.randn(2, 8, 96, 96))
        assert y.shape == (2, 12, 48, 48)

    def test_dilation_does_not_change_param_count(self):
        """空洞卷积的价值：感受野变大而参数量不变。"""
        kw = {"kernel_sizes": (3, 3, 3), "stride": 2}
        a = MultiScaleBlock(64, 96, dilations=(1, 1, 1), **kw)
        b = MultiScaleBlock(64, 96, dilations=(1, 2, 4), **kw)
        assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())

    def test_dilated_333_is_cheaper_than_357_same_rf(self):
        """3×3 配 (1,2,3) 的有效感受野是 3/5/7，与 3/5/7 普通卷积相同，
        但参数量只有约三分之一 —— 这是用空洞替代大核的核心理由。"""
        big = MultiScaleBlock(64, 96, kernel_sizes=(3, 5, 7), dilations=(1, 1, 1), stride=2)
        dil = MultiScaleBlock(64, 96, kernel_sizes=(3, 3, 3), dilations=(1, 2, 3), stride=2)
        assert effective_kernel_size((3, 5, 7), (1, 1, 1)) == (3, 5, 7)
        assert effective_kernel_size((3, 3, 3), (1, 2, 3)) == (3, 5, 7)
        nb = sum(p.numel() for p in big.parameters())
        nd = sum(p.numel() for p in dil.parameters())
        assert nd < nb * 0.5, f"空洞版 {nd} 未显著少于大核版 {nb}"

    def test_effective_kernel_size_formula(self):
        assert effective_kernel_size((3,), (1,)) == (3,)
        assert effective_kernel_size((3,), (2,)) == (5,)
        assert effective_kernel_size((3,), (4,)) == (9,)
        assert effective_kernel_size((5,), (2,)) == (9,)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="dilations 长度"):
            MultiScaleBlock(8, 9, kernel_sizes=(3, 5, 7), dilations=(1, 2))

    def test_invalid_dilation_raises(self):
        with pytest.raises(ValueError, match="空洞率"):
            MultiScaleBlock(8, 9, kernel_sizes=(3, 5, 7), dilations=(1, 2, 0))

    def test_autoencoder_accepts_dilations(self):
        m = build_autoencoder(
            {"input_size": (144, 256), "kernel_sizes": [3, 3, 3], "dilations": [1, 2, 4]}
        )
        x = torch.rand(2, 3, 144, 256)
        assert m(x).shape == x.shape

    def test_config_dilations_reach_the_blocks(self):
        """配置里的 dilations 必须真的传到卷积层，否则配置是摆设。"""
        m = build_autoencoder(
            {"input_size": (144, 256), "kernel_sizes": [3, 3, 3], "dilations": [1, 2, 4]}
        )
        d0 = [c.dilation[0] for c in m.encoder[0].branches]
        assert d0 == [1, 2, 4], f"空洞率没传到第一层: {d0}"

    def test_dilations_none_means_all_ones(self):
        m = build_autoencoder({"input_size": (144, 256), "dilations": None})
        assert all(c.dilation[0] == 1 for c in m.encoder[0].branches)


class TestPreLatentProjection:
    """瓶颈前的 1×1 压缩 —— 过拟合的根源就在这里。

    实测：不压时两个全连接占模型约 90% 参数（9.46M），
    在 3605 张训练图上导致留出集误差比训练集高 45.6%。
    """

    def test_reduces_parameters_significantly(self):
        cfg = {"input_size": (144, 256)}
        big = sum(p.numel() for p in build_autoencoder({**cfg, "pre_latent_channels": 0}).parameters())
        small = sum(p.numel() for p in build_autoencoder({**cfg, "pre_latent_channels": 32}).parameters())
        assert small < big * 0.5, f"压缩后 {small} 未显著少于 {big}"

    def test_fc_layers_are_the_dominant_cost_when_uncompressed(self):
        """锁定「为什么需要压」：不压时全连接占绝对多数。"""
        m = build_autoencoder({"input_size": (144, 256), "pre_latent_channels": 0})
        fc = sum(p.numel() for p in m.to_latent.parameters()) + sum(
            p.numel() for p in m.from_latent.parameters()
        )
        tot = sum(p.numel() for p in m.parameters())
        assert fc / tot > 0.85, f"全连接占比仅 {fc / tot:.2f}，前提变了，需重新评估"

    def test_feat_dim_shrinks(self):
        big = build_autoencoder({"input_size": (144, 256), "pre_latent_channels": 0})
        small = build_autoencoder({"input_size": (144, 256), "pre_latent_channels": 32})
        assert small.feat_dim == big.feat_dim // 4

    def test_roundtrip_shape_still_correct(self):
        for plc in (0, 32, 64):
            m = build_autoencoder({"input_size": (144, 256), "pre_latent_channels": plc})
            x = torch.rand(2, 3, 144, 256)
            assert m(x).shape == x.shape, f"plc={plc} 形状不对"
            assert m.encode(x).shape == (2, 256)

    def test_zero_disables_compression(self):
        m = build_autoencoder({"input_size": (144, 256), "pre_latent_channels": 0})
        assert isinstance(m.pre_latent, torch.nn.Identity)


class TestReconZDisabled:
    """重建误差停用后，判定不得依赖它。

    实测重建误差与退化程度反相关（雾 -1.80 vs 清晰 -0.57），
    两个方向都不成立，因此默认不参与判定。
    """

    def test_default_gate_ignores_recon_z(self):
        gate = VisibilityGate()
        assert gate.thresholds.use_recon_z is False
        # 离谱的 z 值不应改变判定
        base = gate.judge(make_score(info=0.9, z=0.0))
        extreme = gate.judge(make_score(info=0.9, z=1000.0))
        assert base.level is extreme.level is VisibilityLevel.VISIBLE

    def test_extreme_z_with_low_info_still_blind_via_info(self):
        gate = VisibilityGate()
        v = gate.judge(make_score(info=0.05, z=-3.0))
        assert v.level is VisibilityLevel.BLIND
        assert "recon_extreme_low_info" not in v.triggered

    def test_can_be_reenabled_via_config(self):
        gate = VisibilityGate.from_config({"thresholds": {"use_recon_z": True, "z_blind": 5}})
        assert gate.thresholds.use_recon_z is True
        v = gate.judge(make_score(info=0.4, z=50.0))
        assert "recon_extreme_low_info" in v.triggered

    def test_config_defaults_to_disabled(self):
        assert VisibilityGate.from_config({}).thresholds.use_recon_z is False


# ---------------------------------------------------------------------------
# 自编码器
# ---------------------------------------------------------------------------


class TestConvAutoencoder:
    def test_multiscale_roundtrip_shape(self):
        m = build_autoencoder({"input_size": (144, 256)})
        x = torch.rand(2, 3, 144, 256)
        assert m(x).shape == x.shape

    def test_encode_shape(self):
        m = build_autoencoder({"input_size": (144, 256), "latent_dim": 64})
        assert m.encode(torch.rand(3, 3, 144, 256)).shape == (3, 64)

    def test_non_divisible_size_falls_back_to_interpolate(self):
        m = build_autoencoder({"input_size": (144, 256)})
        x = torch.rand(2, 3, 150, 270)
        assert m(x).shape == x.shape

    def test_plain_encoder_also_works(self):
        """单尺度对照分支必须可用 —— 消融实验依赖它。"""
        m = build_autoencoder({"input_size": (144, 256), "encoder_type": "plain"})
        x = torch.rand(2, 3, 144, 256)
        assert m(x).shape == x.shape

    def test_bottleneck_makes_multiscale_cheaper_than_plain(self):
        """多尺度 + 1×1 降维后，编码器参数量**反而低于**单尺度 4×4。

        这是 bottleneck 的价值所在：5×5 / 7×7 的参数量本来是 3×3 的
        2.8 倍与 5.4 倍，但先把通道压到 out/3 再做多尺度卷积之后，
        净效果比单尺度还省。所以「多尺度一定更重」是错的直觉。

        只比 encoder 部分 —— 总参数被 to_latent/from_latent 两个全连接层
        （约 9.4M）主导，比 total 看不出编码器的差异。
        """
        cfg = {"input_size": (144, 256)}
        ms = sum(
            p.numel()
            for p in build_autoencoder({**cfg, "encoder_type": "multiscale"}).encoder.parameters()
        )
        pl = sum(
            p.numel()
            for p in build_autoencoder({**cfg, "encoder_type": "plain"}).encoder.parameters()
        )
        assert ms < pl, f"含降维的多尺度编码器 {ms} 应少于单尺度 {pl}"

    def test_multiscale_without_bottleneck_is_much_heavier(self):
        """关掉降维后多尺度明显更重 —— 这正是默认开启降维的理由。"""
        cfg = {"input_size": (144, 256), "encoder_type": "multiscale"}
        with_b = sum(p.numel() for p in build_autoencoder({**cfg, "use_bottleneck": True}).encoder.parameters())
        without = sum(p.numel() for p in build_autoencoder({**cfg, "use_bottleneck": False}).encoder.parameters())
        assert without > with_b * 2, f"无降维 {without} 未显著重于含降维 {with_b}"

    def test_invalid_encoder_type_raises(self):
        with pytest.raises(ValueError, match="encoder_type"):
            build_autoencoder({"encoder_type": "banana"})

    def test_encoder_parameters_change_with_kernels(self):
        """换核尺寸必须真的改变结构，否则配置是摆设。"""
        a = build_autoencoder({"input_size": (144, 256), "kernel_sizes": [3, 5, 7]})
        b = build_autoencoder({"input_size": (144, 256), "kernel_sizes": [3]})
        assert sum(p.numel() for p in a.parameters()) != sum(p.numel() for p in b.parameters())


class TestReconstructionError:
    def test_returns_one_per_sample(self):
        m = build_autoencoder({"input_size": (144, 256)})
        errs = reconstruction_error(m, torch.rand(5, 3, 144, 256))
        assert len(errs) == 5
        assert all(e.mean >= 0 for e in errs)

    def test_block_stats_are_ordered(self):
        m = build_autoencoder({"input_size": (144, 256)})
        e = reconstruction_error(m, torch.rand(1, 3, 144, 256))[0]
        assert e.max_block >= e.p90_block >= 0
        assert e.std_block >= 0

    def test_identical_to_itself_gives_zero_error(self):
        """恒等映射应得 0 误差 —— 校验误差计算本身没写错。"""

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        e = reconstruction_error(Identity(), torch.rand(1, 3, 144, 256))[0]
        assert e.mean == pytest.approx(0.0, abs=1e-9)

    def test_reconstruction_mean_matches_per_image_mse(self):
        class ScaledIdentity(torch.nn.Module):
            def forward(self, x):
                return x * 0.8

        x = torch.rand(4, 3, 31, 47)
        actual = reconstruction_mean(ScaledIdentity(), x)
        expected = torch.nn.functional.mse_loss(
            x * 0.8, x, reduction="none"
        ).mean(dim=(1, 2, 3))
        torch.testing.assert_close(actual, expected)

    def test_vectorized_stats_match_per_sample_reference(self):
        class ScaledIdentity(torch.nn.Module):
            def forward(self, x):
                return x * 0.8

        x = torch.rand(3, 3, 31, 47)
        actual = reconstruction_error(ScaledIdentity(), x, block_grid=(2, 3))
        err = torch.nn.functional.mse_loss(x * 0.8, x, reduction="none").mean(1)
        expected = []
        for image_error in err:
            blocks = torch.nn.functional.adaptive_avg_pool2d(
                image_error[None, None], (2, 3)
            ).flatten()
            expected.append(
                (
                    float(image_error.mean()),
                    float(torch.quantile(blocks, 0.9)),
                    float(blocks.max()),
                    float(blocks.std()),
                )
            )
        for got, want in zip(actual, expected, strict=True):
            assert got.mean == pytest.approx(want[0])
            assert got.p90_block == pytest.approx(want[1])
            assert got.max_block == pytest.approx(want[2])
            assert got.std_block == pytest.approx(want[3])


# ---------------------------------------------------------------------------
# 信息量特征
# ---------------------------------------------------------------------------


class TestInformationFeatures:
    def test_structured_image_scores_high(self):
        """注意量纲：contrast 是 [0,1] 单位（函数内部会把图归一化），
        不是 0-255 单位。真实 ACDC 图的典型值是 0.24 左右。"""
        f = compute_information_features(structured_image())
        assert f.contrast > 0.05, f"contrast={f.contrast} 量纲疑似不对"
        assert f.entropy > 3
        assert f.edge_density > 0.0

    def test_contrast_scale_matches_feature_units(self):
        """回归测试：归一化尺度必须与特征的实际量纲一致。

        早期版本 contrast 尺度按 0-255 量纲填了 64.0，而特征实际是 [0,1] 量纲，
        导致归一化后恒为 ~0.004，几何平均被整体拽到接近 0 ——
        所有图的信息量分数一起塌陷、阈值全线失效，且不报任何错。
        """
        from car_smart_assist.perception.visibility.scorer import (
            _FEATURE_SCALES,
            normalize_features,
        )

        # 一张典型清晰图的 contrast 约 0.2~0.3，归一化后应落在合理区间
        f = compute_information_features(structured_image())
        n = normalize_features(f)
        assert 0.3 < n["contrast"] <= 1.0, (
            f"contrast 归一化后为 {n['contrast']:.4f} —— 尺度 {_FEATURE_SCALES['contrast']} "
            "与特征量纲不匹配"
        )

    def test_clear_image_gets_meaningful_information_score(self):
        """清晰图的信息量分数必须落在可判定区间，不能被尺度问题压到贴地。"""
        s = information_score(compute_information_features(structured_image()))
        assert s > 0.3, f"清晰图信息量仅 {s:.4f}，尺度配置有误"

    def test_uniform_image_scores_near_zero(self):
        """纯色图的信息量必须塌陷 —— 这是「看不见」最直接的信号。

        注意分数不会真的到 0：聚合用广义平均 + eps 下限（默认 0.10），
        所有维度都触底时总分就是 eps。这是刻意的 ——
        没有下限的话单维就能把分数钉死，反而制造误报。
        关键是它必须远低于 info_blind 阈值，仍然能被判 BLIND。
        """
        f = compute_information_features(uniform_image(0))
        assert f.contrast == pytest.approx(0.0, abs=1e-6)
        assert f.entropy == pytest.approx(0.0, abs=1e-6)
        s = information_score(f)
        assert s < 0.20, f"纯色图信息量 {s:.4f} 过高"
        assert s < GateThresholds().info_blind, "纯色图必须低于 BLIND 阈值"

    def test_ordering_structured_above_uniform(self):
        hi = information_score(compute_information_features(structured_image()))
        lo = information_score(compute_information_features(uniform_image(128)))
        assert hi > lo

    def test_accepts_both_uint8_and_float(self):
        img = structured_image()
        a = compute_information_features(img)
        b = compute_information_features(img.astype(np.float32) / 255.0)
        assert a.contrast == pytest.approx(b.contrast, rel=1e-4)

    def test_information_score_in_unit_range(self):
        for img in (uniform_image(0), uniform_image(255), structured_image()):
            s = information_score(compute_information_features(img))
            assert 0.0 <= s <= 1.0

    def test_scoring_parameters_can_be_overridden_from_config(self):
        image = structured_image()
        features = compute_information_features(
            image,
            {"features": {"edge_gradient_threshold": 2.0, "hf_cutoff_ratio": 0.0}},
        )
        assert features.edge_density == 0.0
        assert features.hf_ratio == pytest.approx(1.0)

        score = information_score(
            compute_information_features(uniform_image(0)),
            scoring_cfg={"aggregation": {"feature_floor": 0.9}},
        )
        assert score == pytest.approx(0.9)

    def test_frequency_mask_matches_shifted_reference(self):
        image = structured_image(31, 47)
        gray = (image.astype(np.float32) / 255.0) @ np.array(
            [0.299, 0.587, 0.114], dtype=np.float32
        )
        shifted = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
        power = np.abs(shifted) ** 2
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt(
            ((yy - cy) / max(h / 2, 1)) ** 2
            + ((xx - cx) / max(w / 2, 1)) ** 2
        )
        expected = power[radius > 0.5].sum() / power.sum()

        actual = compute_information_features(image).hf_ratio
        assert actual == pytest.approx(float(expected), abs=1e-7)


# ---------------------------------------------------------------------------
# 合成退化
# ---------------------------------------------------------------------------


class TestDegradation:
    @pytest.mark.parametrize("kind", ["fog", "darkness", "occlusion", "blur"])
    def test_each_kind_changes_the_image(self, kind):
        base = structured_image()
        out, spec = degrade(base, kind, 1.0, seed=0)
        assert out.shape == base.shape
        assert out.dtype == np.uint8
        assert spec.kind == kind
        assert not np.array_equal(out, base), f"{kind} 没有改变图像"

    @pytest.mark.parametrize("kind", ["fog", "darkness", "blur"])
    def test_information_decreases_with_severity(self, kind):
        """退化越重，信息量越低 —— 单调性是门控可信度的前提。"""
        base = structured_image()
        scores = [
            information_score(compute_information_features(degrade(base, kind, s, seed=0)[0]))
            for s in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        for a, b in zip(scores, scores[1:], strict=False):
            assert a >= b - 1e-6, f"{kind} 信息量非单调: {scores}"

    def test_severity_zero_is_identity(self):
        base = structured_image()
        out, _ = degrade(base, "fog", 0.0, seed=0)
        assert np.allclose(out, base), "severity=0 应当不改变图像"

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="未知退化类型"):
            degrade(structured_image(), "earthquake", 0.5)

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            degrade(structured_image(), "fog", 1.5)


# ---------------------------------------------------------------------------
# 判定契约（最重要的一组）
# ---------------------------------------------------------------------------


def make_score(info: float, z: float = 0.0) -> VisibilityScore:
    """构造一个指定信息量与 z 分数的打分结果。"""
    feat = InformationFeatures(contrast=1.0, entropy=1.0, edge_density=0.0, hf_ratio=0.0)
    return VisibilityScore(
        path="<synthetic>", recon_mean=0.0, recon_p90_block=0.0,
        recon_z=z, features=feat, information=info,
    )


class TestGateContract:
    """门控的判定契约。两条不对称的代价都在这里守住。

    测试用的信息量取值**从阈值本身推导**，不写死数字 ——
    阈值会随标定变化（已经从 0.12/0.30 改到 0.34/0.50 一次），
    写死的话每次重新标定都会产生一批与行为无关的假失败。
    """

    def setup_method(self):
        self.gate = VisibilityGate()
        t = self.gate.thresholds
        self.blind_info = t.info_blind * 0.5                          # 明确低于 BLIND 线
        self.degraded_info = (t.info_blind + t.info_degraded) / 2      # 落在两线之间
        self.visible_info = min(0.99, t.info_degraded + 0.2)           # 明确高于 DEGRADED 线

    # --- 方向一：看不见必须报警（漏报 = 安全问题）---

    def test_very_low_information_is_blind(self):
        v = self.gate.judge(make_score(info=0.005))
        assert v.level is VisibilityLevel.BLIND
        assert not v.allows_perception
        assert "info_low" in v.triggered

    def test_blind_blocks_perception_and_zeroes_confidence(self):
        v = self.gate.judge(make_score(info=0.005))
        assert v.confidence_multiplier == 0.0

    # --- 方向二：看得见不能被拦（误报 = 用户学会忽略它）---

    def test_high_information_is_never_blind(self):
        """**核心契约**：信息量够就绝不判 BLIND，无论重建误差多离谱。

        理由：隧道、施工区这类训练集里没见过的新奇场景重建误差会很高，
        但画面里明明有东西，判 BLIND 等于让系统在能工作时拒绝工作。
        普通异常检测把「新奇」当「危险」，在驾驶场景里是错的。
        """
        for z in (5.0, 10.0, 50.0, 1000.0, -1000.0):
            v = self.gate.judge(make_score(info=self.visible_info, z=z))
            assert v.level is not VisibilityLevel.BLIND, (
                f"高信息量(info={self.visible_info}) 在 z={z} 时被判 BLIND —— 违反核心契约"
            )

    def test_high_information_is_visible_regardless_of_z(self):
        v = self.gate.judge(make_score(info=self.visible_info, z=50.0))
        assert v.level is VisibilityLevel.VISIBLE
        assert v.allows_perception is True

    def test_clean_frame_is_visible(self):
        v = self.gate.judge(make_score(info=self.visible_info, z=0.5))
        assert v.level is VisibilityLevel.VISIBLE
        assert v.confidence_multiplier == 1.0

    def test_degraded_downgrades_confidence_but_allows(self):
        v = self.gate.judge(make_score(info=self.degraded_info, z=0.0))
        assert v.level is VisibilityLevel.DEGRADED
        assert v.allows_perception is True
        assert 0.0 < v.confidence_multiplier < 1.0

    # --- 校准缺失时的保守行为 ---

    def test_missing_calibration_blocks_blind_verdict(self):
        """没有零校准统计时不允许给 BLIND —— 单靠信息量一路信号风险太高。"""
        v = self.gate.judge(make_score(info=0.005, z=float("nan")))
        assert v.level is VisibilityLevel.DEGRADED
        assert "no_calibration" in v.triggered

    def test_missing_calibration_allowed_if_configured(self):
        g = VisibilityGate(require_calibration=False)
        v = g.judge(make_score(info=0.01, z=float("nan")))
        assert v.level is VisibilityLevel.BLIND

    # --- 阈值可配置 ---

    def test_thresholds_from_config(self):
        g = VisibilityGate.from_config(
            {
                "thresholds": {"info_blind": 0.5},
                "degraded_confidence_multiplier": 0.25,
            }
        )
        assert g.thresholds.info_blind == 0.5
        assert g.judge(make_score(info=0.4)).level is VisibilityLevel.BLIND

    def test_degraded_confidence_multiplier_from_config(self):
        g = VisibilityGate.from_config({"degraded_confidence_multiplier": 0.25})
        verdict = g.judge(make_score(info=0.4))
        assert verdict.level is VisibilityLevel.DEGRADED
        assert verdict.confidence_multiplier == 0.25

    def test_verdict_is_serializable(self):
        d = self.gate.judge(make_score(info=0.9, z=0.1)).to_dict()
        assert d["level"] == "visible"
        assert d["allows_perception"] is True

    def test_judge_many(self):
        vs = self.gate.judge_many([make_score(0.9), make_score(0.01)])
        assert [v.level for v in vs] == [VisibilityLevel.VISIBLE, VisibilityLevel.BLIND]
