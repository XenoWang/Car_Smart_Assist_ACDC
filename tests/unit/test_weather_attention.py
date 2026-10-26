"""新候选的梯度、阈值、缓存读取和恢复逻辑，不下载测试数据。"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from car_smart_assist.config.weather_attention import (
    attention_model_config,
    load_weather_attention_config,
    validate_weather_attention_config,
)
from car_smart_assist.perception.weather_attention import (
    WeatherAttention,
    WeatherAttentionPredictor,
    optimize_attention_inference,
)
from car_smart_assist.perception.weather_attention_training import (
    BackboneL2SP,
    attention_sampling_weights,
    balanced_thresholds,
    batch_images,
    learning_rate_factor,
    train_attention,
)
from car_smart_assist.perception.weather_training import WeatherSample


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def cfg():
    root = Path(__file__).resolve().parents[2]
    original = load_weather_attention_config(root / "configs/model/weather_attention.yaml")
    return replace(
        original,
        image_size=(64, 96),
        device="cpu",
        head={**original.head, "dimension": 16, "heads": 2},
        backbone={**original.backbone, "freeze_epochs": 1},
        train={**original.train, "epochs": 3, "warmup_epochs": 1, "batch_size": 4, "amp": False},
        evaluation={
            **original.evaluation,
            "batch_size": 4,
            "old_checkpoint": "old.pt",
            "asl_checkpoint": "asl.pt",
        },
        checkpoint_root="checkpoints",
        report_dir="reports",
        report_file="model_report.md",
    )


class ToyDataset:
    def __init__(self, signature, cfg):
        self.signature = signature
        self.targets = np.tile(np.eye(4, dtype=np.float32), (2, 1))
        self.mask = np.ones_like(self.targets)
        self.images = np.random.default_rng(42).integers(
            0, 256, (8, *cfg.image_size, 3), dtype=np.uint8
        )
        self.samples = [
            WeatherSample(
                tuple(map(int, row)),
                f"group{i // 4}",
                "day:rain15",
                path=Path(f"{i}.png") if i < 4 else None,
                archive_path=Path("test.zip") if i >= 4 else None,
                member=f"{i}.png" if i >= 4 else None,
            )
            for i, row in enumerate(self.targets)
        ]

    def __len__(self):
        return len(self.samples)


def test_config_keeps_500_epochs_and_separate_attributes():
    root = Path(__file__).resolve().parents[2]
    cfg = load_weather_attention_config(root / "configs/model/weather_attention.yaml")
    assert cfg.train["epochs"] == 500
    assert cfg.image_size == (384, 672)
    assert cfg.attributes == ("fog", "rain", "snow", "night")


@pytest.mark.parametrize(
    "field,key,value",
    [
        ("head", "dimension", 15),
        ("backbone", "feature_indices", [5, 12, 15]),
        ("train", "warmup_epochs", 3),
        ("train", "batch_size", 0),
        ("loss", "attribute_weights", {"rain": 1}),
    ],
)
def test_bad_config_rejected(cfg, field, key, value):
    with pytest.raises(ValueError):
        validate_weather_attention_config(
            replace(cfg, **{field: {**getattr(cfg, field), key: value}})
        )


def test_four_queries_have_finite_gradient_and_frozen_bn(cfg):
    model = WeatherAttention(cfg)
    model.set_training_epoch(2)
    before = {n: t.clone() for n, t in model.backbone.named_buffers()}
    model.train()
    result = model(torch.rand(2, 3, *cfg.image_size))
    assert result.shape == (2, 4)
    result.sum().backward()
    assert torch.isfinite(model.class_weights.grad).all()
    assert model.projections[0].weight.grad is not None
    assert not any(
        m.training for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)
    )
    assert all(torch.equal(value, before[name]) for name, value in model.backbone.named_buffers())
    assert not any(p.requires_grad for p in model.backbone[0].parameters())
    assert any(p.grad is not None for p in model.backbone[12].parameters())


def test_warmup_freezes_then_unlocks_late_features(cfg):
    model = WeatherAttention(cfg)
    model.set_training_epoch(1)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    model.set_training_epoch(2)
    assert any(p.requires_grad for p in model.backbone[7].parameters())
    assert not any(p.requires_grad for p in model.backbone[6].parameters())


def test_inference_fusion_preserves_training_model_and_logits(cfg):
    model = WeatherAttention(cfg).eval()
    optimized = optimize_attention_inference(model)
    image = torch.rand(2, 3, *cfg.image_size)
    with torch.inference_mode():
        assert torch.allclose(model(image), optimized(image), atol=1e-6, rtol=1e-5)
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in model.backbone.modules())
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in optimized.backbone.modules())


def test_vector_batch_preserves_pixels_and_channels_last(cfg):
    dataset = ToyDataset("train", cfg)
    indices = np.asarray([5, 2, 5, 1])
    result = batch_images(dataset, indices, torch.device("cpu"), cfg)
    expected = torch.from_numpy(dataset.images[indices]).permute(0, 3, 1, 2).float() / 255
    assert torch.equal(result, expected)
    assert result.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(
        batch_images(dataset, slice(1, 3), torch.device("cpu"), cfg),
        expected.new_tensor(dataset.images[1:3]).permute(0, 3, 1, 2) / 255,
    )


def test_sampling_allocates_configured_source_mass(cfg):
    ds = ToyDataset("train", cfg)
    weights = attention_sampling_weights(ds, cfg)
    assert float(weights[:4].sum() / weights.sum()) == pytest.approx(0.85)
    assert float(weights[4:].sum() / weights.sum()) == pytest.approx(0.15)


def test_thresholds_ignore_unknown_labels_and_reject_missing_class(cfg):
    ds = ToyDataset("calibration", cfg)
    values = ds.targets * 0.8 + 0.1
    thresholds, score = balanced_thresholds(ds, values, cfg)
    assert all(0.1 < value <= 0.9 for value in thresholds.values())
    assert score["score"] == pytest.approx(1)
    ds.targets[0, 0], ds.mask[0, 0] = 0, 0
    _, score = balanced_thresholds(ds, values, cfg)
    assert score["per_attribute"]["fog"]["source_balanced_ap"] == pytest.approx(1)
    ds.mask[:, 0] = 0
    with pytest.raises(ValueError, match="fog"):
        balanced_thresholds(ds, values, cfg)


def test_threshold_selection_uses_source_weights_and_ignores_unknowns(cfg):
    ds = ToyDataset("calibration", cfg)
    ds.targets = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        dtype=np.float32,
    )
    ds.mask = np.ones_like(ds.targets)
    ds.mask[3, 0] = 0
    ds.samples = [
        WeatherSample((1, 0, 0, 0), "a", "fog", path=Path("acdc-a.png")),
        WeatherSample((0, 1, 0, 0), "b", "rain", path=Path("acdc-b.png")),
        WeatherSample((0, 0, 1, 0), "c", "snow", archive_path=Path("pab.zip"), member="c.png"),
        WeatherSample((1, 0, 0, 1), "unknown", "night", archive_path=Path("pab.zip"), member="u.png"),
    ]
    # fog 的已知样本为：ACDC 正例(.7,w=.5)、ACDC 负例(.6,w=.5)、
    # PAB 负例(.8,w=1)；PAB 的 .99 标为 unknown，权重应为 0。
    cfg = replace(
        cfg,
        evaluation={**cfg.evaluation, "threshold_steps": 3, "ap_selection_fraction": 0.5,
                    "selection_weights": {name: 1.0 for name in cfg.attributes}},
    )
    probabilities = np.array(
        [[0.7, 0.9, 0.1, 0.1], [0.6, 0.1, 0.8, 0.1],
         [0.8, 0.1, 0.1, 0.9], [0.99, 0.8, 0.8, 0.8]],
        dtype=np.float32,
    )
    thresholds, score = balanced_thresholds(ds, probabilities, cfg)
    # 网格 .25/.5 都预测全部已知样本，F1=2*.5/(.5+2)=.4；.75 只选中负例。
    # 最优平台中心 .375 到 .25/.5 等距，规则取较小阈值；AP 仅在正例排名处
    # 累计：precision=.5/(1+.5)=1/3，recall 从 0 到 1，故 AP=1/3。
    assert thresholds["fog"] == pytest.approx(0.25)
    assert score["per_attribute"]["fog"]["source_balanced_max_f1"] == pytest.approx(0.4)
    assert score["per_attribute"]["fog"]["source_balanced_ap"] == pytest.approx(1 / 3)


def test_predictor_factory_does_not_alias_nested_caller_config(cfg, tmp_path):
    path = tmp_path / "best.pt"
    model = WeatherAttention(cfg).eval()
    torch.save({
        "format_version": 1,
        "model_config": attention_model_config(cfg),
        "model_state": model.state_dict(),
        "calibration_signature": "synthetic",
        "decision_thresholds": dict.fromkeys(cfg.attributes, 0.5),
    }, path)
    predictor = WeatherAttentionPredictor.from_checkpoint(path, cfg)
    original = predictor.model.cfg.head["dimension"]
    cfg.head["dimension"] = original + 2
    assert predictor.model.cfg.head["dimension"] == original


def test_schedule_uses_configured_horizon(cfg):
    assert learning_rate_factor(1, cfg) == 1
    assert learning_rate_factor(3, cfg) == pytest.approx(cfg.train["minimum_lr_ratio"])
    assert learning_rate_factor(2, cfg) < learning_rate_factor(1, cfg)


def test_checkpoint_requires_calibration_and_roundtrip(cfg, tmp_path):
    model = WeatherAttention(cfg).eval()
    path = tmp_path / "best.pt"
    payload = {
        "format_version": 1,
        "model_config": attention_model_config(cfg),
        "model_state": model.state_dict(),
    }
    torch.save(payload, path)
    with pytest.raises(ValueError, match="校准"):
        WeatherAttentionPredictor.from_checkpoint(path, cfg)
    payload.update(
        calibration_signature="test-only", decision_thresholds=dict.fromkeys(cfg.attributes, 0.5)
    )
    torch.save(payload, path)
    loaded = WeatherAttentionPredictor.from_checkpoint(path, cfg)
    image = torch.rand(1, 3, *cfg.image_size)
    with torch.inference_mode():
        assert torch.allclose(model(image), loaded.model(image))
    result = loaded.predict(np.zeros((*cfg.image_size, 3), np.uint8))
    assert set(result.probabilities) == set(cfg.attributes)


@pytest.mark.parametrize(
    "mode", ["explicit", "default", "explicit_override", "checkpoint_override"]
)
def test_pipeline_selects_explicit_or_default_candidate(cfg, tmp_path, mode):
    from dataclasses import asdict

    import yaml

    from car_smart_assist.inference.pipeline import InferencePipeline

    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump({"weather_attention": asdict(cfg)}), encoding="utf-8")
    checkpoint = tmp_path / cfg.checkpoint_root / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    initialized = WeatherAttention(cfg).eval()
    with torch.no_grad():
        for parameter in initialized.parameters():
            parameter.zero_()
    torch.save(
        {
            "format_version": 1,
            "model_config": attention_model_config(cfg),
            "model_state": initialized.state_dict(),
            "calibration_signature": "test-only",
            "decision_thresholds": dict.fromkeys(cfg.attributes, 0.5),
        },
        checkpoint,
    )
    override = None
    if mode != "explicit":
        route = tmp_path / "configs/model/weather_default.yaml"
        route.parent.mkdir(parents=True)
        route.write_text(
            yaml.safe_dump(
                {
                    "weather_default": {
                        "config": "missing.yaml" if mode == "explicit_override" else str(path),
                        "checkpoint": str(checkpoint),
                    }
                }
            ),
            encoding="utf-8",
        )
    if mode == "checkpoint_override":
        override = checkpoint.with_name("override.pt")
        payload = torch.load(checkpoint, weights_only=True)
        payload["model_state"]["class_bias"].fill_(float(torch.log(torch.tensor(3.0))))
        torch.save(payload, override)
    pipeline = InferencePipeline.from_config(
        {},
        project_root=tmp_path,
        require_visibility=False,
        weather_config=path if mode in ("explicit", "explicit_override") else None,
        weather_checkpoint=override,
        device="cpu",
    )
    assert isinstance(pipeline.weather_predictor, WeatherAttentionPredictor)
    assert pipeline.weather_predictor.device.type == "cpu"
    if override:
        assert torch.equal(
            pipeline.weather_predictor.model.class_bias,
            torch.full((4,), float(torch.log(torch.tensor(3.0)))),
        )
    result = pipeline.run(np.full((80, 120, 3), 100, np.uint8))
    assert result.weather is not None
    assert set(result.weather.probabilities) == set(cfg.attributes)
    if override:
        baseline = InferencePipeline.from_config(
            {}, project_root=tmp_path, require_visibility=False, device="cpu"
        ).run(np.full((80, 120, 3), 100, np.uint8))
        assert baseline.weather is not None
        for attribute in cfg.attributes:
            assert baseline.weather.probabilities[attribute] == pytest.approx(0.5, abs=1e-6)
            assert result.weather.probabilities[attribute] == pytest.approx(0.75, abs=1e-6)


@pytest.mark.parametrize(
    "route", [None, {"config": "only-config.yaml"}, {"config": "", "checkpoint": "best.pt"}]
)
def test_invalid_default_route_does_not_silently_load_a_legacy_model(tmp_path, route):
    import yaml

    from car_smart_assist.inference.pipeline import InferencePipeline

    path = tmp_path / "configs/model/weather_default.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"weather_default": route}), encoding="utf-8")
    with pytest.raises(ValueError, match="默认天气路由"):
        InferencePipeline.from_config({}, project_root=tmp_path, require_visibility=False)


def test_missing_routed_checkpoint_is_reported(cfg, tmp_path):
    from dataclasses import asdict

    import yaml

    from car_smart_assist.inference.pipeline import InferencePipeline

    path = tmp_path / "configs/model/weather_default.yaml"
    path.parent.mkdir(parents=True)
    (tmp_path / "candidate.yaml").write_text(
        yaml.safe_dump({"weather_attention": asdict(cfg)}), encoding="utf-8"
    )
    path.write_text(
        yaml.safe_dump(
            {"weather_default": {"config": "candidate.yaml", "checkpoint": "missing.pt"}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="天气权重"):
        InferencePipeline.from_config({}, project_root=tmp_path, require_visibility=False)


@pytest.mark.parametrize("coefficient,use_swad", [(0.0, False), (0.0001, False), (0.0, True)])
def test_interruption_resume_matches_uninterrupted_training(
    cfg, tmp_path, monkeypatch, coefficient, use_swad
):
    import car_smart_assist.perception.weather_attention_training as training

    cfg = replace(cfg, train={**cfg.train, "l2_sp": coefficient})
    if use_swad:
        cfg = replace(
            cfg,
            train={
                **cfg.train,
                "swad": {
                    "enabled": True,
                    "n_converge": 2,
                    "n_tolerance": 3,
                    "tolerance_ratio": 0.3,
                    "start_epoch": 2,
                },
            },
        )
        monkeypatch.setattr(training, "source_balanced_bce", lambda *_: 1.0)

    root = Path(__file__).resolve().parents[2]
    weights = (
        root / "artifacts/checkpoints/weather_attention/pretrained/mobilenet_v3_large-5c1a4163.pth"
    )
    assert weights.exists(), "先准备项目内预训练权重；单测不联网下载"
    monkeypatch.setattr(training, "ensure_pretrained_weights", lambda *_: weights)
    datasets = {
        name: ToyDataset(name, cfg) for name in ("train", "validation", "calibration", "test")
    }
    for folder in (tmp_path / "complete", tmp_path / "resumed"):
        folder.mkdir()
        (folder / "old.pt").write_bytes(b"original preserved")
        (folder / "asl.pt").write_bytes(b"asl preserved")
    full, expected = train_attention(cfg, tmp_path / "complete", datasets, {"signature": "split"})
    writer = training.write_json

    def interrupt(path, value):
        if path.name == "progress.json" and value.get("epoch") == 2:
            raise RuntimeError("simulated interruption after checkpoint")
        return writer(path, value)

    monkeypatch.setattr(training, "write_json", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        train_attention(cfg, tmp_path / "resumed", datasets, {"signature": "split"})
    monkeypatch.setattr(training, "write_json", writer)
    # 模拟 best 与 last 两次提交之间中断，last 仍能找到自己的 best 备份。
    import shutil

    best = tmp_path / "resumed/checkpoints" / ("single_best.pt" if use_swad else "best.pt")
    shutil.copy2(best, best.with_name("best.rollback.pt"))
    best.write_bytes(b"interrupted replacement")
    resumed, actual = train_attention(cfg, tmp_path / "resumed", datasets, {"signature": "split"})
    assert actual["best_epoch"] == expected["best_epoch"]
    assert all(
        torch.equal(full.state_dict()[key], resumed.state_dict()[key]) for key in full.state_dict()
    )
    last = torch.load(tmp_path / "resumed/checkpoints/last.pt", weights_only=True)
    assert [r["epoch"] for r in last["history"]] == [1, 2, 3]
    assert last["training_complete"]
    if use_swad:
        assert actual["swad"]["averaged_updates"] == 4
        assert last["swad_state"]["step"] == 6
        for name in ("single_best", "swad", "best"):
            path = tmp_path / f"resumed/checkpoints/{name}.pt"
            predictor = WeatherAttentionPredictor.from_checkpoint(path, cfg)
            assert predictor.predict(datasets["test"].images[0]).probabilities
        plain_root = tmp_path / "plain"
        plain_root.mkdir()
        (plain_root / "old.pt").write_bytes(b"original preserved")
        (plain_root / "asl.pt").write_bytes(b"asl preserved")
        plain = replace(
            cfg, train={key: value for key, value in cfg.train.items() if key != "swad"}
        )
        train_attention(plain, plain_root, datasets, {"signature": "split"})
        plain_last = torch.load(plain_root / "checkpoints/last.pt", weights_only=True)
        assert all(
            torch.equal(value, plain_last["model_state"][name])
            for name, value in last["model_state"].items()
        )
        assert [row["sampling_sha256"] for row in last["history"]] == [
            row["sampling_sha256"] for row in plain_last["history"]
        ]
    changed = replace(cfg, train={**cfg.train, "learning_rate": 0.1})
    with pytest.raises(ValueError, match="配方"):
        train_attention(changed, tmp_path / "resumed", datasets, {"signature": "split"})


def test_l2_sp_anchors_only_active_backbone_and_has_correct_gradient(cfg):
    model = WeatherAttention(cfg)
    regularizer = BackboneL2SP(model, 0.001)
    assert regularizer().item() == 0
    model.set_training_epoch(2)
    named = dict(model.backbone.named_parameters())
    name, parameter = next((name, p) for name, p in named.items() if p.requires_grad)
    reference = regularizer.references[name].clone()
    with torch.no_grad():
        parameter.add_(0.5)
        model.class_weights.add_(10)
    penalty = regularizer()
    assert penalty.item() == pytest.approx(0.001 * 0.5 * parameter.numel() * 0.25, rel=1e-5)
    penalty.backward()
    assert torch.allclose(parameter.grad, torch.full_like(parameter, 0.0005), atol=1e-7)
    assert model.class_weights.grad is None
    assert torch.equal(regularizer.references[name], reference)


@pytest.mark.parametrize(
    "key,value", [("l2_sp", -0.1), ("l2_sp", float("nan")), ("schedule_epochs", 2)]
)
def test_invalid_ablation_training_parameters_rejected(cfg, key, value):
    with pytest.raises(ValueError):
        validate_weather_attention_config(replace(cfg, train={**cfg.train, key: value}))


def test_shorter_budget_keeps_original_learning_rate_schedule(cfg):
    full = replace(cfg, train={**cfg.train, "epochs": 5})
    short = replace(cfg, train={**cfg.train, "schedule_epochs": 5})
    validate_weather_attention_config(short)
    assert learning_rate_factor(3, short) == learning_rate_factor(3, full)
    assert learning_rate_factor(3, short) > cfg.train["minimum_lr_ratio"]


def test_continuation_preserves_state_reopens_swad_and_resumes_exactly(cfg, tmp_path, monkeypatch):
    import hashlib

    import car_smart_assist.perception.weather_attention_training as training

    weights = (
        Path(__file__).resolve().parents[2]
        / "artifacts/checkpoints/weather_attention/pretrained/mobilenet_v3_large-5c1a4163.pth"
    )
    monkeypatch.setattr(training, "ensure_pretrained_weights", lambda *_: weights)
    monkeypatch.setattr(training, "source_balanced_bce", lambda *_: 1.0)
    (tmp_path / "old.pt").write_bytes(b"old baseline")
    (tmp_path / "asl.pt").write_bytes(b"ASL baseline")
    options = {
        "enabled": True,
        "n_converge": 2,
        "n_tolerance": 3,
        "tolerance_ratio": 0.3,
        "start_epoch": 2,
    }
    source_cfg = replace(
        cfg,
        checkpoint_root="source",
        report_dir="source_report",
        report_file="source_report/REPORT.md",
        train={**cfg.train, "schedule_epochs": 6, "swad": options},
    )
    datasets = {
        name: ToyDataset(name, cfg) for name in ("train", "validation", "calibration", "test")
    }
    train_attention(source_cfg, tmp_path, datasets, {"signature": "split"})
    source_path = tmp_path / "source/last.pt"
    origin = torch.load(source_path, weights_only=True)
    origin["swad_state"]["valley"]["dead"] = True
    torch.save(origin, source_path)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    stage = {
        "checkpoint": "source/last.pt",
        "checkpoint_sha256": source_hash,
        "single_best": "source/single_best.pt",
        "single_best_sha256": origin["best_sha256"],
        "source_epoch": 3,
        "additional_epochs": 3,
        "start_factor": origin["history"][-1]["head_lr"] / cfg.train["learning_rate"],
    }
    extended = replace(
        source_cfg,
        checkpoint_root="full",
        report_dir="full_report",
        report_file="full_report/REPORT.md",
        train={
            **source_cfg.train,
            "epochs": 6,
            "continuation": stage,
            "swad": {**options, "start_epoch": 4},
        },
    )
    validate_weather_attention_config(extended)
    full, expected = train_attention(extended, tmp_path, datasets, {"signature": "split"})
    other = replace(
        extended,
        checkpoint_root="resume",
        report_dir="resume_report",
        report_file="resume_report/REPORT.md",
    )
    writer = training.write_json

    def interrupt(path, value):
        if path.name == "progress.json" and value.get("epoch") == 5:
            raise RuntimeError("stage interruption")
        return writer(path, value)

    monkeypatch.setattr(training, "write_json", interrupt)
    with pytest.raises(RuntimeError, match="stage interruption"):
        train_attention(other, tmp_path, datasets, {"signature": "split"})
    monkeypatch.setattr(training, "write_json", writer)
    resumed, actual = train_attention(other, tmp_path, datasets, {"signature": "split"})
    assert all(
        torch.equal(value, resumed.state_dict()[name]) for name, value in full.state_dict().items()
    )
    assert expected["swad"] == actual["swad"]
    assert actual["swad"]["start_epoch"] == 4 and actual["swad"]["averaged_updates"] == 6
    assert actual["swad"]["start_step"] == 7 and actual["swad"]["end_step"] == 12
    last = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert [row["epoch"] for row in last["history"]] == list(range(1, 7))
    assert last["history"][3]["head_lr"] == pytest.approx(origin["history"][-1]["head_lr"])
    assert last["history"][-1]["head_lr"] == pytest.approx(
        cfg.train["learning_rate"] * cfg.train["minimum_lr_ratio"]
    )
    generator = torch.Generator()
    generator.set_state(origin["generator_state"])
    next_order = torch.multinomial(
        attention_sampling_weights(datasets["train"], cfg), 8, replacement=True, generator=generator
    ).numpy()
    assert last["history"][3]["sampling_sha256"] == hashlib.sha256(next_order.tobytes()).hexdigest()
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    with pytest.raises(ValueError, match="独立目录"):
        train_attention(
            replace(extended, checkpoint_root="source"),
            tmp_path,
            datasets,
            {"signature": "split"},
            fresh=True,
        )
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize("channels_last", [False, True])
def test_attention_checkpoint_factory_matches_public_copy_and_isolates_config(
    cfg, tmp_path, monkeypatch, channels_last
):
    import car_smart_assist.perception.weather_attention as attention_module

    cfg = replace(
        cfg,
        device="cpu",
        train={**cfg.train, "channels_last": channels_last},
    )
    source = WeatherAttention(cfg)
    if channels_last:
        source.to(memory_format=torch.channels_last)
    source.train()
    original_state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    original_modes = {name: module.training for name, module in source.named_modules()}
    reference = optimize_attention_inference(source)

    assert source.training
    assert {name: module.training for name, module in source.named_modules()} == original_modes
    assert all(
        torch.equal(value, source.state_dict()[name]) for name, value in original_state.items()
    )

    checkpoint = tmp_path / "tiny_calibrated.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": attention_model_config(cfg),
            "model_state": source.state_dict(),
            "calibration_signature": "synthetic-test-only",
            "decision_thresholds": dict.fromkeys(cfg.attributes, 0.5),
        },
        checkpoint,
    )

    real_deepcopy = attention_module.copy.deepcopy

    def reject_whole_model_deepcopy(value, memo=None):
        if isinstance(value, WeatherAttention):
            raise AssertionError("checkpoint factory must not deepcopy the whole model")
        return real_deepcopy(value, memo)

    monkeypatch.setattr(attention_module.copy, "deepcopy", reject_whole_model_deepcopy)
    predictor = WeatherAttentionPredictor.from_checkpoint(checkpoint, cfg, device="cpu")

    assert predictor.cfg is cfg
    assert predictor.model.cfg is not cfg
    assert predictor.model.cfg.train is not cfg.train
    assert predictor.model.cfg.backbone is not cfg.backbone
    assert predictor.model.cfg.train["channels_last"] is channels_last
    assert not predictor.model.training
    reference_state = reference.state_dict()
    actual_state = predictor.model.state_dict()
    assert actual_state.keys() == reference_state.keys()
    assert all(torch.equal(actual_state[name], reference_state[name]) for name in actual_state)
    four_dimensional = next(p for p in predictor.model.parameters() if p.ndim == 4)
    assert four_dimensional.is_contiguous(
        memory_format=torch.channels_last if channels_last else torch.contiguous_format
    )

    image = np.arange(64 * 96 * 3, dtype=np.uint32).reshape(64, 96, 3)
    image = (image % 256).astype(np.uint8)
    input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    with torch.inference_mode():
        expected_probabilities = reference(input_tensor).float().sigmoid()[0].numpy()
    result = predictor.predict(image)
    expected = dict(zip(cfg.attributes, expected_probabilities.tolist(), strict=True))
    assert result.probabilities == expected
    assert result.decisions == {
        name: probability >= 0.5 for name, probability in expected.items()
    }


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要可用 CUDA GPU")
@pytest.mark.parametrize("channels_last", [False, True])
def test_attention_checkpoint_factory_matches_public_copy_on_cuda(
    cfg, tmp_path, monkeypatch, channels_last
):
    import car_smart_assist.perception.weather_attention as attention_module

    cfg = replace(
        cfg,
        device="cuda",
        train={**cfg.train, "channels_last": channels_last},
    )
    source = WeatherAttention(cfg).to("cuda")
    if channels_last:
        source.to(memory_format=torch.channels_last)
    source.train()
    original_state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    original_modes = {name: module.training for name, module in source.named_modules()}
    reference = optimize_attention_inference(source)  # 旧工厂也在目标设备上融合。

    assert source.training
    assert {name: module.training for name, module in source.named_modules()} == original_modes
    assert all(
        torch.equal(value, source.state_dict()[name]) for name, value in original_state.items()
    )

    checkpoint = tmp_path / "tiny_calibrated_cuda.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": attention_model_config(cfg),
            "model_state": source.state_dict(),
            "calibration_signature": "synthetic-test-only",
            "decision_thresholds": dict.fromkeys(cfg.attributes, 0.5),
        },
        checkpoint,
    )

    real_deepcopy = attention_module.copy.deepcopy

    def reject_whole_model_deepcopy(value, memo=None):
        if isinstance(value, WeatherAttention):
            raise AssertionError("checkpoint factory must not deepcopy the whole model")
        return real_deepcopy(value, memo)

    monkeypatch.setattr(attention_module.copy, "deepcopy", reject_whole_model_deepcopy)
    predictor = WeatherAttentionPredictor.from_checkpoint(checkpoint, cfg, device="cuda")

    assert predictor.device.type == "cuda"
    assert predictor.model.cfg is not cfg
    assert predictor.model.cfg.train is not cfg.train
    assert predictor.model.cfg.backbone is not cfg.backbone
    assert predictor.model.cfg.train["channels_last"] is channels_last
    assert not predictor.model.training
    reference_state = reference.state_dict()
    actual_state = predictor.model.state_dict()
    assert actual_state.keys() == reference_state.keys()
    assert all(torch.equal(actual_state[name], reference_state[name]) for name in actual_state)
    four_dimensional = next(p for p in predictor.model.parameters() if p.ndim == 4)
    assert four_dimensional.device.type == "cuda"
    assert four_dimensional.is_contiguous(
        memory_format=torch.channels_last if channels_last else torch.contiguous_format
    )

    image = np.arange(64 * 96 * 3, dtype=np.uint32).reshape(64, 96, 3)
    image = (image % 256).astype(np.uint8)
    input_tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to("cuda", dtype=torch.float32)
        .div_(255.0)
    )
    with torch.inference_mode():
        expected_probabilities = reference(input_tensor).float().sigmoid()[0].cpu().numpy()
    result = predictor.predict(image)
    expected = dict(zip(cfg.attributes, expected_probabilities.tolist(), strict=True))
    assert result.probabilities == expected
    assert result.decisions == {
        name: probability >= 0.5 for name, probability in expected.items()
    }
