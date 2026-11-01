"""Weather report targets cannot replace synthetic deployed evidence or weights."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest
import yaml


def _synthetic_project(
    root: Path, *, deployed_report_file: str = "shared/REPORT.json.tmp"
) -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[2]
    deployed = yaml.safe_load((repo / "configs/model/weather_attention.yaml").read_text(encoding="utf-8"))
    deployed["weather_attention"]["checkpoint_root"] = "deployed/weights"
    deployed["weather_attention"]["report_dir"] = "deployed/reports"
    deployed["weather_attention"]["report_file"] = deployed_report_file
    deployed["weather_attention"]["evaluation"]["summary_file"] = "deployed/reports/summary.md"
    config_path = root / "configs/model/deployed.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(deployed, sort_keys=False), encoding="utf-8")
    checkpoint = root / "deployed/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic deployment checkpoint\x00")
    report = root / deployed_report_file
    report.parent.mkdir(parents=True)
    report.write_bytes(b"synthetic deployment report\x00")
    (root / "configs/model/weather_default.yaml").write_text(
        yaml.safe_dump({"weather_default": {"config": "configs/model/deployed.yaml",
                                            "checkpoint": "deployed/weights/best.pt"}},
                       sort_keys=False), encoding="utf-8"
    )
    return report, checkpoint


@pytest.mark.parametrize("target_kind", ["summary", "json-temp", "checkpoint", "report-file"])
def test_weather_report_publishers_reject_deployed_targets_without_touching_assets(
    tmp_path, monkeypatch, target_kind
):
    from car_smart_assist.perception import weather_attention_training as training
    from car_smart_assist.perception.weather_attention_training import require_independent_weather_reports

    deployed_report, checkpoint = _synthetic_project(tmp_path)
    candidates = {
        "summary": "deployed/reports/summary.md",
        "json-temp": "shared/REPORT.json",
        "checkpoint": "deployed/weights/best.pt",
        "report-file": "shared/REPORT.json.tmp",
    }
    original_targets = [deployed_report, checkpoint]
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in original_targets
    }
    repo = Path(__file__).resolve().parents[2]
    from car_smart_assist.config.weather_attention import load_weather_attention_config
    base = load_weather_attention_config(repo / "configs/model/weather_attention.yaml")
    report_file = candidates[target_kind] if target_kind in {"report-file", "checkpoint"} else "trial/reports/REPORT.md"
    cfg = replace(
        base, report_dir="trial/reports",
        report_file=report_file,
        checkpoint_root="trial/weights",
        evaluation={**base.evaluation,
                    "summary_file": candidates[target_kind] if target_kind in {"summary", "json-temp"} else "trial/reports/summary.md"},
    )
    if target_kind == "summary":
        # The summary itself is inside a protected deployment directory.
        with pytest.raises(ValueError, match="受保护") as raised:
            require_independent_weather_reports(cfg, tmp_path)
        assert str((tmp_path / candidates[target_kind]).resolve()) in str(raised.value)
        with pytest.raises(ValueError, match="受保护"):
            training.write_json(tmp_path / cfg.evaluation["summary_file"],
                                {"synthetic": True}, cfg=cfg, root=tmp_path)
    elif target_kind == "json-temp":
        # The final JSON path is safe; write_json's actual .json.tmp target is
        # the protected deployment report at shared/REPORT.json.tmp.
        summary = tmp_path / cfg.evaluation["summary_file"]
        assert summary == tmp_path / "shared/REPORT.json"
        with pytest.raises(ValueError, match="受保护") as raised:
            training.write_json(summary, {"synthetic": True}, cfg=cfg, root=tmp_path)
        assert str(deployed_report.resolve()) in str(raised.value)
        assert not summary.exists()
    elif target_kind == "report-file":
        with pytest.raises(ValueError, match="受保护") as raised:
            require_independent_weather_reports(cfg, tmp_path)
        assert str(deployed_report.resolve()) in str(raised.value)
        with pytest.raises(ValueError, match="受保护"):
            training.model_report(cfg, tmp_path, {"status": "paused", "epoch": 0})
    else:
        with pytest.raises(ValueError, match="受保护") as raised:
            require_independent_weather_reports(cfg, tmp_path)
        assert str(checkpoint.resolve()) in str(raised.value)
    if target_kind != "json-temp":
        monkeypatch.setattr(training, "select_device", lambda *_: pytest.fail("device selection began before output guard"))
        monkeypatch.setattr(training, "ensure_pretrained_weights", lambda *_: pytest.fail("weights loaded before output guard"))
        monkeypatch.setattr(training, "WeatherAttention", lambda *_: pytest.fail("model initialized before output guard"))
        with pytest.raises(ValueError, match="受保护"):
            training.train_attention(cfg, tmp_path, {}, {}, fresh=True)
    for path, (content, mtime_ns) in before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime_ns


def test_weather_report_guard_allows_independent_summary_and_mps_is_rejected_early(tmp_path):
    from car_smart_assist.config.weather_attention import load_weather_attention_config
    from car_smart_assist.perception.weather_attention_training import (
        model_report,
        require_independent_weather_reports,
    )

    deployed_report, checkpoint = _synthetic_project(tmp_path)
    protected_before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (deployed_report, checkpoint)
    }
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs/model/weather_attention.yaml")
        .read_text(encoding="utf-8")
    )
    raw["weather_attention"]["device"] = "mps"
    config = tmp_path / "configs/model/mps.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="device 必须"):
        load_weather_attention_config(config)

    base = load_weather_attention_config(
        Path(__file__).resolve().parents[2] / "configs/model/weather_attention.yaml"
    )
    cfg = replace(base, report_dir="trial/reports", report_file="trial/reports/REPORT.md",
                  checkpoint_root="trial/weights",
                  evaluation={**base.evaluation, "summary_file": "trial/reports/summary.md"})
    summary = tmp_path / cfg.evaluation["summary_file"]
    require_independent_weather_reports(cfg, tmp_path)
    model_report(cfg, tmp_path, {"status": "paused", "epoch": 0})
    report_text = (tmp_path / "trial/reports/REPORT.md").read_text(encoding="utf-8")
    assert report_text.startswith("# 天气模型报告\n")
    assert "- 状态：paused\n" in report_text
    import json
    summary_payload = {"status": "completed", "epoch": 3}
    from car_smart_assist.perception.weather_attention_training import write_json
    write_json(summary, summary_payload, cfg=cfg, root=tmp_path)
    assert json.loads(summary.read_text(encoding="utf-8")) == summary_payload
    for path, (content, mtime_ns) in protected_before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime_ns


@pytest.mark.parametrize("target_field", ["report_file", "summary_file"])
def test_compare_report_and_summary_temp_paths_cannot_replace_deployed_report(
    tmp_path, monkeypatch, target_field
):
    from car_smart_assist.config.weather_attention import load_weather_attention_config
    from car_smart_assist.perception import weather_attention_training as training

    deployed_report, checkpoint = _synthetic_project(
        tmp_path, deployed_report_file="shared/REPORT.tmp.md"
    )
    protected_before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (deployed_report, checkpoint)
    }
    base = load_weather_attention_config(
        Path(__file__).resolve().parents[2] / "configs/model/weather_attention.yaml"
    )
    cfg = replace(
        base, report_dir="trial/reports",
        report_file=("shared/REPORT.md" if target_field == "report_file"
                     else "trial/reports/REPORT.md"),
        checkpoint_root="trial/weights",
        evaluation={
            **base.evaluation,
            "summary_file": ("shared/REPORT.md" if target_field == "summary_file"
                             else "trial/reports/summary.json"),
        },
    )
    device_calls = []
    monkeypatch.setattr(training, "select_device", lambda *_args: device_calls.append(True))
    with pytest.raises(ValueError, match="受保护") as raised:
        training.compare_attention(cfg, tmp_path, {}, {}, object(), {})

    assert "shared\\REPORT.tmp.md" in str(raised.value) or "shared/REPORT.tmp.md" in str(raised.value)
    assert device_calls == []
    for path, (content, mtime_ns) in protected_before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime_ns
