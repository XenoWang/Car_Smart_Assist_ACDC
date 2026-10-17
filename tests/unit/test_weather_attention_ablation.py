"""对照配置只改变约定因素，报告保留基线和用户文字。"""

from dataclasses import asdict
from pathlib import Path

from car_smart_assist.config.weather_attention import load_weather_attention_config
from car_smart_assist.perception.weather_attention_ablation import (
    experiment_configs,
    load_ablation_config,
    write_ablation_report,
)


def test_groups_keep_seed_data_initialization_and_only_change_one_factor():
    root = Path(__file__).resolve().parents[2]
    base = load_weather_attention_config(root / "configs/model/weather_attention.yaml")
    options = load_ablation_config(root / "configs/model/weather_attention_ablation.yaml")
    unchanged = asdict(base)
    groups = experiment_configs(base, options, root)
    assert asdict(base) == unchanged
    for cfg in groups.values():
        assert cfg.data == base.data
        assert cfg.data["seed"] == 42
        assert cfg.head == base.head
        assert cfg.loss == base.loss
        assert cfg.evaluation == base.evaluation
        assert cfg.train["epochs"] == 300
        assert cfg.train["schedule_epochs"] == 500
        assert cfg.checkpoint_root != base.checkpoint_root
        assert cfg.report_file != base.report_file
    assert groups["reduced_tuning"].backbone["fine_tune_from"] == 13
    assert groups["reduced_tuning"].train.get("l2_sp", 0) == 0
    assert groups["l2_sp"].backbone == base.backbone
    assert groups["l2_sp"].train["l2_sp"] == 0.0001


def test_report_updates_own_section_without_replacing_baseline_or_user_notes(tmp_path):
    report = tmp_path / "model_report.md"
    report.write_text("Original baseline\n", encoding="utf-8")
    options = {
        "training_epochs": 300,
        "schedule_epochs": 500,
        "report_dir": "reports",
        "report_file": "model_report.md",
    }
    manifest = {"snapshot_root": "baseline"}
    write_ablation_report(options, tmp_path, manifest, {}, "检查完成")
    report.write_text(report.read_text(encoding="utf-8") + "User notes\n", encoding="utf-8")
    write_ablation_report(options, tmp_path, manifest, {}, "训练中")
    text = report.read_text(encoding="utf-8")
    assert text.startswith("Original baseline\n")
    assert text.endswith("User notes\n")
    assert text.count("<!-- WEATHER_ABLATION_BEGIN -->") == 1
    assert "状态：训练中" in text
