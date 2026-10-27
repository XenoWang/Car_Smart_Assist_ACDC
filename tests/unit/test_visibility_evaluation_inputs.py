"""Image-size preservation, evaluation inputs and checkpoint path contracts."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import yaml

from _model_test_helpers import load_module
from car_smart_assist.perception.visibility.degradation import add_blur


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (16, 16), (32, 32), (40, 80)])
def test_blur_retains_small_input_dimensions_and_preserves_source(shape):
    image = np.full((*shape, 3), 255, dtype=np.uint8)
    before = image.copy()
    result, _ = add_blur(image, 1.0, np.random.default_rng(2))
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(image, before)


def test_blur_center_crop_matches_an_independent_one_pixel_reference():
    # sigma=.5, radius=2: central 1-D coefficient is
    # 1/(1+2*exp(-2)+2*exp(-8)); two passes give floor(255*c*c)=157.
    image = np.full((1, 1, 3), 255, dtype=np.uint8)
    result, _ = add_blur(image, 1.0, np.random.default_rng(2), kernel_max=3)
    np.testing.assert_array_equal(result, np.full((1, 1, 3), 157, dtype=np.uint8))
    impulse = np.zeros((9, 11, 3), np.uint8)
    impulse[4, 5] = 255
    normal, _ = add_blur(impulse, 1.0,
                         np.random.default_rng(2), kernel_max=3)
    assert normal.shape == (9, 11, 3)
    np.testing.assert_array_equal(normal[4, 5], [157, 157, 157])
    np.testing.assert_array_equal(normal[0, 0], [0, 0, 0])


def _evaluation_cli(tmp_path, monkeypatch, config):
    cli = load_module("scripts/evaluate_visibility.py", "visibility_input_cli")
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({"visibility": config}), encoding="utf-8")
    (root / "checkpoint.pt").write_bytes(b"not a serialized checkpoint")
    monkeypatch.setattr(cli, "__file__", str(root / "scripts/evaluate_visibility.py"))
    monkeypatch.setattr(sys, "argv", ["evaluate_visibility.py", "--config", "config.yaml",
                                      "--checkpoint", "checkpoint.pt", "--device", "cpu"])
    return cli, root


@pytest.mark.parametrize("kind", ["missing-config", "bad-yaml", "checkpoint", "severity", "gate", "manifest"])
def test_evaluation_rejects_expected_bad_inputs_without_traceback_or_report(
    tmp_path, monkeypatch, capsys, caplog, kind,
):
    cfg = {"device": "cpu", "data": {}, "report": {"output_dir": "reports"}}
    if kind == "gate":
        cfg["thresholds"] = {"info_blind": 0.8, "info_degraded": 0.2}
    if kind == "manifest":
        cfg["data"]["cleaning_manifest"] = "missing-manifest.json"
    cli, root = _evaluation_cli(tmp_path, monkeypatch, cfg)
    if kind == "missing-config":
        (root / "config.yaml").unlink()
    elif kind == "bad-yaml":
        (root / "config.yaml").write_text("visibility: [", encoding="utf-8")
    elif kind == "severity":
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--severities", "2"])
    if kind in ("gate", "manifest"):
        monkeypatch.setattr(cli.VisibilityScorer, "from_checkpoint", lambda *_a, **_k:
                            SimpleNamespace(input_size=(8, 8), calibration=None, scoring_cfg={}))
    assert cli.main() == 2
    text = capsys.readouterr()
    assert "Traceback" not in text.out + text.err + caplog.text
    assert not (root / "reports").exists()
    if kind == "severity":
        assert "--severities" in caplog.text
    elif kind == "gate":
        assert "门控" in caplog.text
    elif kind == "manifest":
        assert "清单" in caplog.text and "missing-manifest.json" in caplog.text
    else:
        assert str(root / ("checkpoint.pt" if kind == "checkpoint" else "config.yaml")) in caplog.text


def test_synthetic_maximum_and_monotonicity_use_severity_values_not_input_order(monkeypatch):
    cli = load_module("scripts/evaluate_visibility.py", "visibility_severity_report")
    from car_smart_assist.perception.visibility.gate import VisibilityGate
    from car_smart_assist.perception.visibility.scorer import InformationFeatures, VisibilityScore

    monkeypatch.setattr(cli, "degrade", lambda image, _kind, severity, seed:
                        (np.full_like(image, round(severity * 100)), None))

    class Scores:
        def score_arrays(self, images):
            return [VisibilityScore("sample", 0.0, 0.0, float("nan"),
                                    InformationFeatures(0, 0, 0, 0),
                                    1.0 - float(image[0, 0, 0]) / 100) for image in images]

    result = cli.eval_synthetic(Scores(), VisibilityGate(), [np.zeros((8, 8, 3), np.uint8)],
                                [1.0, 0.5], 1, 0)
    for row in result.values():
        assert [point["severity"] for point in row["curve"]] == [1.0, 0.5]
        assert row["recall_at_max_severity"] == 1.0
        assert row["information_monotonic_decreasing"] is True


def test_report_records_the_actual_seed_for_each_evaluation_stage(tmp_path, monkeypatch):
    cfg = {"device": "cpu", "data": {"split_seed": 73}, "report": {"output_dir": "reports"}}
    cli, root = _evaluation_cli(tmp_path, monkeypatch, cfg)
    monkeypatch.setattr(cli.VisibilityScorer, "from_checkpoint", lambda *_a, **_k:
                        SimpleNamespace(input_size=(8, 8), calibration=None, scoring_cfg={}))
    monkeypatch.setattr(cli, "list_ref_images", lambda *_a, **_k: [root / "ref.png"])
    monkeypatch.setattr(cli, "load_images", lambda *_a, **_k: [np.zeros((8, 8, 3), np.uint8)])
    observed = []
    monkeypatch.setattr(cli, "eval_fit_quality", lambda _s, _r, _p, data, _g, _n:
                        (observed.append(("fit_quality", data["split_seed"])), {})[1])
    monkeypatch.setattr(cli, "eval_clean_control", lambda _s, _g, _r, _n, seed:
                        (observed.append(("clean_control", seed)), {})[1])
    monkeypatch.setattr(cli, "eval_real_adverse", lambda _s, _g, _a, _z, _n, seed, **_k:
                        (observed.append(("real_adverse", seed)), {})[1])
    monkeypatch.setattr(cli, "eval_synthetic", lambda _s, _g, _r, _v, _n, seed:
                        (observed.append(("synthetic", seed)), {})[1])
    reports = []

    class ReportBoundary(Exception):
        pass

    def capture(report, _path):
        reports.append(report)
        raise ReportBoundary

    monkeypatch.setattr(cli, "write_report", capture)
    with pytest.raises(ReportBoundary):
        cli.main()
    assert observed == [("fit_quality", 73), ("clean_control", 0), ("real_adverse", 0), ("synthetic", 0)]
    assert reports[0]["evaluation_protocol"]["split_seed"] == 73
    assert reports[0]["evaluation_protocol"]["sample_seeds"] == dict(observed)
    assert not (root / "reports").exists()


def test_trainer_resolves_explicit_resume_from_project_root_not_cwd(tmp_path, monkeypatch):
    helpers = load_module("tests/unit/test_visibility_cache_and_cli_contracts.py", "resume_size_fixtures")
    from car_smart_assist.perception.visibility.trainer import VisibilityTrainer

    project, external = tmp_path / "project", tmp_path / "external"
    checkpoint = project / "weights/last.pt"
    checkpoint.parent.mkdir(parents=True)
    external.mkdir()
    model = {"input_size": [8, 8], "base_channels": 2, "latent_dim": 4}
    helpers._write_synthetic_resume_checkpoint(checkpoint, model)
    wrong = external / "weights/last.pt"
    wrong.parent.mkdir()
    wrong.write_bytes(b"wrong CWD checkpoint")
    before = checkpoint.read_bytes()
    monkeypatch.chdir(external)
    with torch.random.fork_rng(devices=[]):
        trainer = VisibilityTrainer({"device": "cpu", "model": model,
                                     "train": {"checkpoint_dir": "weights"}},
                                    project_root=project, resume="weights/last.pt")
    assert trainer.resume_path == checkpoint
    assert checkpoint.read_bytes() == before
    assert wrong.read_bytes() == b"wrong CWD checkpoint"


def test_partition_input_failures_have_a_specific_input_error_type(tmp_path, monkeypatch):
    cli = load_module("scripts/evaluate_visibility.py", "partition_input_boundary")
    from car_smart_assist.perception.visibility import dataset

    failure = ValueError("invalid configured partition ratios")

    def reject(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(dataset, "split_ref_indices", reject)
    with pytest.raises(cli.EvaluationInputError) as raised:
        cli.eval_fit_quality(SimpleNamespace(input_size=(4, 4)), [], [],
                             {"split_by_sequence": False}, None, 1)
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("stage", ["fit", "adverse"])
def test_scoring_program_errors_are_not_reclassified_as_input_errors(tmp_path, monkeypatch, stage):
    cli = load_module("scripts/evaluate_visibility.py", "scoring_error_boundary")
    from car_smart_assist.perception.visibility import dataset

    failure = ValueError("unrelated scoring program error")

    def score(*_args, **_kwargs):
        raise failure

    refs = [np.zeros((4, 4, 3), np.uint8)] * 4
    paths = [tmp_path / f"reference-{index}.png" for index in range(4)]
    scorer = SimpleNamespace(input_size=(4, 4), split_signature="accepted", score_arrays=score)
    monkeypatch.setattr(dataset, "split_ref_indices", lambda *_a, **_k:
                        tuple(np.array([index]) for index in range(4)))
    monkeypatch.setattr(dataset, "reference_split_signature", lambda *_a: "accepted")
    monkeypatch.setattr(cli, "list_adverse_images", lambda *_a, **_k: paths[:1])
    monkeypatch.setattr(cli, "load_images", lambda *_a, **_k: refs[:1])
    with pytest.raises(ValueError) as raised:
        if stage == "fit":
            cli.eval_fit_quality(scorer, refs, paths, {"split_by_sequence": False}, None, 1)
        else:
            cli.eval_real_adverse(scorer, None, tmp_path, (4, 4), 1, 0)
    assert raised.value is failure
