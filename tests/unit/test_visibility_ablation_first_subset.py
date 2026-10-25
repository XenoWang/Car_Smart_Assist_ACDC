"""CLI regressions for first-time --only visibility ablation continuation."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _prepare(tmp_path, monkeypatch):
    helpers = _load("tests/unit/test_remaining_control_bugs.py", "rev02_existing_test_helpers")
    module, base, _ = helpers._minimal_ablation_setup(tmp_path, monkeypatch)
    fit_calls = []

    class FakeTrainer:
        def __init__(self, cfg, project_root, resume):
            self.cfg = cfg
            self.root = Path(project_root)
            self.tag = Path(cfg["train"]["checkpoint_dir"]).name
            self.resume = resume

        def fit(self, *, stop_controller, continue_only):
            assert self.resume == "none"
            assert continue_only is False
            fit_calls.append(self.tag)
            folder = self.root / self.cfg["train"]["checkpoint_dir"]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "last.pt").write_bytes(f"last:{self.tag}".encode("utf-8"))
            (folder / "best.pt").write_bytes(f"best:{self.tag}".encode("utf-8"))
            return SimpleNamespace(manual_stop=False, train_loss=[0.25], best_epoch=1)

    def scorer_factory(_cls, _checkpoint, *, cfg, scoring_cfg, **_kwargs):
        return SimpleNamespace(
            input_size=(4, 4),
            scoring_cfg=scoring_cfg if scoring_cfg is not None else {},
        )

    monkeypatch.setattr(module, "resolve_device", lambda *_args: "cpu")
    monkeypatch.setattr(module, "VisibilityTrainer", FakeTrainer)
    monkeypatch.setattr(module.VisibilityScorer, "from_checkpoint", classmethod(scorer_factory))
    monkeypatch.setattr(module.VisibilityGate, "from_config", classmethod(
        lambda _cls, _cfg: SimpleNamespace(thresholds=SimpleNamespace())
    ))
    monkeypatch.setattr(module, "evaluate_run", lambda *_args, **_kwargs: {
        "params_m": 0.1,
        "overfit_ratio": 1.0,
        "val_fpr": 0.0,
        "mean_recall": 0.5,
        "synth": {"fog": {"recall": 0.5}},
    })
    state_path = tmp_path / "artifacts/reports/visibility/ablations/training_progress.json"
    assert not state_path.exists()
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--only", "alpha"])
    assert module.main() == 0
    assert fit_calls == ["alpha"]

    out_dir = tmp_path / "artifacts/reports/visibility/ablations"
    state_path = out_dir / "training_progress.json"
    results_path = out_dir / "results.json"
    checkpoint_dir = tmp_path / "artifacts/checkpoints/visibility_ablations/alpha"
    return SimpleNamespace(
        module=module,
        base=base,
        fit_calls=fit_calls,
        out_dir=out_dir,
        state_path=state_path,
        results_path=results_path,
        checkpoint_dir=checkpoint_dir,
        root=tmp_path,
    )


def _expected_row(ctx, tag):
    module, base, root = ctx.module, ctx.base, ctx.root
    preset = next((p for p in [{"tag": "baseline", "set": {}}] + base.get("ablations", [])
                   if p["tag"] == tag), None)
    assert preset is not None
    cfg = module.apply_config_overrides(base, preset.get("set", {}))
    cfg.setdefault("train", {})["checkpoint_dir"] = f"artifacts/checkpoints/visibility_ablations/{tag}"
    inputs = module._prepare_inputs(cfg, root)
    identity = {"version": 1, "config": cfg, "data": inputs["data_identity"],
                "evaluation_protocol": inputs["protocol"]}
    return {
        "experiment_signature": module._signature(identity),
        "experiment_identity": identity,
        "val_fpr": 0.0,
        "synth": {"fog": {"recall": 0.5}},
    }


def _continue(ctx, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--continue", *args])
    return ctx.module.main()


def _persistent_snapshot(ctx):
    checkpoints = {
        path.relative_to(ctx.root).as_posix(): path.read_bytes()
        for path in ctx.root.rglob("*.pt")
    }
    return {
        "progress": ctx.state_path.read_bytes() if ctx.state_path.exists() else None,
        "results": ctx.results_path.read_bytes() if ctx.results_path.exists() else None,
        "checkpoints": checkpoints,
    }


def test_first_subset_then_full_continue_runs_only_missing_groups_and_preserves_alpha(
    tmp_path, monkeypatch
):
    ctx = _prepare(tmp_path, monkeypatch)
    alpha_result_before = json.loads(ctx.results_path.read_text(encoding="utf-8"))["alpha"]
    alpha_last = ctx.checkpoint_dir / "last.pt"
    alpha_best = ctx.checkpoint_dir / "best.pt"
    alpha_before = {
        "last": (alpha_last.read_bytes(), alpha_last.stat().st_mtime_ns),
        "best": (alpha_best.read_bytes(), alpha_best.stat().st_mtime_ns),
    }
    first_state = json.loads(ctx.state_path.read_text(encoding="utf-8"))
    assert set(first_state["identity"]) == {"alpha"}
    assert first_state["groups"]["alpha"]["status"] == "completed"

    assert _continue(ctx, monkeypatch) == 0
    assert ctx.fit_calls == ["alpha", "baseline", "beta"]
    final_state = json.loads(ctx.state_path.read_text(encoding="utf-8"))
    final_results = json.loads(ctx.results_path.read_text(encoding="utf-8"))
    assert set(final_state["identity"]) == {"baseline", "alpha", "beta"}
    assert final_state["identity"]["alpha"] == first_state["identity"]["alpha"]
    assert final_state["groups"]["alpha"] == first_state["groups"]["alpha"]
    assert set(final_state["groups"]) == {"baseline", "alpha", "beta"}
    assert all(row["status"] == "completed" for row in final_state["groups"].values())
    assert set(final_results) == {"baseline", "alpha", "beta"}
    assert final_results["alpha"] == alpha_result_before
    assert (alpha_last.read_bytes(), alpha_last.stat().st_mtime_ns) == alpha_before["last"]
    assert (alpha_best.read_bytes(), alpha_best.stat().st_mtime_ns) == alpha_before["best"]

    progress_before_repeat = ctx.state_path.read_bytes()
    results_before_repeat = ctx.results_path.read_bytes()
    calls_before_repeat = list(ctx.fit_calls)
    assert _continue(ctx, monkeypatch) == 0
    assert ctx.fit_calls == calls_before_repeat
    assert ctx.state_path.read_bytes() == progress_before_repeat
    assert ctx.results_path.read_bytes() == results_before_repeat


@pytest.mark.parametrize(
    "corruption",
    [
        "stored_identity",
        "epoch_budget",
        "data_identity",
        "new_group_cache_result",
        "new_group_orphan_best",
        "new_group_orphan_last",
        "unknown_suite_tag",
        "inconsistent_group_keys",
        "invalid_group_status",
        "non_mapping_alpha_group",
        "completed_alpha_missing_checkpoint",
        "completed_alpha_changed_checkpoint",
        "missing_progress_file",
        "continue_unregistered_beta",
    ],
)
def test_first_subset_continue_rejects_conflicts_before_any_new_fit(tmp_path, monkeypatch, corruption):
    ctx = _prepare(tmp_path, monkeypatch)
    original_state = json.loads(ctx.state_path.read_text(encoding="utf-8"))
    full_continue_args = ()

    if corruption == "stored_identity":
        original_state["identity"]["alpha"] = "different-data-or-config"
        ctx.state_path.write_text(json.dumps(original_state), encoding="utf-8")
    elif corruption == "epoch_budget":
        full_continue_args = ("--epochs", "2")
    elif corruption == "data_identity":
        prepare = ctx.module._prepare_inputs
        def changed_data(cfg, root):
            inputs = prepare(cfg, root)
            inputs["data_identity"] = {**inputs["data_identity"], "signature": "changed-data"}
            return inputs
        monkeypatch.setattr(ctx.module, "_prepare_inputs", changed_data)
    elif corruption == "new_group_cache_result":
        results = json.loads(ctx.results_path.read_text(encoding="utf-8"))
        results["beta"] = _expected_row(ctx, "beta")
        ctx.results_path.write_text(json.dumps(results), encoding="utf-8")
    elif corruption in {"new_group_orphan_best", "new_group_orphan_last"}:
        name = "best.pt" if corruption.endswith("best") else "last.pt"
        orphan = ctx.root / "artifacts/checkpoints/visibility_ablations/beta" / name
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(f"orphan-{name}".encode("utf-8"))
    elif corruption == "unknown_suite_tag":
        original_state["identity"]["ghost"] = "unknown-signature"
        original_state["groups"]["ghost"] = {"status": "pending"}
        ctx.state_path.write_text(json.dumps(original_state), encoding="utf-8")
    elif corruption == "inconsistent_group_keys":
        original_state["groups"]["ghost"] = {"status": "pending"}
        ctx.state_path.write_text(json.dumps(original_state), encoding="utf-8")
    elif corruption == "invalid_group_status":
        original_state["groups"]["alpha"]["status"] = "damaged"
        ctx.state_path.write_text(json.dumps(original_state), encoding="utf-8")
    elif corruption == "non_mapping_alpha_group":
        original_state["groups"]["alpha"] = None
        ctx.state_path.write_text(json.dumps(original_state), encoding="utf-8")
    elif corruption == "completed_alpha_missing_checkpoint":
        (ctx.checkpoint_dir / "best.pt").unlink()
    elif corruption == "completed_alpha_changed_checkpoint":
        (ctx.checkpoint_dir / "best.pt").write_bytes(b"tampered-best-checkpoint")
    elif corruption == "missing_progress_file":
        ctx.state_path.unlink()
    elif corruption == "continue_unregistered_beta":
        full_continue_args = ("--only", "beta")

    snapshot_before = _persistent_snapshot(ctx)
    fit_calls_before = list(ctx.fit_calls)
    assert _continue(ctx, monkeypatch, *full_continue_args) == 2
    assert ctx.fit_calls == fit_calls_before
    assert _persistent_snapshot(ctx) == snapshot_before

