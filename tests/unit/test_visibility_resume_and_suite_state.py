"""Visibility checkpoint calibration and ablation STOP/lock state behavior."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_updated_visibility_weights_do_not_retain_old_calibration_on_stop(tmp_path, monkeypatch):
    helper = _load("tests/unit/test_visibility_retrain_training.py", "visibility_calibration_resume_helper")
    from car_smart_assist.perception.visibility.trainer import VisibilityTrainer
    from car_smart_assist.utils.stop import StopController

    first = helper._trainer_with_controlled_validation(tmp_path, monkeypatch, [1.0, 1.1], epochs=2)
    first.fit()
    last_path = tmp_path / "checkpoints/last.pt"
    best_path = tmp_path / "checkpoints/best.pt"
    initial_last = torch.load(last_path, map_location="cpu", weights_only=False)
    best_bytes = best_path.read_bytes()
    best_mtime_ns = best_path.stat().st_mtime_ns
    initial_best = torch.load(best_path, map_location="cpu", weights_only=False)
    assert initial_last["calibration"] is None
    assert initial_best["calibration"] is not None
    assert initial_best["epoch"] == 1
    assert initial_best["optimizer_state"]["param_groups"][0]["lr"] > 0
    best_model = {
        name: value.detach().cpu().clone()
        for name, value in initial_best["model_state"].items()
    }

    resume_cfg = helper._config(tmp_path, epochs=2)
    resume_cfg["train"]["resume_extra_epochs"] = 1
    resumed = VisibilityTrainer(
        resume_cfg, project_root=tmp_path, resume=best_path
    )
    samples = torch.linspace(0, 1, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)

    class Rows:
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index]

    rows = Rows()
    monkeypatch.setattr(resumed, "_build_datasets", lambda: (rows, rows, rows, []))
    monkeypatch.setattr(resumed, "_eval_loss", lambda _loader: 1.1)
    controller = StopController(default_directory=resumed.ckpt_dir)
    batches = []

    def request_after_forward(_module, _inputs, _output):
        batches.append(True)
        if len(batches) == 1:
            controller.request()

    hook = resumed.model.register_forward_hook(request_after_forward)
    history = resumed.fit(stop_controller=controller, continue_only=False)
    hook.remove()
    stopped = torch.load(last_path, map_location="cpu", weights_only=False)

    assert history.manual_stop is True
    assert len(history.train_loss) == 2  # Best epoch 1 plus the explicitly added epoch.
    assert batches
    assert stopped["calibration"] is None
    assert any(
        not torch.equal(stopped["model_state"][name].cpu(), value)
        for name, value in best_model.items()
    )
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    assert best_path.read_bytes() == best_bytes
    assert best_path.stat().st_mtime_ns == best_mtime_ns
    assert best["calibration"] is not None
    assert best["best_model_sha256"] == initial_best["best_model_sha256"]
    assert stopped["best_model_sha256"] == best["best_model_sha256"]
    assert all(torch.equal(best["model_state"][name], value)
               for name, value in best_model.items())


def _ablation_setup(tmp_path, monkeypatch, argv, *, tags=("alpha",)):
    helper = _load("tests/unit/test_training_cli_and_resume_contracts.py", "visibility_lock_test_helpers")
    module, base, _ = helper._minimal_ablation_setup(tmp_path, monkeypatch, tags=tags)
    module._test_base = base
    from car_smart_assist.utils.stop import StopController

    monkeypatch.setattr(module, "StopController", StopController)
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", *argv])
    return module


@pytest.mark.parametrize("lock_kind", ["suite", "selected-group"])
@pytest.mark.parametrize("marker_kind", ["suite", "selected-group"])
def test_fresh_lock_conflict_preserves_existing_stop_marker(tmp_path, monkeypatch, lock_kind, marker_kind):
    import threading
    from filelock import FileLock, Timeout

    module = _ablation_setup(tmp_path, monkeypatch, ["--fresh", "--only", "alpha"])
    out_dir = tmp_path / "artifacts/reports/visibility/ablations"
    group_dir = tmp_path / "artifacts/checkpoints/visibility_ablations/alpha"
    out_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)
    stop = (tmp_path / "artifacts/checkpoints/visibility_ablations/STOP"
            if marker_kind == "suite" else group_dir / "STOP")
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_bytes(b"user stop request\n")
    before = (stop.read_bytes(), stop.stat().st_mtime_ns)
    lock_path = out_dir / "run.lock" if lock_kind == "suite" else group_dir / "run.lock"

    entered, release = threading.Event(), threading.Event()
    holder_errors = []

    def hold_lock():
        try:
            with FileLock(str(lock_path.resolve()), timeout=2):
                entered.set()
                release.wait(5)
        except BaseException as exc:
            holder_errors.append(exc)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert entered.wait(5), "lock-holder thread did not acquire the selected lock"
    try:
        with pytest.raises(Timeout):
            module.main()
    finally:
        release.set()
        holder.join(5)

    assert not holder.is_alive() and not holder_errors
    assert (stop.read_bytes(), stop.stat().st_mtime_ns) == before


@pytest.mark.parametrize("mode", ["fresh", "continue"])
@pytest.mark.parametrize(
    "marker_kind", ["suite", "selected-group", "custom", "default-suite-with-custom"]
)
def test_new_stop_arriving_after_snapshot_survives_resume_request(tmp_path, monkeypatch, mode, marker_kind):
    args = [f"--{mode}", "--only", "alpha"]
    custom = tmp_path / "external-stop/STOP"
    if marker_kind in {"custom", "default-suite-with-custom"}:
        args.extend(("--stop-file", str(custom)))
    module = _ablation_setup(tmp_path, monkeypatch, args)
    # The helper's synthetic input preparation runs after STOP identities are captured.
    prepare = module._prepare_inputs
    injected = (
        tmp_path / "artifacts/checkpoints/visibility_ablations/STOP"
        if marker_kind in {"suite", "default-suite-with-custom"} else
        tmp_path / "artifacts/checkpoints/visibility_ablations/alpha/STOP"
        if marker_kind == "selected-group" else custom
    )

    def create_stop_during_preparation(cfg, root):
        result = prepare(cfg, root)
        injected.parent.mkdir(parents=True, exist_ok=True)
        injected.write_bytes(b"arrived after lock-snapshot\n")
        return result

    monkeypatch.setattr(module, "_prepare_inputs", create_stop_during_preparation)
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 75
    assert injected.read_bytes() == b"arrived after lock-snapshot\n"
    if marker_kind == "default-suite-with-custom":
        assert not custom.exists()
    out_dir = tmp_path / "artifacts/reports/visibility/ablations"
    assert not (out_dir / "results.json").exists()


@pytest.mark.parametrize("lock_kind", ["suite", "selected-group"])
def test_stop_request_completes_while_suite_or_group_lock_is_held(tmp_path, monkeypatch, lock_kind):
    import threading
    from filelock import FileLock

    module = _ablation_setup(tmp_path, monkeypatch, ["--stop", "--only", "alpha"])
    out_dir = tmp_path / "artifacts/reports/visibility/ablations"
    group_dir = tmp_path / "artifacts/checkpoints/visibility_ablations/alpha"
    out_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / "run.lock" if lock_kind == "suite" else group_dir / "run.lock"
    entered, release = threading.Event(), threading.Event()
    holder_errors = []

    def hold_lock():
        try:
            with FileLock(str(lock_path.resolve()), timeout=2):
                entered.set()
                release.wait(5)
        except BaseException as exc:
            holder_errors.append(exc)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert entered.wait(5), "lock-holder thread did not acquire the selected lock"
    try:
        assert module.main() == 0
    finally:
        release.set()
        holder.join(5)

    stop = tmp_path / "artifacts/checkpoints/visibility_ablations/STOP"
    assert stop.is_file()
    assert not holder.is_alive() and not holder_errors


def test_fresh_pause_then_continue_trains_instead_of_reusing_archived_score(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    helper = _load("tests/unit/test_training_cli_and_resume_contracts.py", "visibility_fresh_resume_helper")
    module, base, _ = helper._minimal_ablation_setup(tmp_path, monkeypatch, tags=("alpha", "beta"))
    from car_smart_assist.utils.stop import StopController

    monkeypatch.setattr(module, "StopController", StopController)
    out_dir = tmp_path / "artifacts/reports/visibility/ablations"
    checkpoint_dir = tmp_path / "artifacts/checkpoints/visibility_ablations/alpha"
    out_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    cfg = module.apply_config_overrides(base, {})
    cfg.setdefault("train", {})["checkpoint_dir"] = "artifacts/checkpoints/visibility_ablations/alpha"
    inputs = module._prepare_inputs(cfg, tmp_path)
    identity = {"version": 1, "config": cfg, "data": inputs["data_identity"],
                "evaluation_protocol": inputs["protocol"]}
    signature = module._signature(identity)
    old_last, old_best = b"old completed last", b"old completed best"
    (checkpoint_dir / "last.pt").write_bytes(old_last)
    (checkpoint_dir / "best.pt").write_bytes(old_best)
    old_result = {"experiment_identity": identity, "experiment_signature": signature,
                  "val_fpr": 0.75, "synth": {"origin": "old score"}}
    beta_dir = tmp_path / "artifacts/checkpoints/visibility_ablations/beta"
    beta_dir.mkdir(parents=True)
    beta_old = {"last.pt": b"unselected old last", "best.pt": b"unselected old best"}
    for name, content in beta_old.items():
        (beta_dir / name).write_bytes(content)
    beta_result = {"experiment_identity": {"unselected": True},
                   "experiment_signature": "unselected-old-signature",
                   "val_fpr": 0.25, "synth": {"origin": "unselected old score"}}
    beta_cfg = module.apply_config_overrides(base, {})
    beta_cfg.setdefault("train", {})["checkpoint_dir"] = (
        "artifacts/checkpoints/visibility_ablations/beta"
    )
    beta_inputs = module._prepare_inputs(beta_cfg, tmp_path)
    beta_identity = {"version": 1, "config": beta_cfg,
                     "data": beta_inputs["data_identity"],
                     "evaluation_protocol": beta_inputs["protocol"]}
    beta_signature = module._signature(beta_identity)
    (out_dir / "results.json").write_text(
        json.dumps({"alpha": old_result, "beta": beta_result}), encoding="utf-8"
    )
    (out_dir / "training_progress.json").write_text(json.dumps({
        "identity": {"alpha": signature, "beta": beta_signature},
        "groups": {
            "alpha": {
                "status": "completed",
                "last_sha256": hashlib.sha256(old_last).hexdigest(),
                "best_sha256": hashlib.sha256(old_best).hexdigest(),
            },
            "beta": {"status": "pending"},
        },
    }), encoding="utf-8")

    calls = []

    class Trainer:
        def __init__(self, run_cfg, project_root, resume):
            self.cfg, self.root, self.resume = run_cfg, Path(project_root), resume

        def fit(self, *, stop_controller, continue_only):
            tag_dir = self.root / self.cfg["train"]["checkpoint_dir"]
            calls.append((self.resume, continue_only))
            if continue_only:
                assert self.resume == tag_dir / "last.pt"
                (tag_dir / "last.pt").write_bytes(b"new continued last")
                (tag_dir / "best.pt").write_bytes(b"new continued best")
                return SimpleNamespace(manual_stop=False, train_loss=[0.25], best_epoch=2)
            assert self.resume == "none" and not continue_only
            (tag_dir / "last.pt").write_bytes(b"new paused last")
            (tag_dir / "best.pt").write_bytes(b"new paused best")
            return SimpleNamespace(manual_stop=True, train_loss=[0.5], best_epoch=1)

    monkeypatch.setattr(module, "VisibilityTrainer", Trainer)
    monkeypatch.setattr(module, "resolve_device", lambda *_args: "cpu")
    monkeypatch.setattr(module.VisibilityScorer, "from_checkpoint", classmethod(
        lambda _cls, *_args, **kwargs: SimpleNamespace(input_size=(4, 4), scoring_cfg=kwargs["scoring_cfg"] or {})
    ))
    monkeypatch.setattr(module.VisibilityGate, "from_config", classmethod(
        lambda _cls, _cfg: SimpleNamespace(thresholds=SimpleNamespace())
    ))
    monkeypatch.setattr(module, "evaluate_run", lambda *_args, **_kwargs: {
        "params_m": 0.1, "overfit_ratio": 1.0, "val_fpr": 0.1,
        "mean_recall": 0.5, "synth": {"origin": "new score"},
    })

    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--fresh", "--only", "alpha"])
    assert module.main() == module.STOP_EXIT_CODE
    paused = json.loads((out_dir / "training_progress.json").read_text(encoding="utf-8"))
    assert paused["groups"]["alpha"]["status"] == "paused"
    fresh_results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert "alpha" not in fresh_results
    assert fresh_results["beta"] == beta_result
    assert paused["groups"]["beta"]["status"] == "pending"
    assert paused["identity"]["beta"] == beta_signature
    assert (checkpoint_dir / "last.pt").read_bytes() == b"new paused last"
    assert {name: (beta_dir / name).read_bytes() for name in beta_old} == beta_old
    alpha_archives = list((checkpoint_dir / "history").glob("*/last.pt"))
    assert len(alpha_archives) == 1
    archive_dir = alpha_archives[0].parent
    assert {name: (archive_dir / name).read_bytes() for name in ("last.pt", "best.pt")} == {
        "last.pt": old_last, "best.pt": old_best,
    }

    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--continue", "--only", "alpha"])
    assert module.main() == 0
    completed = json.loads((out_dir / "training_progress.json").read_text(encoding="utf-8"))
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert completed["groups"]["alpha"]["status"] == "completed"
    assert len(calls) == 2 and calls[1] == (checkpoint_dir / "last.pt", True)
    assert results["alpha"]["synth"] == {"origin": "new score"}


def test_fresh_archives_old_checkpoints_for_all_selected_pending_groups(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    module = _ablation_setup(
        tmp_path, monkeypatch, ["--fresh", "--only", "alpha", "beta"], tags=("alpha", "beta")
    )
    base = module._test_base
    out_dir = tmp_path / "artifacts/reports/visibility/ablations"
    checkpoint_root = tmp_path / "artifacts/checkpoints/visibility_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    identities, groups, old_pairs = {}, {}, {}
    for tag in ("alpha", "beta"):
        cfg = module.apply_config_overrides(base, {})
        cfg.setdefault("train", {})["checkpoint_dir"] = (
            f"artifacts/checkpoints/visibility_ablations/{tag}"
        )
        inputs = module._prepare_inputs(cfg, tmp_path)
        identity = {"version": 1, "config": cfg, "data": inputs["data_identity"],
                    "evaluation_protocol": inputs["protocol"]}
        identities[tag] = module._signature(identity)
        group_dir = checkpoint_root / tag
        group_dir.mkdir(parents=True, exist_ok=True)
        old_pairs[tag] = {"last.pt": f"old-{tag}-last".encode(),
                          "best.pt": f"old-{tag}-best".encode()}
        for name, content in old_pairs[tag].items():
            (group_dir / name).write_bytes(content)
        groups[tag] = {"status": "pending"}

    progress_path = out_dir / "training_progress.json"
    progress_path.write_text(json.dumps({"identity": identities, "groups": groups}), encoding="utf-8")
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps({"alpha": {"old": True}, "beta": {"old": True}}),
                            encoding="utf-8")
    calls = []

    class Trainer:
        def __init__(self, cfg, project_root, resume):
            self.tag = Path(cfg["train"]["checkpoint_dir"]).name
            self.root, self.resume = Path(project_root), resume

        def fit(self, *, stop_controller, continue_only):
            calls.append((self.tag, self.resume, continue_only))
            assert self.tag == "alpha"  # beta is pending and must wait for a later invocation.
            assert self.resume == "none" and continue_only is False
            (self.root / f"artifacts/checkpoints/visibility_ablations/{self.tag}/last.pt").write_bytes(
                b"new alpha last"
            )
            (self.root / f"artifacts/checkpoints/visibility_ablations/{self.tag}/best.pt").write_bytes(
                b"new alpha best"
            )
            return SimpleNamespace(manual_stop=True, train_loss=[0.5], best_epoch=1)

    monkeypatch.setattr(module, "VisibilityTrainer", Trainer)
    assert module.main() == module.STOP_EXIT_CODE
    assert calls == [("alpha", "none", False)]
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["groups"]["alpha"]["status"] == "paused"
    assert progress["groups"]["beta"]["status"] == "pending"
    assert json.loads(results_path.read_text(encoding="utf-8")) == {}
    for tag in ("alpha", "beta"):
        group_dir = checkpoint_root / tag
        archive_pairs = list((group_dir / "history").glob("*/last.pt"))
        assert len(archive_pairs) == 1
        archive_dir = archive_pairs[0].parent
        assert {name: (archive_dir / name).read_bytes() for name in ("last.pt", "best.pt")} == old_pairs[tag]
    assert not (checkpoint_root / "beta/last.pt").exists()
    assert not (checkpoint_root / "beta/best.pt").exists()


def test_fresh_pause_then_continue_runs_cpu_training_and_updates_checkpoint(tmp_path, monkeypatch):
    import json
    import numpy as np
    import yaml
    from types import SimpleNamespace

    helper = _load("tests/unit/test_visibility_retrain_training.py", "visibility_suite_cpu_training_helper")
    module = _ablation_setup(tmp_path, monkeypatch, ["--fresh", "--only", "alpha"])
    from car_smart_assist.perception.visibility.trainer import VisibilityTrainer

    base = module._test_base
    config = helper._config(tmp_path, epochs=2, patience=5)
    config["model"]["input_size"] = [32, 32]
    config["train"]["checkpoint_dir"] = "artifacts/checkpoints/visibility_ablations/alpha"
    config["data"]["split_mode"] = "fixed"
    base.update(config)
    base["thresholds"] = {"use_recon_z": True}
    (tmp_path / "configs/model/visibility.yaml").write_text(
        yaml.safe_dump({"visibility": base}, sort_keys=False), encoding="utf-8"
    )

    samples = torch.linspace(0.0, 1.0, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)

    class Rows:
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index]

    rows = Rows()

    def build_synthetic_partitions(trainer):
        trainer.split_signature = "fixed-synthetic-visibility-training-data"
        trainer.split_stats = {"train": 4, "val": 4, "calib": 4, "test": 0}
        return rows, rows, rows, []

    validation_calls = []
    suite_stop = tmp_path / "artifacts/checkpoints/visibility_ablations/STOP"

    def evaluate_training_loss(_trainer, _loader):
        validation_calls.append(True)
        if len(validation_calls) == 1:
            suite_stop.parent.mkdir(parents=True, exist_ok=True)
            suite_stop.write_bytes(b"pause after first validation\n")
            return 1.0
        return 0.8

    monkeypatch.setattr(VisibilityTrainer, "_build_datasets", build_synthetic_partitions)
    monkeypatch.setattr(VisibilityTrainer, "_eval_loss", evaluate_training_loss)
    real_fit = VisibilityTrainer.fit
    fit_states = []

    def observe_fit(trainer, *, stop_controller=None, continue_only=False):
        fit_states.append((trainer.resume_path, continue_only))
        return real_fit(trainer, stop_controller=stop_controller, continue_only=continue_only)

    monkeypatch.setattr(VisibilityTrainer, "fit", observe_fit)
    synthetic_refs = [
        np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
        for value in (24, 72, 128, 200)
    ]
    monkeypatch.setattr(module, "_prepare_inputs", lambda *_args: {
        "paths": [], "size": (32, 32), "splits": tuple(np.arange(4) for _ in range(4)),
        "data_identity": {"signature": "synthetic-reference-set"},
        "protocol": {"signature": "synthetic-evaluation-protocol"},
        "refs": synthetic_refs,
    })
    monkeypatch.setattr(module, "resolve_device", lambda *_args: "cpu")
    monkeypatch.setattr(module.VisibilityScorer, "from_checkpoint", classmethod(
        lambda _cls, *_args, **_kwargs: SimpleNamespace(input_size=(32, 32), scoring_cfg={})
    ))
    monkeypatch.setattr(module.VisibilityGate, "from_config", classmethod(
        lambda _cls, _cfg: SimpleNamespace(thresholds=SimpleNamespace())
    ))
    monkeypatch.setattr(module, "evaluate_run", lambda *_args, **_kwargs: {
        "params_m": 0.1, "overfit_ratio": 1.0, "val_fpr": 0.0,
        "mean_recall": 0.5, "synth": {"fog": {"recall": 0.5}},
    })

    assert module.main() == module.STOP_EXIT_CODE
    checkpoint_dir = tmp_path / "artifacts/checkpoints/visibility_ablations/alpha"
    last_path = checkpoint_dir / "last.pt"
    paused = torch.load(last_path, map_location="cpu", weights_only=False)
    assert paused["epoch"] == 1 and paused["manual_stop"] is True
    paused_bytes = last_path.read_bytes()
    assert fit_states == [(None, False)]

    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--continue", "--only", "alpha"])
    assert module.main() == 0
    completed = torch.load(last_path, map_location="cpu", weights_only=False)
    progress = json.loads(
        (tmp_path / "artifacts/reports/visibility/ablations/training_progress.json")
        .read_text(encoding="utf-8")
    )
    assert completed["epoch"] == 2 and completed["training_complete"] is True
    assert completed["manual_stop"] is False
    assert last_path.read_bytes() != paused_bytes
    assert fit_states == [(None, False), (last_path, True)]
    assert len(validation_calls) == 2
    assert progress["groups"]["alpha"]["status"] == "completed"


def _ablation_identity(module, base, root):
    cfg = module.apply_config_overrides(base, {})
    cfg.setdefault("train", {})["checkpoint_dir"] = (
        "artifacts/checkpoints/visibility_ablations/alpha"
    )
    inputs = module._prepare_inputs(cfg, root)
    identity = {"version": 1, "config": cfg, "data": inputs["data_identity"],
                "evaluation_protocol": inputs["protocol"]}
    return cfg, inputs, identity, module._signature(identity)


def test_ablation_continue_recovers_valid_first_checkpoint_transaction(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    module = _ablation_setup(tmp_path, monkeypatch, ["--continue", "--only", "alpha"])
    _cfg, _inputs, identity, signature = _ablation_identity(module, module._test_base, tmp_path)
    out_dir = tmp_path / "artifacts/reports/visibility/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = tmp_path / "artifacts/checkpoints/visibility_ablations/alpha"
    transaction_helper = _load(
        "tests/unit/test_visibility_checkpoint_transactions.py", "visibility_journal_fixture"
    )
    transaction_helper._publish_interrupted_pair(checkpoint_dir, monkeypatch, "last.pt")
    transaction = checkpoint_dir / ".visibility_checkpoint_transaction"
    assert transaction.is_dir()
    assert not (checkpoint_dir / "last.pt").exists()
    (out_dir / "training_progress.json").write_text(json.dumps({
        "identity": {"alpha": signature}, "groups": {"alpha": {"status": "paused"}},
    }), encoding="utf-8")
    (out_dir / "results.json").write_text("{}", encoding="utf-8")
    calls = []

    class Trainer:
        def __init__(self, _cfg, project_root, resume):
            self.root, self.resume = Path(project_root), Path(resume)

        def fit(self, *, stop_controller, continue_only):
            calls.append((self.resume, continue_only))
            assert continue_only is True
            assert self.resume == checkpoint_dir / "last.pt"
            assert not transaction.exists()
            assert torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False) == {
                "payload": "new-best"
            }
            assert torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=False) == {
                "payload": "new-last"
            }
            return SimpleNamespace(manual_stop=True)

    monkeypatch.setattr(module, "VisibilityTrainer", Trainer)
    assert module.main() == module.STOP_EXIT_CODE
    assert len(calls) == 1
    progress = json.loads((out_dir / "training_progress.json").read_text(encoding="utf-8"))
    assert progress["groups"]["alpha"]["status"] == "paused"


def test_ablation_continue_rejects_tampered_transaction_before_training(tmp_path, monkeypatch):
    import json

    module = _ablation_setup(tmp_path, monkeypatch, ["--continue", "--only", "alpha"])
    _cfg, _inputs, _identity, signature = _ablation_identity(module, module._test_base, tmp_path)
    out_dir = tmp_path / "artifacts/reports/visibility/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = tmp_path / "artifacts/checkpoints/visibility_ablations/alpha"
    transaction_helper = _load(
        "tests/unit/test_visibility_checkpoint_transactions.py", "visibility_tampered_journal_fixture"
    )
    transaction_helper._publish_interrupted_pair(checkpoint_dir, monkeypatch, "last.pt")
    transaction = checkpoint_dir / ".visibility_checkpoint_transaction"
    journal = transaction / "journal.json"
    journal.write_text("{}", encoding="utf-8")
    journal_bytes = journal.read_bytes()
    formal_before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in checkpoint_dir.glob("*.pt")
    }
    progress_path = out_dir / "training_progress.json"
    progress_path.write_text(json.dumps({
        "identity": {"alpha": signature}, "groups": {"alpha": {"status": "paused"}},
    }), encoding="utf-8")
    results_path = out_dir / "results.json"
    results_path.write_text("{}", encoding="utf-8")
    progress_before, results_before = progress_path.read_bytes(), results_path.read_bytes()
    calls = []
    monkeypatch.setattr(module, "VisibilityTrainer", lambda *_args, **_kwargs: calls.append(True))

    with pytest.raises(ValueError, match="journal 无效"):
        module.main()

    assert calls == []
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in checkpoint_dir.glob("*.pt")} == formal_before
    assert transaction.is_dir() and journal.read_bytes() == journal_bytes
    assert progress_path.read_bytes() == progress_before
    assert results_path.read_bytes() == results_before

