"""Content identity survives same-stat replacement, and flags require booleans."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from _model_test_helpers import load_module
from car_smart_assist.perception.visibility.dataset import (
    ensure_cache, reference_split_signature, split_ref_indices)
from car_smart_assist.perception.visibility.gate import GateThresholds, VisibilityGate


@pytest.mark.parametrize("entry", ["thresholds_config", "thresholds_direct", "gate_config", "gate_direct"])
@pytest.mark.parametrize("value", ["false", "true", "", 0, 1, None, [], {}])
def test_nonboolean_switches_are_rejected(entry, value):
    with pytest.raises(ValueError, match="use_recon_z|require_calibration"):
        if entry == "thresholds_config":
            GateThresholds.from_config({"use_recon_z": value})
        elif entry == "thresholds_direct":
            GateThresholds(use_recon_z=value)
        elif entry == "gate_config":
            VisibilityGate.from_config({"require_calibration": value})
        else:
            VisibilityGate(require_calibration=value)


@pytest.mark.parametrize("value", [False, True])
def test_real_booleans_keep_their_exact_meaning(value):
    assert GateThresholds.from_config({"use_recon_z": value}).use_recon_z is value
    assert VisibilityGate.from_config({"require_calibration": value}).require_calibration is value
    assert GateThresholds().use_recon_z is False


def test_same_stat_image_replacement_changes_signature_cache_and_rejects_evaluation(tmp_path):
    paths = [tmp_path / f"{i}.bmp" for i in range(20)]
    for path in paths:
        Image.new("RGB", (8, 8), (0, 0, 0)).save(path)
    parts = split_ref_indices(len(paths), seed=42, groups=None)
    signature = reference_split_signature(paths, tmp_path, (8, 8), None, *parts)
    assert reference_split_signature(paths, tmp_path, (8, 8), None, *parts) == signature
    cache = ensure_cache(paths, (8, 8), tmp_path / "cache.npy")
    assert np.all(np.load(cache, allow_pickle=False) == 0)
    stat, original = paths[0].stat(), paths[0].read_bytes()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(paths[0])
    os.utime(paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert paths[0].stat().st_size == stat.st_size
    assert paths[0].stat().st_mtime_ns == stat.st_mtime_ns
    assert paths[0].read_bytes() != original
    assert reference_split_signature(paths, tmp_path, (8, 8), None, *parts) != signature
    rebuilt = ensure_cache(paths, (8, 8), cache)
    pixels = np.load(rebuilt, allow_pickle=False)
    assert np.all(pixels[0] == 255) and np.all(pixels[1:] == 0)
    module = load_module("scripts/evaluate_visibility.py", "same_stat_evaluation")
    scorer = SimpleNamespace(input_size=(8, 8), split_signature=signature,
                             score_arrays=lambda *_: pytest.fail("stale identity reached scoring"))
    with pytest.raises(module.SplitIdentityError, match="身份不一致"):
        module.eval_fit_quality(scorer, [], paths,
                                {"acdc_root": str(tmp_path), "split_by_sequence": False}, None, 1)


def test_restore_rejects_changed_signature_before_loading_weights(tmp_path, monkeypatch):
    from car_smart_assist.perception.visibility.trainer import VisibilityTrainer
    from test_visibility_retrain_training import _config
    import car_smart_assist.perception.visibility.trainer as training
    import torch

    trainer = VisibilityTrainer(_config(tmp_path), project_root=tmp_path, resume="none")
    trainer.resume_path = tmp_path / "synthetic.pt"
    trainer.split_signature = "current-content"
    monkeypatch.setattr(training, "_load_visibility_checkpoint", lambda _: {"split_signature": "old-content"})
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    opt = torch.optim.AdamW(trainer.model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
    with pytest.raises(ValueError, match="数据/切分不匹配"):
        trainer._restore(opt, scheduler)
    assert all(torch.equal(value, before[key]) for key, value in trainer.model.state_dict().items())


@pytest.fixture(autouse=True)
def isolated_random_state():
    import random
    import numpy as np
    import torch
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        torch.set_num_threads(threads)
