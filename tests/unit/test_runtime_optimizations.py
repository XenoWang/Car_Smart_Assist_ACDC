"""回归覆盖：缓存跨进程传递、张量拷贝、训练汇总与推理配置一致性。"""

import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from car_smart_assist.inference.pipeline import InferencePipeline
from car_smart_assist.perception.visibility.dataset import VisibilityImageDataset
from car_smart_assist.perception.visibility.scorer import VisibilityScorer
from car_smart_assist.perception.visibility.trainer import VisibilityTrainer


@pytest.mark.parametrize("augment", [False, True])
def test_dataset_copy_preserves_pixels_and_cache(tmp_path, augment):
    pixels = np.random.default_rng(17).integers(0, 256, (2, 16, 32, 3), dtype=np.uint8)
    cache = tmp_path / "images.npy"
    np.save(cache, pixels)
    dataset = VisibilityImageDataset(
        [tmp_path / "0.png", tmp_path / "1.png"], (16, 32),
        cache_path=cache, augment=augment,
    )
    # 固定随机状态只在上下文内有效，比较优化前后的光度增强与布局。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        actual = dataset[1]
        torch.manual_seed(7)
        expected = torch.from_numpy(pixels[1].astype(np.float32)).div_(255.0)
        if augment:
            expected = (expected - 0.5) * float(torch.empty(1).uniform_(0.9, 1.1)) + 0.5
            expected = expected + float(torch.empty(1).uniform_(-0.03, 0.03))
            expected.clamp_(0, 1)
    torch.testing.assert_close(actual, expected.permute(2, 0, 1), rtol=0, atol=0)
    assert actual.is_contiguous()
    actual.zero_()
    np.testing.assert_array_equal(np.load(cache), pixels)


def test_opened_dataset_remains_small_when_pickled(tmp_path):
    pixels = np.zeros((200, 64, 64, 3), dtype=np.uint8)
    cache = tmp_path / "images.npy"
    np.save(cache, pixels)
    dataset = VisibilityImageDataset(
        [tmp_path / f"{i}.png" for i in range(200)], (64, 64), cache_path=cache,
    )
    before = len(pickle.dumps(dataset))
    expected = dataset[1]
    blob = pickle.dumps(dataset)
    assert len(blob) == before
    assert len(blob) < pixels.nbytes * 0.05
    restored = pickle.loads(blob)
    assert restored._cache is None
    torch.testing.assert_close(restored[1], expected)
    assert dataset._cache is not None


def test_pipeline_uses_checkpoint_size_and_scoring_fallback(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    scorer = VisibilityScorer(torch.nn.Identity(), input_size=(16, 32))
    received = {}

    def load(*args, **kwargs):
        received.update(kwargs)
        return scorer

    monkeypatch.setattr(VisibilityScorer, "from_checkpoint", load)
    pipeline = InferencePipeline.from_config(
        {"model": {"input_size": [144, 256]}, "device": "cpu"},
        checkpoint=checkpoint,
    )
    assert pipeline.gate_input_size == (16, 32)
    assert received["scoring_cfg"] is None
    assert pipeline._resize_for_gate(np.zeros((48, 64, 3), dtype=np.uint8)).shape == (16, 32, 3)


def test_eval_loss_weights_tail_batch_and_rejects_empty():
    trainer = VisibilityTrainer.__new__(VisibilityTrainer)
    trainer.device = torch.device("cpu")
    # 恒零模型的误差由输入确定，尾部 batch 必须只算一个样本。
    trainer.model = torch.nn.Conv2d(3, 3, 1, bias=False)
    torch.nn.init.zeros_(trainer.model.weight)
    samples = torch.arange(5, dtype=torch.float32)[:, None, None, None].expand(5, 3, 4, 4)
    assert trainer._eval_loss(DataLoader(samples, batch_size=2)) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="验证数据为空"):
        trainer._eval_loss(DataLoader(samples[:0], batch_size=2))


def test_short_training_saves_and_calibrates_best_once(tmp_path, monkeypatch):
    cfg = {
        "device": "cpu",
        "model": {"input_size": [16, 16], "base_channels": 2, "latent_dim": 4},
        "data": {"num_workers": 0},
        "train": {
            "epochs": 2, "batch_size": 2, "amp": False,
            "checkpoint_dir": str(tmp_path), "early_stopping": {"enabled": False},
        },
    }
    trainer = VisibilityTrainer(cfg, resume="none")
    trainer.model = torch.nn.Conv2d(3, 3, 1)
    samples = torch.linspace(0, 1, 5 * 3 * 4 * 4).reshape(5, 3, 4, 4)
    monkeypatch.setattr(trainer, "_build_datasets", lambda: (samples, samples[:3], samples[2:], []))
    original_load = torch.load
    loads = []

    def load(path, **kwargs):
        loads.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    history = trainer.fit()
    assert len(history.train_loss) == len(history.val_loss) == 2
    assert np.isfinite(history.train_loss + history.val_loss).all()
    assert trainer.calibration.n == 3
    assert loads == [tmp_path / "best.pt"]
    saved = original_load(tmp_path / "best.pt", weights_only=False)
    assert saved["calibration"] == trainer.calibration.to_dict()
    assert (tmp_path / "last.pt").is_file()
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, saved["model_state"][key])
