"""回归测试，覆盖这些点：缓存跨进程传递、张量拷贝、训练汇总、推理配置是否一致。"""

import pickle

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from car_smart_assist.inference.pipeline import InferencePipeline
from car_smart_assist.perception.visibility.dataset import VisibilityImageDataset, read_rgb_image
from car_smart_assist.perception.visibility.scorer import CalibrationStats, VisibilityScorer
from car_smart_assist.perception.visibility.trainer import VisibilityTrainer


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("size", [None, (8, 12)])
def test_shared_image_reader_preserves_known_channel_values(tmp_path, channels, size):
    # Constant fields remain constant under bilinear resizing. RGBA drops alpha
    # by contract; it must not composite the RGB values against a background.
    if channels == 1:
        pixels = np.full((17, 23), 85, np.uint8)
        rgb = [85, 85, 85]
    else:
        values = [17, 85, 170] + ([0] if channels == 4 else [])
        pixels = np.broadcast_to(np.array(values, np.uint8), (17, 23, channels)).copy()
        rgb = [17, 85, 170]
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    actual = read_rgb_image(path, size)
    path.unlink()  # Windows file handles must be closed.
    shape = (17, 23) if size is None else size
    expected = np.broadcast_to(np.array(rgb, np.uint8), (*shape, 3))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8


def test_shared_image_reader_bilinear_reduction_has_known_center_average(tmp_path):
    # Four samples contribute equally at the sole output pixel.
    pixels = np.array([[[0, 64, 255], [0, 64, 255]],
                       [[255, 64, 0], [255, 64, 0]]], np.uint8)
    path = tmp_path / "step.png"
    Image.fromarray(pixels).save(path)
    np.testing.assert_array_equal(read_rgb_image(path, (1, 1)),
                                  np.array([[[128, 64, 128]]], np.uint8))


def test_shared_calibration_statistics():
    stats = CalibrationStats.from_errors([1, 2, 3, 4, 5])
    assert stats.to_dict() == pytest.approx(
        {"mean": 3, "std": 2 ** 0.5, "p95": 4.8, "p99": 4.96, "n": 5}
    )


@pytest.mark.parametrize("errors", [[], [float("nan")], [float("inf")], [[1, 2]]])
def test_shared_calibration_rejects_invalid_errors(errors):
    with pytest.raises(ValueError, match="校准"):
        CalibrationStats.from_errors(errors)


@pytest.mark.parametrize("augment,contrast,brightness,expected_levels", [
    (False, 1., 0., [0., 1 / 3, 2 / 3, 1.]),
    (True, 1.5, -.25, [0., 0., .5, 1.]),
    (True, .75, .125, [.25, .5, .75, 1.]),
])
def test_dataset_copy_preserves_pixels_and_cache(
    tmp_path, augment, contrast, brightness, expected_levels
):
    # Fixed ranges select exact parameters through the public configuration.
    # RGB ramps have hand-computed outputs, including both clipping boundaries.
    pixels = np.broadcast_to(
        np.array([0, 85, 170, 255], np.uint8)[None, None, :, None], (2, 16, 4, 3)
    ).copy()
    cache = tmp_path / "images.npy"
    np.save(cache, pixels)
    dataset = VisibilityImageDataset(
        [tmp_path / "0.png", tmp_path / "1.png"], (16, 4),
        cache_path=cache, augment=augment,
        augmentation_cfg={"contrast_range": (contrast, contrast),
                          "brightness_range": (brightness, brightness)},
    )
    with torch.random.fork_rng(devices=[]):
        actual = dataset[1]
    expected = torch.tensor(expected_levels).reshape(1, 1, 4).expand(3, 16, 4)
    # At most a few float32 rounding steps; no fitted execution output.
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
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
    # Input is explicitly all black; serialization must preserve that value.
    expected = torch.zeros(3, 64, 64)
    torch.testing.assert_close(dataset[1], expected, rtol=0, atol=0)
    blob = pickle.dumps(dataset)
    assert len(blob) == before
    assert len(blob) < pixels.nbytes * 0.05
    restored = pickle.loads(blob)
    torch.testing.assert_close(restored[1], expected)


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


def test_trainer_calibration_matches_known_zero_model_errors():
    trainer = VisibilityTrainer.__new__(VisibilityTrainer)
    trainer.device = torch.device("cpu")
    trainer.model = torch.nn.Conv2d(3, 3, 1, bias=False)
    torch.nn.init.zeros_(trainer.model.weight)
    levels = torch.tensor([0.0, 0.5, 0.5, 1.0, 1.0])
    samples = levels[:, None, None, None].expand(5, 3, 2, 2)
    stats = trainer._calibrate(DataLoader(samples, batch_size=2))
    # 恒零输出的逐图 MSE 为 [0, .25, .25, 1, 1]；均值 .5，总体方差 .175，
    # 第95/99百分位都为 1，且跨三个 batch 合计五个样本。
    assert stats.to_dict() == pytest.approx(
        {"mean": 0.5, "std": 0.175**0.5, "p95": 1.0, "p99": 1.0, "n": 5}
    )


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
