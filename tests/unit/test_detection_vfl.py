"""Independent VFL arithmetic, gradients, sparse supervision and legacy BCE behavior."""
import io
import math

import pytest
import torch

from car_smart_assist.perception.joint_yolo import MaskedClassificationLoss, JointDetectionLoss


def configured(recipe=None, *, exclusive=True, device="cpu", dtype=torch.float64):
    loss = MaskedClassificationLoss(exclusive, classification_loss=recipe)
    loss.valid = torch.ones(1, 1, 3, dtype=torch.bool, device=device)
    loss.weights = torch.full((1, 1, 1), 2.0, dtype=dtype, device=device)
    return loss


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_vfl_hand_loss_and_complete_gradient_include_negative_weight_derivative(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    logits = torch.zeros(1, 1, 3, dtype=torch.float64, device=device, requires_grad=True)
    targets = torch.tensor([[[1., .5, 0.]]], dtype=torch.float64, device=device)
    loss = configured({"mode": "vfl"}, device=device)
    actual = loss(logits, targets)
    # p=1/2; source weight2. VFL weights are1,1/2,3/16.
    expected = torch.tensor([[[2*math.log(2), math.log(2), 3/8*math.log(2)]]], dtype=torch.float64, device=device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert actual.shape == (1, 1, 3) and actual.dtype == torch.float64
    actual.sum().backward()
    # Negative derivative includes d(p^2)/dz; detaching the focal weight fails this oracle.
    gradient = torch.tensor([[[-1., 0., 3/8*math.log(2)+3/16]]], dtype=torch.float64, device=device)
    torch.testing.assert_close(logits.grad, gradient, rtol=0, atol=1e-12)


@pytest.mark.parametrize("exclusive", [False, True])
def test_vfl_preserves_unknown_roi_mask_positive_override_and_source_weight(exclusive):
    x = torch.zeros(2, 3, 3, dtype=torch.float64, requires_grad=True)
    q = torch.zeros_like(x)
    q[1, 2, 2] = 1
    loss = MaskedClassificationLoss(exclusive, classification_loss={"mode": "vfl"})
    loss.valid = torch.tensor([[[True, True, False]]*3,
                               [[False, False, True], [False, False, False], [False, False, False]]])
    loss.weights = torch.tensor([[[1.]], [[4.]]], dtype=torch.float64)
    values = loss(x, q)
    values.sum().backward()
    assert values.shape == x.shape
    assert torch.count_nonzero(x.grad[0, :, 2]) == 0
    assert torch.count_nonzero(x.grad[1, :2, :2]) == 0
    assert x.grad[1, 1, 2].item() == 0
    assert x.grad[1, 2, 2].item() == -2  # Positive outside ROI is retained: 4*(.5-1).
    negative_derivative = 3/16*math.log(2)+3/32
    assert x.grad[1, 0, 2].item() == pytest.approx(4*negative_derivative, abs=1e-12)
    if exclusive:
        assert x.grad[1, 2, 0].item() == pytest.approx(4*negative_derivative, abs=1e-12)
    else:
        assert torch.count_nonzero(x.grad[1, 2, :2]) == 0


def test_alpha_one_gamma_zero_matches_bce_for_hard_labels():
    targets = torch.tensor([[[1., 0., 1.]]], dtype=torch.float64)
    a = torch.tensor([[[-2., .7, 3.]]], dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    plain = configured()(a, targets)
    vfl = configured({"mode": "vfl", "alpha": 1, "gamma": 0})(b, targets)
    assert torch.equal(plain, vfl)
    plain.sum().backward()
    vfl.sum().backward()
    assert torch.equal(a.grad, b.grad)


def test_zero_alpha_keeps_soft_positives_and_removes_negative_gradient():
    x = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[[1., .25, 0.]]], dtype=torch.float64)
    actual = configured({"mode": "vfl", "alpha": 0, "gamma": 0})(x, q)
    torch.testing.assert_close(actual, torch.tensor([[[2*math.log(2), .5*math.log(2), 0.]]], dtype=torch.float64),
                               rtol=0, atol=1e-12)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[[-1., .125, 0.]]], dtype=torch.float64), rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode", [None, {"mode": "bce"}])
def test_default_and_explicit_bce_retain_hand_computed_loss_and_gradient(mode):
    x = torch.tensor([[[0., math.log(3), 0.]]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[[1., 0., .5]]], dtype=torch.float64)
    actual = configured(mode)(x, q)
    expected = torch.tensor([[[2*math.log(2), 2*math.log(4), 2*math.log(2)]]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[[-1., 1.5, 0.]]], dtype=torch.float64), rtol=0, atol=1e-12)


def test_old_pickled_loss_without_recipe_retains_bce_abi():
    old = object.__new__(MaskedClassificationLoss)
    torch.nn.Module.__init__(old)
    old.valid = torch.ones(1, 1, 3, dtype=torch.bool)
    old.weights = torch.full((1, 1, 1), 2., dtype=torch.float64)
    old.foreground_exclusive = True
    stream = io.BytesIO()
    torch.save(old, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert not hasattr(restored, "classification_loss")
    x = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
    restored(x, torch.tensor([[[1., 0., .5]]], dtype=torch.float64)).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[[-1., 1., 0.]]], dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("device,dtype", [("cpu", torch.bfloat16), ("cuda", torch.float16), ("cuda", torch.bfloat16)])
def test_low_precision_extreme_logits_use_finite_fp32_work(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 unavailable")
    x = torch.tensor([[[-100., 100., 0.]]], dtype=dtype, device=device, requires_grad=True)
    q = torch.tensor([[[1., 0., .5]]], dtype=dtype, device=device)
    loss = configured({"mode": "vfl"}, device=device, dtype=torch.float32)
    with torch.autocast(device_type=device, dtype=dtype):
        actual = loss(x, q)
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    # softplus(100) differs from100 by<4e-44; sigmoid(100) rounds to1 in FP32.
    expected = torch.tensor([[[200., 150., math.log(2)]]], device=device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-5)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad.float(), torch.tensor([[[-2., 1.5, 0.]]], device=device), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_tiny_joint_network_changes_only_class_loss_and_backpropagates(device):
    from types import SimpleNamespace
    from ultralytics.nn.tasks import DetectionModel
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.random.fork_rng(devices=[0] if device == "cuda" else []):
        torch.manual_seed(42)
        architecture = {
            "nc": 9,
            "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [16, 3, 2]],
                         [-1, 1, "Conv", [32, 3, 2]]],
            "head": [[[2], 1, "Detect", [9]]],
        }
        model = DetectionModel(architecture, ch=3, nc=9, verbose=False).to(device)
        model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
        model.train()
        images = torch.linspace(0, 1, 2*3*64*64, device=device).reshape(2, 3, 64, 64)
        batch = {"img": images, "batch_idx": torch.tensor([0., 1.], device=device),
                 "cls": torch.tensor([[0.], [8.]], device=device),
                 "bboxes": torch.tensor([[.5, .5, .5, .5], [.3, .7, .25, .25]], device=device),
                 "class_supervision": torch.ones(2, 9, dtype=torch.bool, device=device),
                 "obstacle_roi": torch.ones(2, 64, 64, dtype=torch.bool, device=device),
                 "source_loss_weight": torch.tensor([1., 4.], device=device)}
        outputs = model(images)
        bce = JointDetectionLoss(model, 8)
        vfl = JointDetectionLoss(model, 8, classification_loss={"mode": "vfl"})
        _, old_parts = bce(outputs, batch)
        total, new_parts = vfl(outputs, batch)
        assert old_parts.shape == new_parts.shape == (3,)
        assert torch.equal(old_parts[[0, 2]], new_parts[[0, 2]])
        assert 0 < new_parts[1] < old_parts[1]
        total.sum().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g).item() > 0 for g in gradients)
        before = [p.detach().clone() for p in model.parameters()]
        torch.optim.SGD(model.parameters(), lr=.001).step()
        assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
        assert vfl.base.bce.valid is None and vfl.base.bce.weights is None
