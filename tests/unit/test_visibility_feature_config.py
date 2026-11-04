"""Feature configuration must fail before pixels/model work, with analytic valid controls."""
import numpy as np
import pytest
import torch

from car_smart_assist.perception.visibility import scorer as module


BAD_THRESHOLDS = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(-float("inf"), id="negative-infinity"),
    pytest.param(-0.1, id="negative"),
    pytest.param(True, id="boolean"),
    pytest.param(np.bool_(False), id="numpy-boolean"),
    pytest.param(None, id="null"),
    pytest.param([], id="list"),
    pytest.param({}, id="mapping"),
    pytest.param("0.04", id="numeric-string"),
    pytest.param(10 ** 400, id="float-overflow"),
]


@pytest.mark.parametrize("field", ["edge_gradient_threshold", "hf_cutoff_ratio"])
@pytest.mark.parametrize("bad", BAD_THRESHOLDS)
def test_feature_threshold_rejection_precedes_image_conversion(monkeypatch, field, bad):
    def unexpected_decode(*args, **kwargs):
        raise AssertionError("invalid scoring configuration reached image conversion")

    monkeypatch.setattr(module, "to_rgb_uint8", unexpected_decode)
    with pytest.raises(ValueError, match=field):
        module.compute_information_features(
            np.zeros((8, 8, 3), np.uint8), {"features": {field: bad}}
        )


@pytest.mark.parametrize("field", ["edge_gradient_threshold", "hf_cutoff_ratio"])
@pytest.mark.parametrize("bad", BAD_THRESHOLDS)
def test_scorer_rejects_feature_threshold_before_model_transfer(field, bad):
    class UntouchedModel:
        def to(self, *args, **kwargs):
            raise AssertionError("invalid scoring configuration reached model transfer")

    with pytest.raises(ValueError, match=field):
        module.VisibilityScorer(UntouchedModel(), scoring_cfg={"features": {field: bad}})


@pytest.mark.parametrize("config,reason", [
    ([], "scoring"), (False, "scoring"), ("invalid", "scoring"),
    ({"features": None}, "features"), ({"features": []}, "features"),
    ({"features": False}, "features"),
])
def test_invalid_feature_container_is_rejected_at_both_public_entries(config, reason):
    image = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match=reason):
        module.compute_information_features(image, config)
    with pytest.raises(ValueError, match=reason):
        module.VisibilityScorer(torch.nn.Identity(), scoring_cfg=config)


@pytest.mark.parametrize("edge,cutoff,expected_edges,expected_high", [
    (0, 0, .25, 1.),
    (2., 2., 0., 0.),
    (np.float64(.04), np.float64(2.), .25, 0.),
])
def test_legal_zero_and_large_feature_thresholds_keep_known_binary_step(
    edge, cutoff, expected_edges, expected_high
):
    # Equal black/white populations have sigma=.5 and one bit of entropy.
    # The step spans two central-difference columns out of eight.
    image = np.zeros((8, 8, 3), np.uint8)
    image[:, 4:] = 255
    config = {"features": {"edge_gradient_threshold": edge, "hf_cutoff_ratio": cutoff}}
    result = module.compute_information_features(image, config)
    assert result.contrast == pytest.approx(.5, rel=0, abs=1e-7)
    assert result.entropy == pytest.approx(1., rel=0, abs=1e-7)
    assert result.edge_density == expected_edges
    assert result.hf_ratio == pytest.approx(expected_high, rel=0, abs=1e-7)
    scorer = module.VisibilityScorer(torch.nn.Identity(), scoring_cfg=config)
    assert scorer.scoring_cfg == config
