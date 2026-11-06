"""Independent pixel expectations for diagnostic low-light augmentation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from car_smart_assist.data.transforms.lowlight import lowlight_rgb


def test_unity_exposure_roundtrips_all_bytes_without_aliasing():
    source = np.repeat(np.arange(256, dtype=np.uint8).reshape(16,16,1),3,axis=2)
    original = source.copy()
    actual = lowlight_rgb(source,1)
    np.testing.assert_array_equal(actual,original)
    assert actual is not source and not np.shares_memory(actual,source)
    np.testing.assert_array_equal(source,original)


def test_quarter_linear_exposure_has_hand_checked_srgb_pixels():
    source = np.array([[[0,64,128],[255,128,64]]],dtype=np.uint8)
    expected = np.array([[[0,30,66],[137,66,30]]],dtype=np.uint8)
    np.testing.assert_array_equal(lowlight_rgb(source,.25),expected)
    np.testing.assert_array_equal(source,[[[0,64,128],[255,128,64]]])


def test_shot_noise_is_scaled_in_the_linear_domain():
    class Counts:
        def poisson(self, mean):
            np.testing.assert_allclose(mean, np.full((1,1,3),250),rtol=0,atol=.0001)
            return np.array([[[0,250,1000]]])
    actual = lowlight_rgb(np.full((1,1,3),255,dtype=np.uint8),.25,shot_scale=.001,rng=Counts())
    np.testing.assert_array_equal(actual,[[[0,137,255]]])


def test_read_noise_uses_linear_signal_then_clips():
    class Noise:
        def normal(self, location, deviation, shape):
            assert location == 0 and deviation == .001 and shape == (1,1,3)
            return np.array([[[-.5,0,1.0]]],dtype=np.float32)
    actual = lowlight_rgb(np.full((1,1,3),255,dtype=np.uint8),.25,read_std=.001,rng=Noise())
    np.testing.assert_array_equal(actual,[[[0,137,255]]])


@pytest.mark.parametrize("options", [{"exposure":0},{"exposure":1.1},{"exposure":True},
                                   {"exposure":float("nan")},{"exposure":.5,"shot_scale":-1},
                                   {"exposure":.5,"read_std":.11},{"exposure":.5,"shot_scale":1e-10},
                                   {"exposure":.5,"read_std":.001}])
def test_invalid_or_unseeded_noise_is_rejected(options):
    with pytest.raises(ValueError):
        lowlight_rgb(np.ones((2,2,3),dtype=np.uint8),**options)


@pytest.mark.parametrize("source",[np.zeros((1,1,3)),np.zeros((1,1,4),dtype=np.uint8),
                                   np.zeros((1,1),dtype=np.uint8),np.zeros((0,1,3),dtype=np.uint8)])
def test_input_pixel_contract_rejects_unsupported_arrays(source):
    with pytest.raises(ValueError,match="RGB"):
        lowlight_rgb(source,.5)


def test_seed_reproducibility_and_distinct_channels():
    source = np.full((12,10,3),180,dtype=np.uint8)
    a = lowlight_rgb(source,.5,shot_scale=.001,read_std=.001,rng=np.random.default_rng(7))
    b = lowlight_rgb(source,.5,shot_scale=.001,read_std=.001,rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a,b)
    assert np.unique(a).size > 1 and a.dtype==np.uint8 and a.flags.c_contiguous
    np.testing.assert_array_equal(source,np.full((12,10,3),180,dtype=np.uint8))


def test_selection_preserves_original_ids_and_quantile_reasons():
    file=Path(__file__).resolve().parents[2]/"scripts/preview_yolo_lowlight.py"
    spec=importlib.util.spec_from_file_location("lowlight_preview",file)
    script=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    records=[{"image":f"night-{i}.png","condition":"night","mean":value,
              "targets":[{"tiny":True,"mean":value,"index":0}]} for i,value in enumerate((10,50,100))]
    selected=script.choose_samples(records)
    assert set(selected)=={"night-0.png","night-1.png","night-2.png"}
    assert selected["night-1.png"]==["night frame mean q0.5"]
    assert selected["night-0.png"]==["night frame mean q0.1","night tiny mean q0.1, target 0"]
