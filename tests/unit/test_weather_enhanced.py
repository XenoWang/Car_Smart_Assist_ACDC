"""仅天气微调：原模型保护、掩码、增强和续训。"""
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from PIL import Image

from car_smart_assist.config.weather import load_weather_config
from car_smart_assist.data.preprocessing import load_invalid_entries
from car_smart_assist.perception.weather import (
    FEATURE_NAMES,
    WeatherClassifier,
    WeatherPredictor,
    visual_cues,
)
from car_smart_assist.perception.weather_enhanced_training import (
    EnhancedDataset,
    distillation_loss,
    enhanced_splits,
    retained,
    train_enhanced_weather,
)
from car_smart_assist.perception.weather_training import (
    WeatherDataset,
    WeatherSample,
    checkpoint_model_config,
    list_condition_images,
)


def enhanced_config():
    cfg = load_weather_config(Path(__file__).resolve().parents[2] / 'configs/model/weather_enhanced.yaml')
    train = {**cfg.train, 'acdc_root': 'acdc', 'cache_dir': 'cache', 'epochs': 1,
             'resume_extra_epochs': 1, 'batch_size': 4, 'patience': 2,
             'pixel_accurate_zip': 'pixel.zip',
             'enhanced': {**cfg.train['enhanced'], 'teacher_checkpoint': 'teacher.pt', 'report_dir': 'reports'}}
    return replace(cfg, image_size=(16, 24), channels=(4, 4, 8), dropout=0.2,
                   device='cpu', checkpoint='candidate/best.pt', train=train)


def write_data(root, cfg):
    for condition in cfg.attributes:
        for i in range(3):
            path = root / 'acdc/rgb_anon' / condition / ('train' if i < 2 else 'val') / f'{condition}{i}'
            path.mkdir(parents=True)
            Image.fromarray(np.full((16, 24, 3), 30 + i * 30, np.uint8)).save(path / 'frame_rgb_anon.png')
    with ZipFile(root / 'pixel.zip', 'w') as archive:
        import io

        for scene in range(1, 5):
            for illumination, condition in (('day', 'clear'), ('night', 'clear'), ('day', 'fog20'), ('day', 'rain55')):
                stream = io.BytesIO()
                Image.fromarray(np.full((16, 24, 3), scene * 40, np.uint8)).save(stream, format='PNG')
                archive.writestr(f'scene{scene}_{illumination}_{condition}_0.png', stream.getvalue())
    teacher_cfg = replace(cfg, rain_adapter_channels=0)
    model = WeatherClassifier(teacher_cfg)
    torch.save({'format_version': 2, 'model_config': checkpoint_model_config(teacher_cfg),
                'model_state': model.state_dict()}, root / 'teacher.pt')


def test_recording_family_and_corrupt_manifest(tmp_path):
    cfg = enhanced_config()
    for sequence in ('GOPR0400', 'GP010400'):
        folder = tmp_path / 'acdc/rgb_anon/rain/train' / sequence
        folder.mkdir(parents=True)
        (folder / 'frame_rgb_anon.png').write_bytes(b'broken')
    records = list_condition_images(tmp_path / 'acdc', 'train', cfg.attributes)
    assert len({s.group for s in records}) == 1
    invalid = {str(records[0].path.resolve())}
    assert len(list_condition_images(tmp_path / 'acdc', 'train', cfg.attributes, invalid)) == 1
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"invalid": ["pixel.zip::bad.png"], "suspect": ["keep.png"]}', encoding='utf-8')
    assert load_invalid_entries(manifest, tmp_path) == {str((tmp_path / 'pixel.zip').resolve()) + '::bad.png'}


def test_feature_configuration_invalidates_cache_and_augmented_cues_match(tmp_path):
    cfg = enhanced_config()
    path = tmp_path / 'image.png'
    Image.fromarray(np.full((16, 24, 3), 190, np.uint8)).save(path)
    sample = WeatherSample((0, 1, 0, 0), 'one', 'rain', path=path)
    base = WeatherDataset([sample], cfg, tmp_path / 'cache.npy')
    old_signature = base.signature
    changed = replace(cfg, features={**cfg.features, 'bright_threshold': 0.95})
    rebuilt = WeatherDataset([sample], changed, tmp_path / 'cache.npy')
    assert rebuilt.signature != old_signature
    options = {**cfg.train['enhanced'], 'augmentation_probability': 1,
               'brightness_range': [0.9, 0.9], 'contrast_range': [1, 1]}
    dataset = EnhancedDataset(rebuilt, options)
    pixels, cues, _, mask, original, _ = dataset[0]
    array = pixels.permute(1, 2, 0).mul(255).round().byte().numpy()
    expected = visual_cues(array, changed)
    assert torch.allclose(cues, torch.tensor([expected[n] for n in FEATURE_NAMES]))
    assert not torch.equal(pixels, original)
    assert mask.tolist() == [0, 1, 0, 0]


def test_wrong_teacher_rain_is_not_distilled():
    cfg = enhanced_config()
    options = cfg.train['enhanced']
    student = torch.zeros(1, 4, requires_grad=True)
    teacher = torch.tensor([[-6., -6., -6., -6.]])
    labels = torch.tensor([[0., 1., 0., 0.]])
    loss = distillation_loss(student, teacher, labels, torch.ones_like(labels), cfg, options)
    loss.backward()
    assert student.grad[0, 1] == 0  # 原模型漏检了雨，这里应由监督损失纠正。
    assert student.grad[0, 0] != 0


def test_source_preservation_rejects_nonrain_regression():
    cfg = enhanced_config()
    attribute = {'support': 10, 'f1': 0.8, 'recall': 0.8, 'false_positive_rate': 0.1}
    baseline = {'acdc': {'macro_f1': .8, 'per_attribute': {a: dict(attribute) for a in cfg.attributes}}}
    import copy

    candidate = copy.deepcopy(baseline)
    candidate['acdc']['per_attribute']['fog']['recall'] = .7
    assert not retained(candidate, baseline, cfg.train['enhanced'])[0]


def test_real_enhanced_train_resume_and_teacher_preserved(tmp_path):
    cfg = enhanced_config()
    write_data(tmp_path, cfg)
    before = (tmp_path / 'teacher.pt').read_bytes()
    splits = enhanced_splits(cfg, tmp_path)
    assert not ({s.group for s in splits['train']} & {s.group for s in splits['validation']})
    first = train_enhanced_weather(cfg, tmp_path)
    assert first['trained_through_epoch'] == 1
    second = train_enhanced_weather(cfg, tmp_path)
    assert second['trained_through_epoch'] == 2
    assert (tmp_path / 'teacher.pt').read_bytes() == before
    assert (tmp_path / 'candidate/last.pt').is_file()
    predictor = WeatherPredictor.from_checkpoint(tmp_path / 'candidate/best.pt', cfg)
    assert set(predictor.predict(np.zeros((16, 24, 3), np.uint8)).probabilities) == set(cfg.attributes)
    teacher_state = torch.load(tmp_path / 'teacher.pt', weights_only=True)['model_state']
    resumed_state = torch.load(tmp_path / 'candidate/last.pt', weights_only=True)
    for name, value in teacher_state.items():
        assert torch.equal(value, resumed_state['model_state'][name])
    assert torch.count_nonzero(resumed_state['model_state']['rain_adapter.2.weight']) > 0
    teacher = WeatherPredictor.from_checkpoint(tmp_path / 'teacher.pt', replace(cfg, rain_adapter_channels=0))
    last = WeatherPredictor.from_checkpoint(tmp_path / 'candidate/last.pt', cfg)
    frame = np.random.default_rng(42).integers(0, 256, (16, 24, 3), dtype=np.uint8)
    for name in ('fog', 'snow', 'night'):
        assert teacher.predict(frame).probabilities[name] == last.predict(frame).probabilities[name]
    uninterrupted = replace(cfg, checkpoint='uninterrupted/best.pt', train={**cfg.train, 'epochs': 2})
    train_enhanced_weather(uninterrupted, tmp_path)
    continuous = torch.load(tmp_path / 'uninterrupted/last.pt', weights_only=True)
    for name, value in resumed_state['model_state'].items():
        assert torch.equal(value, continuous['model_state'][name])
    with pytest.raises(ValueError, match='策略变化'):
        train_enhanced_weather(replace(cfg, rain_adapter_channels=8), tmp_path)
    torch.manual_seed(73)
    external_rng = torch.get_rng_state().clone()
    train_enhanced_weather(cfg, tmp_path)
    assert torch.equal(external_rng, torch.get_rng_state())
    with pytest.raises(ValueError, match='原模型'):
        train_enhanced_weather(replace(cfg, checkpoint='teacher.pt'), tmp_path)


def test_completed_early_stop_resumes_a_new_round(tmp_path):
    cfg = enhanced_config()
    write_data(tmp_path, cfg)
    train_enhanced_weather(cfg, tmp_path)
    path = tmp_path / 'candidate/last.pt'
    state = torch.load(path, weights_only=True)
    state['planned_epochs'] = 5
    state['training_complete'] = True
    state['wait'] = cfg.train['patience']
    torch.save(state, path)
    cfg = replace(cfg, train={**cfg.train, 'epochs': 5, 'resume_extra_epochs': 3, 'patience': 100})
    result = train_enhanced_weather(cfg, tmp_path)
    assert result['trained_through_epoch'] == 4
