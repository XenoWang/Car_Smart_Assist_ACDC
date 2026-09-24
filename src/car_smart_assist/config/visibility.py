"""能见度模块的兼容默认值，供库接口与不完整配置共用。

项目运行参数统一在 configs/model/visibility.yaml 修改；这里保留既有库接口的
默认行为（例如 base_channels=32），YAML 中的实验配置优先。调用方应复制字典后修改。
"""

from copy import deepcopy

MODEL_DEFAULTS = {
    "in_channels": 3,
    "base_channels": 32,
    "latent_dim": 256,
    "input_size": (144, 256),
    "encoder_type": "multiscale",
    "kernel_sizes": (3, 5, 7),
    "dilations": None,
    "use_bottleneck": True,
    "pre_latent_channels": 32,
}

TRAIN_DEFAULTS = {
    "epochs": 40,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "optimizer": "adamw",
    "amp": True,
    "dtype": "bfloat16",
    "augment": True,
    "denoise_sigma": 0.0,
    "grad_clip_norm": 1.0,
    "early_stopping": {"enabled": True, "patience": 8, "min_delta": 1e-5},
}

AUGMENTATION_DEFAULTS = {
    "contrast_range": (0.9, 1.1),
    "brightness_range": (-0.03, 0.03),
}

SCORING_DEFAULTS = {
    "features": {"edge_gradient_threshold": 0.04, "hf_cutoff_ratio": 0.5},
    "feature_scales": {"contrast": 0.25, "entropy": 8.0, "edge_density": 0.15, "hf_ratio": 0.05},
    "aggregation": {
        "weights": {"contrast": 0.40, "entropy": 0.25, "edge_density": 0.25, "hf_ratio": 0.10},
        "power": -1.0,
        "feature_floor": 0.10,
    },
    "reconstruction_error": {"block_grid": (3, 3)},
}

GATE_DEFAULTS = {
    "thresholds": {
        "info_blind": 0.34, "info_degraded": 0.50,
        "z_degraded": 3.0, "z_blind": 8.0, "use_recon_z": False,
    },
    "require_calibration": True,
    "degraded_confidence_multiplier": 0.6,
}


def training_config(overrides: dict) -> dict:
    """补齐训练默认项并隔离嵌套字典，避免覆盖项污染其他实例。"""
    cfg = deepcopy(TRAIN_DEFAULTS)
    cfg.update(deepcopy(overrides))
    cfg["early_stopping"] = {
        **TRAIN_DEFAULTS["early_stopping"], **overrides.get("early_stopping", {}),
    }
    cfg["augmentation"] = deepcopy({**AUGMENTATION_DEFAULTS, **overrides.get("augmentation", {})})
    return cfg
