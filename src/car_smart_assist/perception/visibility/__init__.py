"能见度门控：无监督判断「这一帧是否已经看不清」。"

from car_smart_assist.perception.visibility.autoencoder import (
    DEFAULT_KERNEL_SIZES,
    ConvAutoencoder,
    MultiScaleBlock,
    ReconstructionError,
    build_autoencoder,
    effective_kernel_size,
    reconstruction_error,
)
from car_smart_assist.perception.visibility.degradation import degrade, severity_sweep
from car_smart_assist.perception.visibility.gate import (
    GateThresholds,
    VisibilityGate,
    VisibilityLevel,
    VisibilityVerdict,
)
from car_smart_assist.perception.visibility.scorer import (
    CalibrationStats,
    InformationFeatures,
    VisibilityScore,
    VisibilityScorer,
    compute_information_features,
    information_score,
)
from car_smart_assist.perception.visibility.trainer import (
    TrainHistory,
    VisibilityTrainer,
    resolve_device,
)

__all__ = [
    "DEFAULT_KERNEL_SIZES",
    "CalibrationStats",
    "ConvAutoencoder",
    "GateThresholds",
    "InformationFeatures",
    "MultiScaleBlock",
    "ReconstructionError",
    "TrainHistory",
    "VisibilityGate",
    "VisibilityLevel",
    "VisibilityScore",
    "VisibilityScorer",
    "VisibilityTrainer",
    "VisibilityVerdict",
    "build_autoencoder",
    "compute_information_features",
    "degrade",
    "effective_kernel_size",
    "information_score",
    "reconstruction_error",
    "resolve_device",
    "severity_sweep",
]
