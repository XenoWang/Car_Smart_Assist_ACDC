# YOLO center-assisted loss

This optional joint-detection training objective implements the bounded center
distance term in [arXiv:2603.03788v1, Eq.8–9](https://arxiv.org/html/2603.03788v1#S2.SS4).
It adds no inference head. The default detection configuration leaves it off.

## Configuration and isolated outputs

Use the project-owned top-level section, outside vendor train arguments:

    center_loss:
      enabled: true
      weight: 0.1
      normalization: train_mean_sqrt_area

Only the joint dataset route supports this option. Non-boolean enabled values,
negative/non-finite weights, unknown fields and normalization modes are rejected.
The coefficient 0.1 is an experimental starting point, not a measured optimum.
Missing/disabled configuration follows the existing objective; zero weight
also leaves the box/DFL numbers unchanged.

Two configurations share the same initial model, split, seed and training
settings while using independent run names/checkpoints/calibration paths:

- configs/model/yolo_center_loss_baseline.yaml: objective disabled.
- configs/model/yolo_center_loss_trial.yaml: objective enabled.

Neither file changes the deployed default configuration. Formal training and
quality evaluation need a separately approved protocol. Example commands,
not executed by implementation work:

    .\.venv\Scripts\python.exe -X utf8 scripts\train_detection.py --config configs/model/yolo_center_loss_baseline.yaml --fresh
    .\.venv\Scripts\python.exe -X utf8 scripts\train_detection.py --config configs/model/yolo_center_loss_trial.yaml --fresh

Use the matching config with --stop to request an epoch-boundary pause.
Use --continue to preserve its saved target; never resume a paused run
automatically. Inference remains disabled in these experiment configurations
until a separately evaluated and calibrated candidate is selected.

## Formula, units and supervision

D is the Euclidean distance of predicted and assigned target box centers.
Each FPN grid-coordinate difference is multiplied by its own stride, so D
uses input-image pixels. The bounded term is -expm1(-D/C); FP32 geometry and
vector_norm retain stable small distances and finite zero-distance gradients
under mixed precision.

The paper calls C average object size without defining its exact aggregation.
This project's explicit adaptation defines C as the arithmetic mean sqrt(area)
of all labeled training boxes after the existing training resize/letterbox.
The implementation uses the actual train dataset's normalized xywh labels and
image shapes, including the installed loader's ceil resize dimensions. It
does not inspect validation/calibration/test batches or average predictions.
Invalid shapes/boxes or an empty set of training boxes are rejected.

For assigned foreground anchors, the auxiliary term uses the existing summed
target-score weights and score denominator. It is added to the box component
before the vendor's box gain and batch scaling. CIoU and DFL remain in place;
the three public loss components and AP-based model selection remain unchanged.
Existing classification source weights, known-class coverage, ROI masks and
foreground exclusivity remain in force. Unlabeled regions gain no negative
supervision from this feature.

## Restore identity and evidence

The resolved recipe includes its algorithm version, coefficient, normalization,
C in pixels, labeled training-box count, actual imgsz and dataset signature.
It is saved in run_metadata.json and last_training_state.pt. Both configuration
and resolved normalization must match for same-directory restore. A missing or
changed recipe cannot silently resume an enabled objective. Legacy states with
no center_loss field remain valid with the objective disabled.

A new run may initialize from old weights without inheriting their validation
best or optimizer history. A newly trained candidate needs its own calibration;
it does not reuse the deployed calibration file.

Implementation and software regression do not establish improved AP, recall,
false-positive rate or real-time performance. A quality experiment must report
nine classes, the original eight classes, road_obstacle and size/source groups
with unchanged initialization, split, budget and postprocessing.
