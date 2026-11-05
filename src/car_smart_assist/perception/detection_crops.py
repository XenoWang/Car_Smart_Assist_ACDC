"""Box-preserving training windows for the paired whole-frame/crop experiment."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def enclosing_window(size, boxes):
    """Choose the smallest preset context window containing every annotated box."""
    width, height = size
    if len(boxes) == 0:
        return None
    points = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    centers = points[:, :2] * size
    radii = points[:, 2:] * size / 2
    limits = np.asarray(size)
    low = np.maximum(0, np.floor((centers - radii).min(0)).astype(int) - 1)
    high = np.minimum(limits, np.ceil((centers + radii).max(0)).astype(int) + 1)
    for numerator, denominator in ((1, 2), (2, 3)):
        span = (limits * numerator + denominator - 1) // denominator
        if np.all(high - low <= span):
            origin = np.clip((low + high - span) // 2, 0, limits - span)
            return [*origin.tolist(), *(origin + span).tolist()]
    return None


def project_boxes(boxes, size, window):
    """Translate normalized xywh, preserving every box and its class/order."""
    origin = np.asarray(window[:2], dtype=np.float64)
    extent = np.asarray(window[2:], dtype=np.float64) - origin
    if np.any(origin < 0) or np.any(extent <= 0) or np.any(origin + extent > size):
        raise ValueError("Invalid crop coordinates")
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    centers = boxes[:, :2] * size - origin
    widths = boxes[:, 2:] * size
    # Serialized YOLO boxes may round a native image-edge coordinate by a fraction of a pixel.
    if (centers - widths / 2 < -0.001).any() or (centers + widths / 2 > extent + .001).any():
        raise ValueError("Training window would truncate an annotated box")
    boxes[:, :2] = centers / extent
    boxes[:, 2:] = widths / extent
    return boxes.astype(np.float32)


def recipe_identity(cfg):
    recipe = cfg.get("training_view")
    if recipe is None:
        return None
    if not isinstance(recipe, dict) or set(recipe) != {"mode", "plan", "sha256"}:
        raise ValueError("training_view requires mode, plan and sha256")
    sha = recipe["sha256"]
    if (recipe["mode"] not in ("full", "crop") or not isinstance(recipe["plan"], str) or not recipe["plan"].strip()
            or not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
        raise ValueError("Invalid training view recipe")
    return {"mode": recipe["mode"], "plan_sha256": sha}


def load_windows(root, cfg, manifest):
    identity = recipe_identity(cfg)
    if identity is None:
        return None
    raw = (Path(root) / cfg["training_view"]["plan"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != identity["plan_sha256"]:
        raise ValueError("Training view plan hash differs from its frozen recipe")
    plan = json.loads(raw)
    if (plan["version"] != 1 or not manifest.get("joint_supervision") or
            plan["dataset_signature"] != manifest["signature"] or plan["imgsz"] != cfg["train"]["imgsz"]):
        raise ValueError("Training view dataset/geometry identity mismatch")
    train = {r["image"]: r for r in manifest["images"] if r["split"] == "train"}
    if not plan["windows"] or not set(plan["windows"]).issubset(train):
        raise ValueError("Crop windows require training-only image IDs")
    for key, entry in plan["windows"].items():
        size, window = entry["size"], entry["xyxy"]
        if (len(size) != 2 or len(window) != 4 or any(type(x) is not int for x in [*size, *window])
                or min(size) < 1 or any(a < 0 or b <= a or b > lim for a, b, lim in zip(window[:2], window[2:], size))):
            raise ValueError("Invalid training view geometry: " + key)
        boxes = [b[1:] for b in train[key]["boxes"]]
        if enclosing_window(size, boxes) != window:
            raise ValueError("Window no longer matches complete source annotations: " + key)
    return plan["windows"] if identity["mode"] == "crop" else {}


def cropped_sample(dataset, index, view):
    from copy import deepcopy
    import cv2
    from ultralytics.utils.patches import imread

    label = deepcopy(dataset.labels[index])
    label.pop("shape", None)
    frame = imread(dataset.im_files[index], flags=dataset.cv2_flag)
    if frame is None or list(frame.shape[1::-1]) != view["size"]:
        raise ValueError("Training image dimensions changed: " + dataset.im_files[index])
    if not label["normalized"] or label["bbox_format"] != "xywh" or len(label["segments"]):
        raise ValueError("Training windows require normalized detection boxes")
    window = view["xyxy"]
    x1, y1, x2, y2 = window
    width, height = x2-x1, y2-y1
    ratio = dataset.imgsz / max(width, height)
    shape = (min(dataset.imgsz, math.ceil(width * ratio)), min(dataset.imgsz, math.ceil(height * ratio)))
    pixels = np.ascontiguousarray(frame[y1:y2, x1:x2])
    if shape != (width, height):
        pixels = cv2.resize(pixels, shape, interpolation=cv2.INTER_LINEAR)
    label.update(img=pixels, ori_shape=(height, width), resized_shape=pixels.shape[:2],
                 ratio_pad=(shape[1]/height, shape[0]/width),
                 bboxes=project_boxes(label["bboxes"], view["size"], window))
    return dataset.update_labels_info(label)
