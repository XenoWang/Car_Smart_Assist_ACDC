"""Create training-only low-light previews with the current detector's HSV baseline."""
import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from car_smart_assist.config.detection import load_yolo_config
from car_smart_assist.data.transforms.lowlight import lowlight_rgb

LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)
SEED = 42


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read_frame(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Source image identity changed: " + str(path))
    with Image.open(BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"))


def resized(rgb, maximum, interpolation):
    h, w = rgb.shape[:2]
    width, height = min(maximum, math.ceil(w * maximum/max(h,w))), min(maximum, math.ceil(h * maximum/max(h,w)))
    return cv2.resize(rgb, (width,height), interpolation=interpolation)


def patch_pixels(luma, box):
    _, x, y, w, h = box
    height, width = luma.shape
    x1, x2 = max(0, math.floor((x-w/2)*width)), min(width, math.ceil((x+w/2)*width))
    y1, y2 = max(0, math.floor((y-h/2)*height)), min(height, math.ceil((y+h/2)*height))
    patch = luma[y1:y2, x1:x2]
    if not patch.size:
        raise ValueError("Empty annotation patch")
    return patch


def statistics(rgb, boxes):
    lum = (rgb.astype(np.float32) * LUMA).sum(-1)
    targets = []
    height, width = lum.shape
    for i, box in enumerate(boxes):
        patch = patch_pixels(lum, box)
        targets.append({"index": i, "class_id": int(box[0]), "tiny": box[3]*width*box[4]*height < 256,
                        "mean": float(patch.mean()), "std": float(patch.std()),
                        "dark_fraction_lt16": float((patch < 16).mean())})
    return {"mean": float(lum.mean()), "dark_fraction_lt16": float((lum < 16).mean()), "targets": targets}


def choose_samples(records):
    by_group = defaultdict(list)
    for row in records:
        if any(t["tiny"] for t in row["targets"]):
            by_group[row["condition"]].append(row)
    chosen = {}
    for group, rows in sorted(by_group.items()):
        ordered = sorted(rows, key=lambda r:(r["mean"],r["image"]))
        for q in (.1,.5,.9):
            row = ordered[round(q*(len(ordered)-1))]
            chosen.setdefault(row["image"], []).append(f"{group} frame mean q{q}")
    tiny = sorted((t["mean"], r["image"], t["index"]) for r in records if r["condition"]=="night"
                  for t in r["targets"] if t["tiny"])
    for q in (.1,.9):
        _, name, index = tiny[round(q*(len(tiny)-1))]
        chosen.setdefault(name, []).append(f"night tiny mean q{q}, target {index}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(output/"settings")
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics.data.augment import RandomHSV

    cfg = load_yolo_config(ROOT/"configs/model/yolo_detection.yaml")
    base = (ROOT/cfg["data"]).parent
    manifest = json.loads((base/"manifest.json").read_bytes())
    train = {r["image"]:r for r in manifest["images"] if r["split"]=="train"}
    records = []
    for name, row in train.items():
        rgb = resized(read_frame(base/name,row["sha256"]),960,cv2.INTER_LINEAR)
        record = statistics(rgb,row["boxes"])
        records.append({"image":name,"condition":row["condition"],"dataset":row["dataset"],**record})
    save(output/"train_statistics.json",{"dataset_signature":manifest["signature"],"split":"train","records":records})
    selected = choose_samples(records)
    save(output/"selection.json",selected)
    hsv = RandomHSV(*(cfg["train"][k] for k in ("hsv_h","hsv_s","hsv_v")))
    results = []
    for ordinal, (name,reasons) in enumerate(selected.items(),1):
        row = train[name]
        rgb = resized(read_frame(base/name,row["sha256"]),960,cv2.INTER_LINEAR)
        sample_seed = SEED ^ int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
        bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        state = np.random.get_state()
        try:
            np.random.seed(sample_seed)
            hsv({"img":bgr})
        finally:
            np.random.set_state(state)
        baseline = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        variants = {"original":rgb, "current_hsv":baseline}
        for exposure in (.5,.25):
            variants[f"exposure_{exposure}"] = lowlight_rgb(baseline, exposure)
            variants[f"noise_{exposure}"] = lowlight_rgb(baseline, exposure,shot_scale=.001,read_std=.001,
                                                       rng=np.random.default_rng(sample_seed))
        items = {}
        for label,pixels in variants.items():
            filename = f"{ordinal:02d}_{label}.png"
            Image.fromarray(pixels).save(output/filename)
            items[label] = {"image":filename, **statistics(pixels,row["boxes"])}
        # Six panels in two rows. Rectangles overlay every original tiny target.
        width = 400
        height = round(rgb.shape[0]*width/rgb.shape[1])
        sheet = Image.new("RGB",(width*3,(height+28)*2),"white")
        draw = ImageDraw.Draw(sheet)
        for i,(label,pixels) in enumerate(variants.items()):
            panel = Image.fromarray(pixels).resize((width,height),Image.Resampling.LANCZOS)
            painter = ImageDraw.Draw(panel)
            for box in row["boxes"]:
                if box[3]*rgb.shape[1]*box[4]*rgb.shape[0] >= 256:
                    continue
                _,x,y,w,h = box
                painter.rectangle([(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height],outline="cyan",width=1)
            left,top = (i%3)*width,(i//3)*(height+28)
            sheet.paste(panel,(left,top+28))
            draw.text((left+6,top+7),label,fill="black")
        sheet_name=f"{ordinal:02d}_comparison.png"
        sheet.save(output/sheet_name)
        results.append({"image":name,"sha256":row["sha256"],"condition":row["condition"],"reasons":reasons,
                        "seed":sample_seed,"comparison":sheet_name,"variants":items})
    save(output/"preview.json",{"status":"COMPLETED","split":"train","train_images":len(train),
         "train_boxes":sum(len(r["boxes"]) for r in train.values()),"dataset_signature":manifest["signature"],
         "hsv":{k:cfg["train"][k] for k in ("hsv_h","hsv_s","hsv_v")},"seed":SEED,
         "shot_scale":.001,"read_std":.001,"records":results,
         "scope":"RGB long-side960 diagnostic views, native RandomHSV only; no model or full training augmentation"})
    print("Preview complete:",output,"selected frames:",len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
