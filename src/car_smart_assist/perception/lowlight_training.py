"""Training-only low-light schedule with independent image seeds and budget receipts."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from car_smart_assist.data.transforms.lowlight import lowlight_rgb


def recipe_identity(cfg):
    recipe = cfg.get("training_lowlight")
    if recipe is None:
        return None
    if not isinstance(recipe, dict) or set(recipe) != {"mode", "plan", "sha256"}:
        raise ValueError("training_lowlight requires mode, plan and sha256")
    digest = recipe["sha256"]
    if (recipe["mode"] not in ("control","lowlight") or not isinstance(recipe["plan"], str) or not recipe["plan"].strip()
            or not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("Invalid low-light training recipe")
    return {"mode":recipe["mode"],"plan_sha256":digest}


def load_plan(root, cfg, manifest):
    identity = recipe_identity(cfg)
    if identity is None:
        return None
    raw = (Path(root)/cfg["training_lowlight"]["plan"]).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=identity["plan_sha256"]:
        raise ValueError("Low-light schedule identity changed")
    plan=json.loads(raw)
    train=cfg["train"]
    if (plan["version"]!=1 or not manifest.get("joint_supervision") or
            plan["dataset_signature"]!=manifest["signature"] or plan["epochs"]!=train["epochs"]
            or plan["imgsz"]!=train["imgsz"] or plan["seed"]!=train["seed"] or train.get("bgr",0)!=0):
        raise ValueError("Low-light schedule does not match the training contract")
    if (plan["exposure"],plan["shot_scale"],plan["read_std"],plan["minimum_mean"]) != (.5,.001,.001,80):
        raise ValueError("Unsupported low-light screening parameters")
    rows={r["image"]:r for r in manifest["images"] if r["split"]=="train"}
    eligible=plan["eligible"]
    if not eligible or not set(eligible).issubset(rows) or any(
            rows[k]["condition"]=="night" or not np.isfinite(v) or v<plan["minimum_mean"] for k,v in eligible.items()):
        raise ValueError("Low-light eligibility requires brighter non-night training frames")
    if len(plan["schedule"])!=train["epochs"]:
        raise ValueError("Incomplete low-light epoch schedule")
    for selected in plan["schedule"]:
        if len(selected)!=len(set(selected)) or len(selected)!=len(eligible)//4 or not set(selected).issubset(eligible):
            raise ValueError("Invalid low-light per-epoch selection")
    base=(Path(root)/cfg["data"]).parent
    return {"parameters":plan, "mode":identity["mode"],
            "image_ids":{str((base/k).resolve()):k for k in rows},
            "schedule":[set(s) for s in plan["schedule"]]}


def transform_batch(batch, plan, epoch):
    """Apply only inside native unpadded image bounds, before device normalization."""
    pixels=batch["img"]
    if pixels.dtype!=torch.uint8 or pixels.device.type!="cpu" or pixels.ndim!=4 or pixels.shape[1]!=3:
        raise ValueError("Low-light expects native CPU uint8 RGB batches")
    p=plan["parameters"]
    if pixels.shape[2:]!=(p["imgsz"],p["imgsz"]):
        raise ValueError("Unexpected training geometry")
    selected=plan["schedule"][epoch]
    result=pixels
    selections=0
    applied=0
    for i,path in enumerate(batch["im_file"]):
        key=plan["image_ids"][str(Path(path).resolve())]
        if key not in selected:
            continue
        selections+=1
        if plan["mode"]=="control":
            continue
        if applied==0:
            result=pixels.clone()
        h,w=batch["ori_shape"][i]
        size=p["imgsz"]
        ratio=size/max(h,w)
        rh,rw=min(size,math.ceil(h*ratio)),min(size,math.ceil(w*ratio))
        top,left=round((size-rh)/2-.1),round((size-rw)/2-.1)
        seed=int.from_bytes(hashlib.sha256(f"{p['seed']}|{epoch}|{key}".encode()).digest()[:8],"little")
        source=pixels[i,:,top:top+rh,left:left+rw].permute(1,2,0).numpy()
        generated=lowlight_rgb(source,p["exposure"],shot_scale=p["shot_scale"],read_std=p["read_std"],
                              rng=np.random.default_rng(seed))
        result[i,:,top:top+rh,left:left+rw].copy_(torch.from_numpy(generated).permute(2,0,1))
        applied+=1
    return {**batch,"img":result},selections,applied


def training_class(base, plan):
    if plan is None:
        return base

    class LowlightTrainer(base):
        def preprocess_batch(self,batch):
            if not hasattr(self,"_lowlight_order"):
                self._lowlight_order=hashlib.sha256()
                self._lowlight_counts={"images":0,"batches":0,"optimizer_updates":0,"selected":0,"applied":0}
                self.optimizer.register_step_post_hook(self._count_update)
            paths=batch["im_file"]
            self._lowlight_order.update(json.dumps(list(paths),ensure_ascii=False).encode()+b"\n")
            self._lowlight_counts["images"]+=len(paths)
            self._lowlight_counts["batches"]+=1
            transformed,selected,applied=transform_batch(batch,plan,self.epoch)
            self._lowlight_counts["selected"]+=selected
            self._lowlight_counts["applied"]+=applied
            return super().preprocess_batch(transformed)

        def _count_update(self,optimizer,args,kwargs):
            self._lowlight_counts["optimizer_updates"]+=1

        def lowlight_budget(self):
            return {**self._lowlight_counts,"order_sha256":self._lowlight_order.hexdigest(),
                    "scope":"training batches processed in this invocation"}

    return LowlightTrainer
