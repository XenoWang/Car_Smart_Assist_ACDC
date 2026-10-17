"""从已完成的 last 开启追加阶段，保留训练状态并重开 SWAD 区间。"""

from __future__ import annotations

import hashlib
import math
import shutil

import torch


def load_continuation(cfg, root, recipe, destination):
    options = cfg.train["continuation"]
    source = root / options["checkpoint"]
    best = root / options["single_best"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != options["checkpoint_sha256"]:
        raise ValueError("追加训练的源 last 已改变")
    if hashlib.sha256(best.read_bytes()).hexdigest() != options["single_best_sha256"]:
        raise ValueError("追加训练的源单点 best 已改变")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload["epoch"] != options["source_epoch"] or not payload["training_complete"]:
        raise ValueError("追加训练必须读取指定的已完成轮次")
    if payload["best_sha256"] != options["single_best_sha256"]:
        raise ValueError("源 last 与单点 best 不匹配")
    previous = payload["recipe"]
    for name in (
        "model",
        "backbone_training",
        "loss",
        "split_signature",
        "dataset_signatures",
        "seed",
        "pretrained_sha256",
    ):
        if previous[name] != recipe[name]:
            raise ValueError(f"追加训练的 {name} 与源运行不一致")
    ignored = {"epochs", "schedule_epochs", "continuation", "swad"}
    if {k: v for k, v in previous["train"].items() if k not in ignored} != {
        k: v for k, v in cfg.train.items() if k not in ignored
    }:
        raise ValueError("追加阶段不得同时改变普通训练配方")
    old_swad = {k: v for k, v in previous["train"]["swad"].items() if k != "start_epoch"}
    new_swad = {k: v for k, v in cfg.train["swad"].items() if k != "start_epoch"}
    if old_swad != new_swad:
        raise ValueError("追加阶段只重开 SWAD 区间，不改变谷底参数")
    for name, expected in previous["baseline_hashes"].items():
        if recipe["baseline_hashes"].get(name) != expected:
            raise ValueError(f"追加训练的对照权重已改变：{name}")
    for key in ("selection_weights", "ap_selection_fraction", "threshold_steps"):
        if previous["selection"][key] != cfg.evaluation[key]:
            raise ValueError("追加阶段的选优及校准方法须保持一致")
    for key, base in (("head_lr", "learning_rate"), ("backbone_lr", "backbone_learning_rate")):
        if not math.isclose(
            payload["history"][-1][key], cfg.train[base] * options["start_factor"], rel_tol=1e-10
        ):
            raise ValueError("追加阶段的起始学习率与源末轮不连续")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # copyfile 复制内容，不把只读对照文件的 Windows 只读属性带进可训练目录。
    shutil.copyfile(best, destination)
    return payload
