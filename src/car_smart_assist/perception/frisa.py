"""Equation-led FRISA reconstruction for the native YOLO11n bottom-up P4 block."""
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from car_smart_assist.config.detection import FRISA_VERSION
from car_smart_assist.perception.detection_yolo import configure_yolo_environment

if not os.environ.get("YOLO_CONFIG_DIR"):
    configure_yolo_environment(Path(__file__).resolve().parents[3])
from ultralytics.nn.modules.block import C3k2

P4_INDEX = 19
VERSION = FRISA_VERSION


def norm(channels):
    return nn.BatchNorm2d(channels, eps=.001, momentum=.03)


class FRISA(nn.Module):
    """Implement equations 16-34 with the local choices documented in docs/yolo_frisa.md."""
    def __init__(self, channels):
        super().__init__()
        self.decompose = nn.Conv2d(channels, channels, 1)
        self.prefilter = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), norm(channels), nn.Sigmoid())
        self.details = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                                             groups=channels, bias=False) for d in (1, 2, 3))
        self.aggregate = nn.Sequential(norm(channels), nn.GELU(), nn.Conv2d(channels, channels, 1))
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 1, bias=False), norm(1), nn.Sigmoid())
        self.channel = nn.Conv2d(channels, channels, 1)
        self.residual_to_semantic = nn.Conv2d(channels, 1, 1)
        self.semantic_to_residual = nn.Conv2d(channels, 1, 1)
        self.branch_weights = nn.Parameter(torch.zeros(2))
        self.output_gate = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), norm(channels), nn.Sigmoid())

    @staticmethod
    def pool(x):
        return x.mean(dim=(2, 3), keepdim=True)

    def forward(self, x):
        semantic = x * self.decompose(self.pool(x)).sigmoid()
        residual = x - semantic
        residual = residual * self.prefilter(residual)
        detail = self.aggregate(sum(layer(residual) for layer in self.details))
        spatial = self.spatial(torch.cat((detail.mean(1, keepdim=True), detail.amax(1, keepdim=True)), dim=1))
        semantic = semantic * self.channel(self.pool(semantic)).sigmoid()
        semantic = semantic * self.residual_to_semantic(self.pool(residual)).sigmoid()
        residual = residual * spatial * self.semantic_to_residual(self.pool(semantic)).sigmoid()
        weights = self.branch_weights.sigmoid()
        fused = weights[0] * semantic + weights[1] * residual
        return x + x * self.output_gate(fused)


class FrisaP4(C3k2):
    """Keep the native C3k2 parameter names and graph metadata."""
    def __init__(self, original):
        nn.Module.__init__(self)
        for name, layer in original.named_children():
            self.add_module(name, layer)
        self.c = original.c
        for name in ("i", "f", "type", "np"):
            setattr(self, name, getattr(original, name))
        reference = original.cv2.conv.weight
        # Keep the parent's RNG stream identical across the paired trainers.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.frisa = FRISA(reference.shape[0]).to(device=reference.device, dtype=reference.dtype)
        self.type = "car_smart_assist.perception.frisa.FrisaP4"
        self.np = sum(p.numel() for p in self.parameters())
        self.train(original.training)

    def forward(self, x):
        return self.frisa(super().forward(x))

    def forward_split(self, x):
        return self.frisa(super().forward_split(x))


def validate_source(model, recipe):
    existing = getattr(model, "frisa_version", None)
    modules = model.modules() if isinstance(model, nn.Module) else ()
    count = sum(isinstance(layer, FrisaP4) for layer in modules)
    if (existing is None and count) or (existing is not None and count != 1):
        raise ValueError("FRISA checkpoint marker and module structure disagree")
    if existing is not None and (existing != VERSION or recipe is None or recipe["mode"] != "frisa"
                                 or recipe.get("version") != VERSION):
        raise ValueError("FRISA checkpoint requires its matching FRISA recipe")


def install_p4(model, source):
    layers = model.model
    if (len(layers) != 24 or type(layers[P4_INDEX]) is not C3k2 or layers[P4_INDEX].i != P4_INDEX
            or layers[-1].f != [16, 19, 22] or layers[-1].nc != 9 or layers[P4_INDEX].f != -1
            or layers[18].f != [-1, 13] or layers[20].f != -1 or layers[21].f != [-1, 10]
            or layers[P4_INDEX].cv2.conv.out_channels != 128):
        raise ValueError("FRISA requires the reviewed native YOLO11n P4/16 topology")
    if source is None or model.names != source.names:
        raise ValueError("FRISA requires matching pretrained detection class names")
    original = source.float().state_dict()
    base_state = model.state_dict()
    if set(base_state) - set(original) or any(not torch.equal(t, original[k].to(t.device)) for k, t in base_state.items()):
        raise ValueError("Shared pretrained tensors were not transferred exactly")
    layers[P4_INDEX] = FrisaP4(layers[P4_INDEX])
    if getattr(source, "frisa_version", None) is not None:
        if source.frisa_version != VERSION or type(source.model[P4_INDEX]) is not FrisaP4:
            raise ValueError("Unsupported saved FRISA architecture")
        layers[P4_INDEX].frisa.load_state_dict(source.model[P4_INDEX].frisa.state_dict(), strict=True)
    model.frisa_version = VERSION
    return {"version": VERSION, "p4_index": P4_INDEX, "shared_state_entries": len(base_state),
            "new_parameters": sum(p.numel() for p in layers[P4_INDEX].frisa.parameters()),
            "source_had_frisa": getattr(source, "frisa_version", None) is not None}


def training_class(base, recipe):
    if recipe is None:
        return base

    class FrisaTrainer(base):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            validate_source(weights, recipe)
            model.names = dict(self.data["names"])
            if recipe["mode"] == "frisa":
                self._frisa_transfer = install_p4(model, weights)
            return model

        def preprocess_batch(self, batch):
            if not hasattr(self, "_frisa_order"):
                self._frisa_order = hashlib.sha256()
                self._frisa_counts = {"images": 0, "batches": 0, "optimizer_updates": 0}
                self.optimizer.register_step_post_hook(self._count_update)
            paths = batch["im_file"]
            self._frisa_order.update(json.dumps(list(paths), ensure_ascii=False).encode() + b"\n")
            self._frisa_counts["images"] += len(paths)
            self._frisa_counts["batches"] += 1
            return super().preprocess_batch(batch)

        def _count_update(self, optimizer, args, kwargs):
            self._frisa_counts["optimizer_updates"] += 1

        def frisa_receipt(self):
            return {**self._frisa_counts, "order_sha256": self._frisa_order.hexdigest(),
                    "transfer": getattr(self, "_frisa_transfer", None),
                    "scope": "training batches in this invocation"}

    return FrisaTrainer
