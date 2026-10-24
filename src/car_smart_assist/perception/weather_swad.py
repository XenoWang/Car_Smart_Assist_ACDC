"""逐次更新取平均，用验证损失确定保留哪一段训练轨迹。"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters


def source_balanced_bce(dataset, probabilities, cfg):
    """BCE 能反映过度自信的错误；来源等权，雨雾沿用现有选择权重。"""
    from car_smart_assist.perception.weather_attention_training import source_balanced_weights

    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    scores = []
    for index, name in enumerate(cfg.attributes):
        known = dataset.mask[:, index] > 0
        weights = source_balanced_weights(dataset, known)[known]
        truth, probability = dataset.targets[known, index], values[known, index]
        if not len(truth):
            raise ValueError(f"SWAD 验证集缺少 {name} 的已确认标签")
        terms = -truth * np.log(probability) - (1 - truth) * np.log1p(-probability)
        scores.append(float(np.average(terms, weights=weights)))
    result = float(
        np.average(scores, weights=[cfg.evaluation["selection_weights"][a] for a in cfg.attributes])
    )
    if not np.isfinite(result):
        raise ValueError("SWAD 验证 BCE 非有限")
    return result


class DenseAverage:
    """GPU 上批量更新参数平均值，每个验证区间只向 CPU 复制一次。"""

    def __init__(self, model):
        self.parameters = tuple(model.parameters())
        self.average = [p.detach().clone() for p in self.parameters]
        self.count = 0

    @torch.no_grad()
    def update(self):
        if self.count == 0:
            torch._foreach_copy_(self.average, self.parameters)
        else:
            torch._foreach_lerp_(self.average, self.parameters, 1 / (self.count + 1))
        self.count += 1

    def take(self, epoch, end_step):
        if self.count == 0:
            raise ValueError("没有实际更新，不能生成平均权重")
        segment = {
            "epoch": epoch,
            "start_step": end_step - self.count + 1,
            "end_step": end_step,
            "count": self.count,
            "vector": torch.cat([p.reshape(-1) for p in self.average]).cpu(),
        }
        self.count = 0
        return segment


class LossValley:
    """损失谷底检测；尾部等待足够长后再纳入，避免混进过拟合权重。"""

    def __init__(self, options):
        self.options = dict(options)
        self.pending = deque()
        self.threshold = None
        self.dead = False
        self.start_epoch = None
        self.average = None
        self.count = 0
        self.first_step = self.last_step = self.end_epoch = None

    def _include(self, segment):
        if self.average is None:
            self.average = segment["vector"].clone()
            self.first_step = segment["start_step"]
        else:
            self.average.lerp_(
                segment["vector"], segment["count"] / (self.count + segment["count"])
            )
        self.count += segment["count"]
        self.last_step, self.end_epoch = segment["end_step"], segment["epoch"]

    def add(self, segment, loss):
        if self.dead:
            return
        if not np.isfinite(loss) or loss < 0:
            raise ValueError("SWAD 谷底损失必须有限且非负")
        segment = {**segment, "loss": float(loss)}
        self.pending.append(segment)
        converge, tolerance = self.options["n_converge"], self.options["n_tolerance"]
        if self.threshold is None:
            recent = list(self.pending)[-converge:]
            if len(recent) == converge and np.argmin([s["loss"] for s in recent]) == 0:
                self.threshold = float(np.mean([s["loss"] for s in recent])) * (
                    1 + self.options["tolerance_ratio"]
                )
                self.start_epoch = recent[0]["epoch"]
                self.pending = deque(recent)
            else:
                while len(self.pending) > tolerance:
                    self.pending.popleft()
                return
        if len(self.pending) >= tolerance:
            if min(s["loss"] for s in self.pending) > self.threshold:
                self.dead = True
                self.pending.clear()
            else:
                self._include(self.pending.popleft())

    def finish(self):
        if self.threshold is None:
            return None
        while self.pending and min(s["loss"] for s in self.pending) <= self.threshold:
            self._include(self.pending.popleft())
        self.pending.clear()
        return self.average

    def summary(self):
        return {
            "converged": self.threshold is not None,
            "dead_valley": self.dead,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "start_step": self.first_step,
            "end_step": self.last_step,
            "averaged_updates": self.count,
            "pending_segments": len(self.pending),
            "loss_threshold": self.threshold,
            "monitor": "source_balanced_bce",
            "bn_policy": "frozen_imagenet",
        }

    def state_dict(self):
        return {**vars(self), "pending": list(self.pending)}

    def load_state_dict(self, state):
        if state["options"] != self.options:
            raise ValueError("SWAD 谷底配置已改变")
        self.__dict__.update(state)
        self.pending = deque(state["pending"])


class SWADTracker:
    def __init__(self, model, options):
        self.options = dict(options)
        self.dense = DenseAverage(model)
        self.valley = LossValley(options)
        self.step = 0

    def update(self, epoch):
        self.step += 1
        if epoch >= self.options["start_epoch"] and not self.valley.dead:
            self.dense.update()

    def end_epoch(self, epoch, validation_loss):
        if self.dense.count:
            self.valley.add(self.dense.take(epoch, self.step), validation_loss)

    def state_dict(self):
        if self.dense.count:
            raise ValueError("SWAD 与训练状态应在同一个完整轮次后保存")
        return {"step": self.step, "valley": self.valley.state_dict()}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.valley.load_state_dict(state["valley"])

    def final_model(self, template):
        # 在副本上收尾，重复导出或恢复训练不会再次累加同一段。
        valley = copy.deepcopy(self.valley)
        vector = valley.finish()
        metadata = valley.summary()
        if vector is None:
            metadata["fallback_reason"] = "预算内未找到稳定谷底，使用单点 best"
            return None, metadata
        model = copy.deepcopy(template).eval()
        vector_to_parameters(vector.to(next(model.parameters()).device), model.parameters())
        if model.cfg.train["channels_last"]:
            model.to(memory_format=torch.channels_last)
        return model, metadata


def finish_swad_checkpoints(tracker, model, payload, cfg, root, datasets, device, *, stop_controller=None):
    """两种权重独立校准，按验证评分导出 best；测试集不参与选择。"""
    from car_smart_assist.perception.weather_attention import optimize_attention_inference
    from car_smart_assist.perception.weather_attention_training import (
        attention_probabilities,
        balanced_thresholds,
        detailed_metrics,
        save_attention_best,
    )
    from car_smart_assist.perception.weather_training import _save_checkpoint

    from car_smart_assist.utils.stop import StopController

    stop_controller = stop_controller or StopController(default_directory=root / cfg.checkpoint_root)
    stop_controller.require_running()
    destination = root / cfg.checkpoint_root
    models, payloads = {"single_best": model}, {"single_best": payload}
    average, metadata = tracker.final_model(model)
    if average is not None:
        stop_controller.require_running()
        values = attention_probabilities(average, datasets["validation"], cfg, device)
        _, selection = balanced_thresholds(datasets["validation"], values, cfg)
        models["swad"] = average
        payloads["swad"] = {
            "format_version": 1,
            "model_config": payload["model_config"],
            "model_state": {n: p.detach().cpu() for n, p in average.state_dict().items()},
            "epoch": metadata["end_epoch"],
            "selection": selection,
            "validation": detailed_metrics(
                datasets["validation"], values, cfg, dict.fromkeys(cfg.attributes, 0.5)
            ),
            "split_signature": payload["split_signature"],
            "initial_model_sha256": payload["initial_model_sha256"],
            "loss_name": "asl",
            "averaging": metadata,
        }
    candidates, inference_models = {}, {}
    for name, candidate in models.items():
        stop_controller.require_running()
        candidate_payload = payloads[name]
        inference = optimize_attention_inference(candidate)
        values = attention_probabilities(inference, datasets["calibration"], cfg, device)
        stop_controller.require_running()
        thresholds, _ = balanced_thresholds(datasets["calibration"], values, cfg)
        candidate_payload.update(
            checkpoint_kind=name,
            calibration_signature=datasets["calibration"].signature,
            decision_thresholds=thresholds,
            calibration=detailed_metrics(datasets["calibration"], values, cfg, thresholds),
        )
        path = destination / f"{name}.pt"
        inference_models[name] = inference
        candidates[name] = {
            "checkpoint": str(path),
            "best_epoch": candidate_payload["epoch"],
            "selection": candidate_payload["selection"],
            "validation": candidate_payload["validation"],
            "thresholds": thresholds,
            "calibration": candidate_payload["calibration"],
        }
    chosen = "single_best"
    if (
        "swad" in candidates
        and candidates["swad"]["selection"]["score"]
        > candidates[chosen]["selection"]["score"] + 1e-12
    ):
        chosen = "swad"
    selected = {**payloads[chosen], "swad": metadata, "selected_kind": chosen}
    # Calibrate all candidates before publishing any of them. A stopped calibration
    # leaves the single-best/last hash pair unchanged and fabricates no export.
    stop_controller.require_running()
    for name, candidate_payload in payloads.items():
        path = destination / f"{name}.pt"
        if name == "single_best":
            save_attention_best(candidate_payload, path)
        else:
            _save_checkpoint(candidate_payload, path)
    _save_checkpoint(selected, destination / "best.pt")
    return (
        inference_models[chosen],
        selected,
        {
            "selected_kind": chosen,
            "candidates": candidates,
            "swad": metadata,
        },
    )
