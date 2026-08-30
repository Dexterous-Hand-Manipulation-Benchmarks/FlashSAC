import time
from typing import Any, Optional, Dict
import numpy as np
import torch
from omegaconf import OmegaConf

# Universal Metric Mapping
METRIC_MAPPING: Dict[str, str] = {
    # 3D Mesh Tracking & Evaluation
    "auc_add": "eval/auc_add",
    "AUC_ADD": "eval/auc_add",
    "AUC-ADD": "eval/auc_add",
    "add_auc": "eval/auc_add",
    "ADD_AUC": "eval/auc_add",
    "eval/auc_add": "eval/auc_add",
    "add": "eval/add",
    "ADD": "eval/add",
    "add_mean": "eval/add",
    "eval/add": "eval/add",
    "success_2cm": "eval/success_2cm",
    "success_5cm": "eval/success_5cm",
    "eval/success_2cm": "eval/success_2cm",
    "eval/success_5cm": "eval/success_5cm",
    "avg_success_end": "eval/success_5cm",
    "avg_success_once": "eval/success_once",
    "eval/success_rate": "eval/success_5cm",
    "success_rate": "eval/success_5cm",

    # Returns & Lengths
    "avg_return": "eval/return",
    "eval/mean_reward": "eval/return",
    "rewards/iter": "train/return",
    "rewards/test": "eval/return",
    "eval/avg_return": "eval/return",
    "avg_length": "eval/ep_length",
    "eval/mean_length": "eval/ep_length",
    "length/iter": "train/ep_length",
    "length/test": "eval/ep_length",

    # Physical Errors & Forces
    "contact_force": "env/contact_force",
    "contact_forces": "env/contact_force",
    "Episode/pos_dist": "env/pos_dist",
    "pos_dist": "env/pos_dist",
    "Episode/rot_dist": "env/rot_dist",
    "rot_dist": "env/rot_dist",
    "Episode/arti_dist": "env/arti_dist",
    "arti_dist": "env/arti_dist",

    # Policy & Critic Dynamics
    "actor/entropy": "train/entropy",
    "entropy": "train/entropy",
    "actor/loss": "train/actor_loss",
    "actor_loss": "train/actor_loss",
    "critic/loss": "train/critic_loss",
    "critic_loss": "train/critic_loss",
    "actor/mean_action": "flash_sac/mean_action",
    "critic/max_entropy_bonus": "flash_sac/max_entropy_bonus",

    # System
    "fps/total": "perf/fps",
    "fps": "perf/fps",
}

IGNORED_KEYS = {"episode_rew"}


def _sanitize_val(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        if val.numel() == 0:
            return None
        if val.numel() == 1:
            item = val.item()
            return None if not np.isfinite(item) else item
        val = val.detach().cpu().numpy()
    if isinstance(val, (np.ndarray, list, tuple)):
        arr = np.asarray(val)
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            return None
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return None
        if not finite_mask.all():
            arr = arr[finite_mask]
        if arr.size == 1:
            return float(arr.item())
        return None  # Drop multi-element arrays to prevent W&B histogram serialization crashes
    if isinstance(val, (int, float, np.floating, np.integer)):
        return None if not np.isfinite(val) else (float(val) if isinstance(val, (float, np.floating)) else int(val))
    return val


class WandbTrainerLogger:
    def __init__(self, cfg: Any):
        import wandb

        self._wandb = wandb
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, throw_on_missing=True)
        self.run = wandb.init(
            project=cfg.project_name,
            entity=cfg.entity_name,
            group=cfg.group_name,
            config=dict_cfg,
        )
        self.start_time = time.time()
        self._setup_metrics()
        self.media_dict: dict[str, Any] = {}
        self.reset()

    def _setup_metrics(self) -> None:
        self._wandb.define_metric("global/env_step")
        self._wandb.define_metric("global/*", step_metric="global/env_step")
        self._wandb.define_metric("train/*", step_metric="global/env_step")
        self._wandb.define_metric("eval/*", step_metric="global/env_step")
        self._wandb.define_metric("env/*", step_metric="global/env_step")
        self._wandb.define_metric("perf/*", step_metric="global/env_step")
        self._wandb.define_metric("flash_sac/*", step_metric="global/env_step")

    def update_metric(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in IGNORED_KEYS:
                continue
            if isinstance(v, (float, int)):
                if np.isfinite(v):
                    self.average_meter_dict.update(k, v)
            elif isinstance(v, np.ndarray) and v.ndim == 5:
                self.media_dict[k] = self._wandb.Video(v, fps=30, format="gif")
            else:
                clean = _sanitize_val(v)
                if clean is not None:
                    if isinstance(clean, (float, int)):
                        self.average_meter_dict.update(k, clean)
                    else:
                        self.media_dict[k] = clean

    def log_metric(self, step: int) -> None:
        raw_data = {}
        raw_data.update(self.average_meter_dict.averages())
        raw_data.update(self.media_dict)

        sanitized_payload = {
            "global/env_step": int(step),
            "global/wall_time_s": time.time() - self.start_time,
        }

        for key, val in raw_data.items():
            clean_val = _sanitize_val(val)
            if clean_val is None:
                continue
            if key in METRIC_MAPPING:
                sanitized_payload[METRIC_MAPPING[key]] = clean_val
            elif key.startswith(("global/", "train/", "eval/", "env/", "perf/", "flash_sac/")):
                sanitized_payload[key] = clean_val
            elif isinstance(val, (self._wandb.Video, self._wandb.Image)):
                sanitized_payload[key] = val

        self._wandb.log(sanitized_payload)

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict.clear()


class TensorboardTrainerLogger:
    def __init__(self, cfg: Any):
        from datetime import datetime
        from torch.utils.tensorboard import SummaryWriter

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        log_dir = f"runs/{cfg.group_name}/{cfg.exp_name}/{cfg.env.env_name}_seed{cfg.seed}_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text("config", OmegaConf.to_yaml(cfg))
        self.media_dict: dict[str, Any] = {}
        self.reset()

    def update_metric(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in IGNORED_KEYS:
                continue
            if isinstance(v, (float, int)):
                if np.isfinite(v):
                    self.average_meter_dict.update(k, v)
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> None:
        for k, v in self.average_meter_dict.averages().items():
            mapped_key = METRIC_MAPPING.get(k, k)
            self.writer.add_scalar(mapped_key, v, global_step=step)
        self.writer.flush()

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict.clear()


class AverageMeter:
    def __init__(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        if not np.isfinite(val):
            return
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __format__(self, format: str) -> str:
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class AverageMeterDict:
    def __init__(self, meters: Optional[Dict[str, AverageMeter]] = None):
        self.meters = meters if meters else {}

    def __getitem__(self, key: str) -> AverageMeter:
        if key not in self.meters:
            self.meters[key] = AverageMeter()
        return self.meters[key]

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string: str = "{}") -> dict[str, float]:
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string: str = "{}") -> dict[str, float]:
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}