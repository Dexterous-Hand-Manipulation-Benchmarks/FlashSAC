import argparse
import json
import os
import re
from collections import defaultdict
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import wandb

# isort: split
import genesis as gs
import gymnasium as gym
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.evaluation import (
    EVAL_SEED,
    EvalScore,
    rollout_episode,
    save_trajectory,
    score,
    zero_action_floor,
)
from dexmachina.envs.factory import make_eval_env, task_hash
from dexmachina.envs.run_spec import (
    RUN_SPEC_FILENAME,
    DexmachinaState,
    current_dexmachina_state,
    load_run_spec,
)
from dexmachina.eval.compute_add import compute_for_eval
from genesis.utils import set_random_seed  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from flash_rl.agents import create_agent
from flash_rl.agents.base_agent import BaseAgent
from flash_rl.agents.flashSAC.agent import OBS_NORMALIZER_FILENAME
from flash_rl.common.torch_precision import enable_fast_matmul
from flash_rl.envs import create_envs
from flash_rl.envs.dexmachina import (
    IMAGE_NDIM,
    REPORTED_INTERACTION_STEPS,
    RGB_CHANNELS,
    RUN_CONFIG_FILENAME,
    compose_eval_config,
    curriculum_epoch,
    to_uint8_rgb,
)
from flash_rl.types import Tensor as FlashTensor

# Force global initialization of Genesis and make gs.init idempotent
_orig_gs_init = gs.init


def _safe_gs_init(*args, **kwargs):
    try:
        return _orig_gs_init(*args, **kwargs)
    except Exception as e:
        if "already initialized" in str(e).lower():
            return
        raise e


gs.init = _safe_gs_init

GRAYSCALE_NDIM = 2
MIN_FRAME_SIDE_PX = 10
RENDER_EVERY_N_STEPS = 2
DEFAULT_VIDEO_FPS = 30
# What Genesis' camera.render raises (gs.raise_exception, or the renderer's RuntimeError).
# A missed frame only shortens the video, so it is counted and reported, not fatal.
RENDER_ERRORS = (gs.GenesisException, RuntimeError)
# Success = per-frame ADD below the threshold, in metres.
ADD_SUCCESS_THRESHOLDS_M = {"success_2cm": 0.02, "success_5cm": 0.05}

# The eval agent never samples from or trains on its replay buffer, so it gets the smallest one.
EVAL_BUFFER_LENGTH = 1
# Not "eval/": the in-training evaluation logs its own eval/* keys, and the two must not share a chart.
BENCHMARK_PREFIX = "benchmark"
TRAJECTORY_FILENAME = "benchmark_trajectory_{sha}.npz"
SHORT_SHA_LENGTH = 12
# train.py names a checkpoint directory step<interaction step>.
CHECKPOINT_DIRNAME = re.compile(r"step(\d+)")
# Arguments of the old protocol; the new one takes everything from the run's saved files.
OLD_PROTOCOL_ARGUMENTS = ("step", "num_episodes", "config_path", "config_name", "overrides", "video")

_original_init = BaseEnv.__init__


def _init_with_camera(self, env_cfg, *args, **kwargs):
    if type(env_cfg).__name__ == "DictConfig":
        OmegaConf.set_struct(env_cfg, False)

    if "scene_kwargs" not in env_cfg or env_cfg["scene_kwargs"] is None:
        env_cfg["scene_kwargs"] = {}
    env_cfg["scene_kwargs"]["use_visualizer"] = True
    env_cfg["scene_kwargs"]["show_viewer"] = False

    env_cfg["record_video"] = True
    env_cfg["render_camera"] = "front"
    env_cfg["render_segmentation"] = False

    if "camera_kwargs" not in env_cfg or not env_cfg["camera_kwargs"]:
        env_cfg["camera_kwargs"] = {
            "front": {
                "res": (512, 512),
                "pos": (0.5, -1.5, 1.2),
                "lookat": (0.0, -1.58, 2.0),
                "fov": 40,
            }
        }

    _original_init(self, env_cfg, *args, **kwargs)


def enable_video_recording():
    """Make every BaseEnv built afterwards record from a front camera (dexmachina cannot render on macOS)."""
    BaseEnv.__init__ = _init_with_camera


def sanitize_frame(frame):
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if not isinstance(frame, np.ndarray):
        return None

    while frame.ndim > IMAGE_NDIM and frame.shape[0] == 1:
        frame = frame[0]

    if frame.ndim == GRAYSCALE_NDIM:
        frame = np.stack([frame] * RGB_CHANNELS, axis=-1)

    frame = to_uint8_rgb(frame)

    if frame.ndim == IMAGE_NDIM and frame.shape[-1] == RGB_CHANNELS and min(frame.shape[:2]) > MIN_FRAME_SIDE_PX:
        return frame
    return None


def video_fps(uenv):
    """Frame rate that plays every RENDER_EVERY_N_STEPS-th simulation step in real time."""
    if hasattr(uenv, "dt") and uenv.dt > 0:
        return int(1.0 / uenv.dt / RENDER_EVERY_N_STEPS)
    return DEFAULT_VIDEO_FPS


def write_video_file(frames, video_path, fps):
    try:
        from moviepy import ImageSequenceClip  # noqa: PLC0415  (optional: falls back to imageio)
        clip_obj = ImageSequenceClip(frames, fps=fps)
        clip_obj.write_videofile(video_path, logger=None)
        return True
    except (ImportError, AttributeError):
        pass

    try:
        import imageio  # noqa: PLC0415  (optional: the fallback video writer)
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264")
        for f in frames:
            writer.append_data(f)
        writer.close()
        return True
    # imageio: missing ffmpeg or codec (RuntimeError, ValueError), failed write (OSError)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"[WARNING] Video encoding failed, continuing without a video: {e}")
        return False


def get_demo_state_tensor(uenv, device):
    if hasattr(uenv, "demo_data") and uenv.demo_data is not None:
        obj_pos = uenv.demo_data["obj_pos"]
        obj_quat = uenv.demo_data["obj_quat"]
        obj_arti = uenv.demo_data["obj_arti"]
        if isinstance(obj_pos, torch.Tensor):
            obj_pos = obj_pos.cpu().numpy()
        if isinstance(obj_quat, torch.Tensor):
            obj_quat = obj_quat.cpu().numpy()
        if isinstance(obj_arti, torch.Tensor):
            obj_arti = obj_arti.cpu().numpy()
        if obj_arti.ndim == 1:
            obj_arti = obj_arti[:, None]
        arr = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
        return torch.tensor(arr, dtype=torch.float32, device=device)
    elif hasattr(uenv, "demo_tensors") and uenv.demo_tensors is not None:
        pos = uenv.demo_tensors["obj_pos"]
        quat = uenv.demo_tensors["obj_quat"]
        arti = uenv.demo_tensors["obj_arti"]
        if arti.ndim == 1:
            arti = arti.unsqueeze(-1)
        return torch.cat([pos, quat, arti], dim=-1).to(device)
    return None


def run_eval_pass(target_env, uenv, agent, metric_prefix, file_tag, out_dir, demo_state_tensor, obj_name):
    """Executes a single rollout pass, writes video/npy files, and logs to metric_prefix."""
    ep_len = uenv.max_episode_length
    reset_res = target_env.reset()
    obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res

    target_obj = None
    if hasattr(uenv, "objects") and hasattr(uenv, "object_names") and len(uenv.object_names) > 0:
        target_obj = uenv.objects[uenv.object_names[0]]

    transition = {"next_observation": obs}
    eval_data = defaultdict(list)
    accumulated_rewards = 0.0
    manual_renders = []
    failed_renders = []

    print(f"[INFO] Running rollout for '{file_tag}' ({ep_len} steps)...")
    for i in range(ep_len):
        with torch.no_grad():
            env_step = uenv.episode_length_buf.cpu().numpy()[0] if hasattr(uenv, "episode_length_buf") else i
            actions = agent.sample_actions(interaction_step=0, prev_transition=transition, training=False)
            actions = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)

            step_res = target_env.step(actions)
            obs, rew = step_res[0], step_res[1]
            transition = {"next_observation": obs}

            if isinstance(rew, torch.Tensor):
                accumulated_rewards += rew.mean().item()
            else:
                accumulated_rewards += float(np.mean(rew))

            if target_obj is not None:
                obj_pos, obj_quat, obj_arti = target_obj.root_pos, target_obj.root_quat, target_obj.dof_pos
                if obj_arti.ndim == 1:
                    obj_arti = obj_arti.unsqueeze(-1)
                obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
                eval_data["obj_state"].append(obj_state.detach().cpu().numpy())

            if demo_state_tensor is not None:
                demo_idx = min(env_step, len(demo_state_tensor) - 1)
                eval_data["demo_state"].append(demo_state_tensor[demo_idx].detach().cpu().numpy())

            if i % RENDER_EVERY_N_STEPS == 0 and getattr(uenv, "_floating_camera", None) is not None:
                try:
                    r_out = uenv._floating_camera.render()
                    frame = r_out[0] if isinstance(r_out, tuple) else r_out
                    sanitized = sanitize_frame(frame)
                    if sanitized is not None:
                        manual_renders.append(sanitized)
                except RENDER_ERRORS as e:
                    failed_renders.append(e)

    if failed_renders:
        print(
            f"[WARNING] {len(failed_renders)} frames of '{file_tag}' could not be rendered and are missing "
            f"from the video; first error: {failed_renders[0]!r}"
        )

    # Save rollout arrays using clean file tag
    eval_data = {k: np.stack(v) for k, v in eval_data.items()}
    out_file = os.path.join(out_dir, f"{file_tag}_eval.npy")
    np.save(out_file, eval_data)

    video_path = None
    clean_renders = [sanitize_frame(r) for r in manual_renders if sanitize_frame(r) is not None]
    if len(clean_renders) > 0:
        video_path = os.path.join(out_dir, f"{file_tag}_video.mp4")
        if write_video_file(clean_renders, video_path, fps=video_fps(uenv)):
            print(f"[SUCCESS] Saved video for {file_tag} to {video_path}")

    # Build metric dictionary under metric_prefix
    metrics = {
        f"{metric_prefix}/return": accumulated_rewards,
        f"{metric_prefix}/ep_length": float(ep_len),
    }

    stats_path = os.path.join(out_dir, f"{file_tag}_add_stats.json")
    add_npy_path = os.path.join(out_dir, f"{file_tag}_add.npy")
    try:
        # overwrite=True: compute_for_eval always writes both the stats and the ADD file.
        compute_for_eval(
            eval_path=out_file,
            obj_name=obj_name,
            out_name=f"{file_tag}_add.npy",
            stats_name=f"{file_tag}_add_stats.json",
            overwrite=True,
            output_cache=False,
            cache_dir=None,  # load_part_verts (compute_add.py:65) reads it only when output_cache is set
        )
    except Exception as e:
        raise RuntimeError(
            f"ADD computation failed for '{file_tag}' (cause above). Without ADD the eval metrics would look "
            "complete but lack the primary metric, so the evaluation stops. Fix the cause and rerun."
        ) from e

    with open(stats_path, "r") as f:
        stats = json.load(f)
    metrics[f"{metric_prefix}/auc_add"] = stats["auc"]["overall"]
    metrics[f"{metric_prefix}/add"] = stats["mean_add"]["overall"]

    add_data = np.load(add_npy_path, allow_pickle=True).item()
    for name, threshold in ADD_SUCCESS_THRESHOLDS_M.items():
        per_part = [np.mean(add_data[part] < threshold) for part in add_data]
        metrics[f"{metric_prefix}/{name}"] = float(np.mean(per_part))

    return metrics, video_path


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """One checkpoint scored on the shared eval protocol, with what is needed to compare it across runs."""

    eval_score: EvalScore
    task_hash: str
    saved_state: DexmachinaState
    evaluated_state: DexmachinaState
    floor: EvalScore
    checkpoint_step: int
    env_step: int
    eval_seed: int

    def metrics(self) -> dict[str, float]:
        """What goes into the W&B history."""
        result = self.eval_score
        metrics = {
            "auc_add": result.auc_add,
            "auc_add_of_mean": result.auc_add_of_mean,
            "mean_add": result.mean_add,
            "success_2cm": result.success_2cm,
            "success_5cm": result.success_5cm,
            "nan_rate": result.nan_rate,
            "drop_rate": result.drop_rate,
            "floor/auc_add": self.floor.auc_add,
            "checkpoint_step": float(self.checkpoint_step),
            "env_step": float(self.env_step),
        }
        metrics.update({f"auc_add_part/{part}": value for part, value in result.auc_add_part.items()})
        return {f"{BENCHMARK_PREFIX}/{name}": value for name, value in metrics.items()}

    def provenance(self) -> dict[str, str | bool | int | None]:
        """What goes into the W&B run summary: it identifies the evaluation rather than being plotted."""
        provenance: dict[str, str | bool | int | None] = {
            "task_hash": self.task_hash,
            "eval_seed": self.eval_seed,
            "dexmachina_sha/saved": self.saved_state.sha,
            "dexmachina_sha/evaluated": self.evaluated_state.sha,
            "dexmachina_dirty/saved": self.saved_state.dirty,
            "dexmachina_dirty/evaluated": self.evaluated_state.dirty,
            "dexmachina_diff_sha256/saved": self.saved_state.diff_sha256,
            "dexmachina_diff_sha256/evaluated": self.evaluated_state.diff_sha256,
        }
        return {f"{BENCHMARK_PREFIX}/{name}": value for name, value in provenance.items()}


def check_observation_normalizer(checkpoint_dir: Path, normalize_observation: bool) -> None:
    """The saved config and the checkpoint must agree on the normalizer, because agent.load skips a missing file.

    Only the observation normalizer is checked here; the actor weights are read by agent.load, which fails on a
    missing file itself.
    """
    has_file = (checkpoint_dir / OBS_NORMALIZER_FILENAME).is_file()
    if has_file and not normalize_observation:
        raise ValueError(
            f"{checkpoint_dir} has an {OBS_NORMALIZER_FILENAME}, but the run's saved config has "
            "normalize_observation false. The agent would see raw observations although it trained on normalized "
            "ones, and its actions would be wrong without any error. Likely cause: the config.yaml next to the "
            "checkpoint is not the one of this run. Use the config.yaml the run saved."
        )
    if normalize_observation and not has_file:
        raise FileNotFoundError(
            f"{checkpoint_dir / OBS_NORMALIZER_FILENAME} does not exist, but the run's saved config has "
            "normalize_observation true. agent.load would skip the missing file and evaluate with an untrained "
            "normalizer, so the score would be wrong without any error. Point at a checkpoint directory that "
            "agent.save wrote, or at the right run."
        )


def checkpoint_step(checkpoint_dir: Path) -> int:
    match = CHECKPOINT_DIRNAME.fullmatch(checkpoint_dir.name)
    if match is None:
        raise ValueError(
            f"{checkpoint_dir.name!r} is not a checkpoint directory name (step<interaction step>, as train.py "
            "writes it). The env step of the evaluation is computed from that number. Point at a step directory "
            "of the run, not at the run root."
        )
    return int(match.group(1))


def build_eval_agent(
    agent_config: DictConfig, obs_dim: int, action_dim: int, action_clip: float, device: str
) -> BaseAgent[Any]:
    """The agent of a run's saved config, on `device`, for an env with these observation and action sizes."""
    if agent_config.asymmetric_observation:
        raise ValueError(
            "The run's saved config has asymmetric_observation true. The evaluation hands the agent the env's "
            "policy observation only, so it would act on a different input than in training. Asymmetric "
            "observations are not supported by the benchmark evaluation; evaluate with --old-protocol or add support."
        )
    # The replay buffer is allocated at its configured size (1M transitions in the reported runs) and never used
    # here, and the optimizer state is not needed to act.
    eval_overrides = OmegaConf.create(
        {
            "device_type": device,
            "buffer_device_type": device,
            "buffer_max_length": EVAL_BUFFER_LENGTH,
            "buffer_min_length": EVAL_BUFFER_LENGTH,
            "load_optimizer": False,
        }
    )
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-action_clip, high=action_clip, shape=(action_dim,), dtype=np.float32)
    return create_agent(
        observation_space=observation_space,
        action_space=action_space,
        env_info={},
        cfg=OmegaConf.merge(agent_config, eval_overrides),
    )


def deterministic_actions(agent: BaseAgent[Any], observations: torch.Tensor, device: str) -> torch.Tensor:
    """The agent's mean action (temperature 0); the observation normalizer is not updated."""
    transition: MutableMapping[str, FlashTensor] = {"next_observation": observations}
    actions = agent.sample_actions(interaction_step=0, prev_transition=transition, training=False)
    return torch.as_tensor(actions, dtype=torch.float32, device=device)


def evaluate_benchmark(
    checkpoint_dir: Path, device: str, *, eval_seed: int = EVAL_SEED, allow_dirty_dexmachina: bool = False
) -> BenchmarkEvaluation:
    """Scores one checkpoint on the shared eval protocol, from what its run saved.

    `checkpoint_dir` is a step directory written by agent.save; the run's env_spec.json and config.yaml sit in its
    parent. Genesis must be initialised for `device`.
    The agent acts with the mean of its policy (temperature 0, tanh of the mean) and a frozen observation normalizer.
    The floor (a zero-action episode on the same env) runs before the policy's episode.
    The trajectory is saved next to the checkpoint as `benchmark_trajectory_<short dexmachina sha>.npz`.
    A dexmachina checkout with uncommitted changes, at training or now, is refused unless `allow_dirty_dexmachina`.
    """
    run_dir = checkpoint_dir.parent
    for filename in (RUN_SPEC_FILENAME, RUN_CONFIG_FILENAME):
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(
                f"{run_dir / filename} does not exist. Runs write it into their checkpoint root when they start, so "
                "this run predates it or the directory is wrong. Point at a step directory of a newer run, or "
                "evaluate this run with --old-protocol."
            )
    step = checkpoint_step(checkpoint_dir)
    run_spec = load_run_spec(run_dir)
    config = OmegaConf.load(run_dir / RUN_CONFIG_FILENAME)
    agent_config = cast(DictConfig, config.agent)
    check_observation_normalizer(checkpoint_dir, bool(agent_config.normalize_observation))

    saved_state = run_spec.dexmachina_state
    evaluated_state = current_dexmachina_state()
    if evaluated_state.sha != saved_state.sha:
        print(
            f"[WARNING] The run was trained with dexmachina {saved_state.sha[:12]} but is evaluated with "
            f"{evaluated_state.sha[:12]}. The env may behave differently from training, and the score is only "
            "comparable with other evaluations at this commit."
        )
    if saved_state.dirty or evaluated_state.dirty:
        state = (
            f"dexmachina had uncommitted changes (training: {saved_state.dirty}, evaluation: {evaluated_state.dirty}), "
            "so its commit does not say which code ran"
        )
        if not allow_dirty_dexmachina:
            raise RuntimeError(
                f"{state}, and the score could not be reproduced or compared with other runs. Commit or stash the "
                "changes in dexmachina (for training: retrain from a clean checkout), or pass "
                "--allow-dirty-dexmachina for a debugging evaluation whose numbers are not reported."
            )
        print(f"[WARNING] {state}. Not for reported numbers.")
    spec = run_spec.spec
    task = task_hash(spec)

    enable_fast_matmul()
    env: Any = make_eval_env(spec, device)
    set_random_seed(eval_seed)
    agent = build_eval_agent(agent_config, int(env.obs_dim), int(env.action_dim), float(env.action_clip), device)
    agent.load(str(checkpoint_dir))

    floor = zero_action_floor(env, spec.hand, spec.clip)
    trajectory = rollout_episode(
        env, lambda obs: deterministic_actions(agent, obs, device), spec.hand, spec.clip, task, evaluated_state.sha
    )
    sha = evaluated_state.sha[:SHORT_SHA_LENGTH]
    save_trajectory(checkpoint_dir / TRAJECTORY_FILENAME.format(sha=sha), trajectory)
    return BenchmarkEvaluation(
        score(trajectory), task, saved_state, evaluated_state, floor, step, step * spec.num_envs, eval_seed
    )


def run_benchmark(args: argparse.Namespace) -> None:
    backend: Any = gs.cpu if args.device.startswith("cpu") else gs.gpu  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportAttributeAccessIssue]
    gs.init(backend=backend, logging_level="warning")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    evaluation = evaluate_benchmark(
        Path(args.checkpoint_path).resolve(),
        args.device,
        eval_seed=args.eval_seed,
        allow_dirty_dexmachina=args.allow_dirty_dexmachina,
    )
    metrics = evaluation.metrics()
    for name, value in {**metrics, **evaluation.provenance()}.items():
        print(f"{name}: {value}")
    if args.wandb_run_id:
        run = wandb.init(id=args.wandb_run_id, project=args.project, entity=args.entity, resume="allow")
        wandb.define_metric(f"{BENCHMARK_PREFIX}/*", step_metric=f"{BENCHMARK_PREFIX}/env_step")
        wandb.log(metrics)
        run.summary.update(evaluation.provenance())  # pyright: ignore[reportUnknownMemberType]
        wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--config_path", type=str, default="../../FlashSAC/configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--project", type=str, default="dexmachina_clean")
    parser.add_argument(
        "--entity", type=str, default=None, help="W&B entity (default: $WANDB_ENTITY / your W&B default)"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True,
                        help="record eval videos from a front camera; --no-video where Genesis cannot render "
                             "(macOS, machines without a display)")
    parser.add_argument("--old-protocol", action="store_true",
                        help="the earlier assisted/unassisted evaluation, built from configs/ and the overrides; "
                             "for runs without env_spec.json and config.yaml")
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED,
                        help="seed set after the eval env is built; the default is shared by all methods")
    parser.add_argument("--allow-dirty-dexmachina", action="store_true",
                        help="evaluate although dexmachina had uncommitted changes at training or now; for "
                             "debugging, not for reported numbers (the lab runs' files are recorded as dirty)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.old_protocol:
        run_old_protocol(args)
        return
    given = [name for name in OLD_PROTOCOL_ARGUMENTS if getattr(args, name) != parser.get_default(name)]
    if given:
        raise SystemExit(
            f"error: --{', --'.join(given)} only apply to --old-protocol. The default evaluation takes the env and "
            "the agent from the files the run saved (env_spec.json, config.yaml) and the step from the checkpoint "
            "directory name, so these arguments would be ignored. Drop them, or add --old-protocol."
        )
    run_benchmark(args)


def run_old_protocol(args: argparse.Namespace) -> None:
    if args.video:
        enable_video_recording()

    config_dir = os.path.abspath(args.config_path)
    cfg, train_env_scale = compose_eval_config(config_dir, args.config_name, args.overrides)

    train_env, eval_env, _ = create_envs(**cfg.env)
    target_env = eval_env if eval_env is not None else train_env
    uenv = getattr(target_env, "env", getattr(target_env, "base_env", getattr(target_env, "unwrapped", target_env)))
    device = torch.device(args.device)
    demo_state_tensor = get_demo_state_tensor(uenv, device)

    agent = create_agent(
        observation_space=target_env.observation_space,
        action_space=target_env.action_space,
        env_info={},
        cfg=cfg.agent,
    )
    print(f"[INFO] Loading checkpoint weights from: {args.checkpoint_path}")
    agent.load(args.checkpoint_path)

    ckpt_dir = os.path.abspath(args.checkpoint_path)
    step_tag = f"step{args.step}" if args.step is not None else "eval"
    out_dir = os.path.join(ckpt_dir, f"{step_tag}_results")
    os.makedirs(out_dir, exist_ok=True)

    clip_str = cfg.env.clip
    obj_name = clip_str.split("-")[0]
    log_payload = {}
    target_obj = uenv.objects[uenv.object_names[0]] if (hasattr(uenv, "objects") and uenv.object_names) else None

    # PASS 1: ASSISTED EVALUATION (eval/assisted/*)
    if hasattr(uenv, "set_curriculum") and args.step is not None:
        matched_epoch = curriculum_epoch(args.step, REPORTED_INTERACTION_STEPS, cfg.env.max_epochs)
        uenv.set_curriculum(matched_epoch)
        print(f"[INFO] Pass 1: Active curriculum epoch set to {matched_epoch}")

    m_assisted, vid_assisted = run_eval_pass(
        target_env, uenv, agent,
        metric_prefix="eval/assisted",
        file_tag="assisted",
        out_dir=out_dir,
        demo_state_tensor=demo_state_tensor,
        obj_name=obj_name
    )
    log_payload.update(m_assisted)
    if vid_assisted:
        log_payload["eval/assisted/video"] = wandb.Video(
            vid_assisted, fps=video_fps(uenv), format="mp4", caption=f"Assisted Step {args.step}"
        )

    # PASS 2: UNASSISTED EVALUATION (eval/* benchmark)
    if target_obj is not None and getattr(target_obj, "actuated", False):
        target_obj.set_joint_gains(0.0, 0.0, force_range=0.0)
        print("[INFO] Pass 2: Object gains zeroed out for unassisted physics.")

    m_unassisted, vid_unassisted = run_eval_pass(
        target_env, uenv, agent,
        metric_prefix="eval/milestone",
        file_tag="unassisted",
        out_dir=out_dir,
        demo_state_tensor=demo_state_tensor,
        obj_name=obj_name
    )
    log_payload.update(m_unassisted)
    if vid_unassisted:
        log_payload["eval/video"] = wandb.Video(
            vid_unassisted, fps=video_fps(uenv), format="mp4", caption=f"Benchmark Step {args.step}"
        )

    # WANDB LOGGING
    if args.wandb_run_id:
        print(f"[INFO] Logging results to WandB run: {args.wandb_run_id}...")
        wandb.init(id=args.wandb_run_id, project=args.project, entity=args.entity, resume="allow")
        env_step_num = (args.step * train_env_scale) if args.step is not None else None
        if args.step is not None:
            log_payload["global/step"] = env_step_num
            log_payload["global/env_step"] = env_step_num

        # Resume runs without passing explicit step argument to avoid out-of-order step errors
        wandb.log(log_payload)
        wandb.finish()

    target_env.close()
    print(f"[SUCCESS] Completed dual evaluation for checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
