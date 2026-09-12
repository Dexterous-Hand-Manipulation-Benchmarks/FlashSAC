import os
import sys
import argparse
from collections import defaultdict
import json
import numpy as np
import torch
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

# Force global initialization of Genesis and make gs.init idempotent
import genesis as gs

_orig_gs_init = gs.init


def _safe_gs_init(*args, **kwargs):
    try:
        return _orig_gs_init(*args, **kwargs)
    except Exception as e:
        if "already initialized" in str(e).lower():
            return
        raise e


gs.init = _safe_gs_init

# Workspace paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
flashsac_dir = os.path.join(base_dir, "FlashSAC")
dexmachina_dir = os.path.join(base_dir, "dexmachina")
for p in [base_dir, flashsac_dir, dexmachina_dir, os.path.dirname(__file__)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from flash_rl.agents import create_agent
from flash_rl.envs import create_envs
from dexmachina.envs.base_env import BaseEnv

# --- Camera & Visualizer Activation Hook for Genesis ---
_original_init = BaseEnv.__init__


def _mocked_init(self, env_cfg, *args, **kwargs):
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


BaseEnv.__init__ = _mocked_init


def sanitize_frame(frame):
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if not isinstance(frame, np.ndarray):
        return None

    while frame.ndim > 3 and frame.shape[0] == 1:
        frame = frame[0]

    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)

    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]

    if frame.dtype != np.uint8:
        if frame.max() <= 1.01 and frame.max() > 0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 3 and frame.shape[-1] == 3 and frame.shape[0] > 10 and frame.shape[1] > 10:
        return frame
    return None


def write_video_file(frames, video_path, fps=30):
    try:
        from moviepy import ImageSequenceClip
        clip_obj = ImageSequenceClip(frames, fps=fps)
        clip_obj.write_videofile(video_path, logger=None)
        return True
    except (ImportError, AttributeError):
        pass

    try:
        import imageio
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264")
        for f in frames:
            writer.append_data(f)
        writer.close()
        return True
    except Exception as e:
        print(f"[WARNING] Video encoding failed: {e}")
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
    ep_len = getattr(uenv, "max_episode_length", 200)
    reset_res = target_env.reset()
    obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res

    target_obj = None
    if hasattr(uenv, "objects") and hasattr(uenv, "object_names") and len(uenv.object_names) > 0:
        target_obj = uenv.objects[uenv.object_names[0]]

    transition = {"next_observation": obs}
    eval_data = defaultdict(list)
    accumulated_rewards = 0.0
    manual_renders = []

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

            if i % 2 == 0 and hasattr(uenv, "_floating_camera") and uenv._floating_camera is not None:
                try:
                    r_out = uenv._floating_camera.render()
                    frame = r_out[0] if isinstance(r_out, tuple) else r_out
                    sanitized = sanitize_frame(frame)
                    if sanitized is not None:
                        manual_renders.append(sanitized)
                except Exception:
                    pass

    # Save rollout arrays using clean file tag
    eval_data = {k: np.stack(v) for k, v in eval_data.items()}
    out_file = os.path.join(out_dir, f"{file_tag}_eval.npy")
    np.save(out_file, eval_data)

    video_path = None
    clean_renders = [sanitize_frame(r) for r in manual_renders if sanitize_frame(r) is not None]
    if len(clean_renders) > 0:
        video_path = os.path.join(out_dir, f"{file_tag}_video.mp4")
        fps = int(1.0 / uenv.dt / 2) if hasattr(uenv, "dt") and uenv.dt > 0 else 30
        if write_video_file(clean_renders, video_path, fps=fps):
            print(f"[SUCCESS] Saved video for {file_tag} to {video_path}")

    # Build metric dictionary under metric_prefix
    metrics = {
        f"{metric_prefix}/return": accumulated_rewards,
        f"{metric_prefix}/ep_length": float(ep_len),
    }

    try:
        from dexmachina.eval.compute_add import compute_for_eval
        stats_path = os.path.join(out_dir, f"{file_tag}_add_stats.json")
        cache_sub_verts = os.path.join(dexmachina_dir, "dexmachina", "eval", "sub_verts")
        compute_for_eval(
            eval_path=out_file,
            obj_name=obj_name,
            out_name=f"{file_tag}_add.npy",
            stats_name=f"{file_tag}_add_stats.json",
            overwrite=True,
            output_cache=False,
            cache_dir=cache_sub_verts if os.path.exists(cache_sub_verts) else "dexmachina/eval/sub_verts",
        )
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            metrics[f"{metric_prefix}/auc_add"] = stats.get("auc", {}).get("overall", 0.0)
            metrics[f"{metric_prefix}/add"] = stats.get("mean_add", {}).get("overall", 0.0)

        add_npy_path = os.path.join(out_dir, f"{file_tag}_add.npy")
        if os.path.exists(add_npy_path):
            add_data = np.load(add_npy_path, allow_pickle=True).item()
            succ_2cm = [np.mean(add_data[part] < 0.02) for part in add_data.keys()]
            succ_5cm = [np.mean(add_data[part] < 0.05) for part in add_data.keys()]
            metrics[f"{metric_prefix}/success_2cm"] = float(np.mean(succ_2cm))
            metrics[f"{metric_prefix}/success_5cm"] = float(np.mean(succ_5cm))
    except Exception as e:
        print(f"[WARNING] ADD calculation failed for {file_tag}: {e}")

    return metrics, video_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--config_path", type=str, default="../../FlashSAC/configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--project", type=str, default="dexmachina_clean")
    parser.add_argument("--entity", type=str, default="schrapsi-kit")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_path)
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_env_scale = int(cfg.env.get("num_train_envs", 1024))
    if hasattr(cfg, "env"):
        cfg.env.num_train_envs = 1
        cfg.env.num_eval_envs = 1
        cfg.env.num_record_envs = 1

    train_env, eval_env, _ = create_envs(**cfg.env)
    target_env = eval_env if eval_env is not None else train_env
    uenv = getattr(target_env, "env", getattr(target_env, "base_env", getattr(target_env, "unwrapped", target_env)))
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

    clip_str = getattr(cfg.env, "clip", "box-30-230")
    obj_name = clip_str.split("-")[0]
    log_payload = {}
    target_obj = uenv.objects[uenv.object_names[0]] if (hasattr(uenv, "objects") and uenv.object_names) else None

    # PASS 1: ASSISTED EVALUATION (eval/assisted/*)
    TOTAL_INTERACTION_STEPS = 48829  # 50_000_896 / 1024 envs
    MAX_EPOCHS = 10_000
    if hasattr(uenv, "set_curriculum") and args.step is not None:
        matched_epoch = int((args.step * MAX_EPOCHS) / TOTAL_INTERACTION_STEPS)
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
        fps = int(1.0 / uenv.dt / 2) if hasattr(uenv, "dt") and uenv.dt > 0 else 30
        log_payload["eval/assisted/video"] = wandb.Video(vid_assisted, fps=fps, format="mp4", caption=f"Assisted Step {args.step}")

    # PASS 2: UNASSISTED EVALUATION (eval/* benchmark)
    if target_obj is not None and getattr(target_obj, "actuated", False):
        target_obj.set_joint_gains(0.0, 0.0, force_range=0.0)
        print("[INFO] Pass 2: Object gains zeroed out for unassisted physics.")

    m_unassisted, vid_unassisted = run_eval_pass(
        target_env, uenv, agent,
        metric_prefix="eval",
        file_tag="unassisted",
        out_dir=out_dir,
        demo_state_tensor=demo_state_tensor,
        obj_name=obj_name
    )
    log_payload.update(m_unassisted)
    if vid_unassisted:
        fps = int(1.0 / uenv.dt / 2) if hasattr(uenv, "dt") and uenv.dt > 0 else 30
        log_payload["eval/video"] = wandb.Video(vid_unassisted, fps=fps, format="mp4", caption=f"Benchmark Step {args.step}")

    # WANDB LOGGING
    if args.wandb_run_id:
        print(f"[INFO] Logging results to WandB run: {args.wandb_run_id}...")
        wandb.init(id=args.wandb_run_id, project=args.project, entity=args.entity, resume="allow")
        env_step_num = (args.step * train_env_scale) if args.step is not None else None
        if args.step is not None:
            log_payload["checkpoint_step"] = args.step
            log_payload["global/env_step"] = env_step_num

        # Resume runs without passing explicit step argument to avoid out-of-order step errors
        wandb.log(log_payload)
        wandb.finish()

    target_env.close()
    print(f"[SUCCESS] Completed dual evaluation for checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()