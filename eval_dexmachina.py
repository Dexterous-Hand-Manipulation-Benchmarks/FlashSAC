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

    # 1. Force Genesis to boot the visualizer
    if "scene_kwargs" not in env_cfg or env_cfg["scene_kwargs"] is None:
        env_cfg["scene_kwargs"] = {}
    env_cfg["scene_kwargs"]["use_visualizer"] = True
    env_cfg["scene_kwargs"]["show_viewer"] = False

    # 2. Tell Dexmachina to build the camera
    env_cfg["record_video"] = True
    env_cfg["render_camera"] = "front"
    env_cfg["render_segmentation"] = False

    # 3. Supply camera parameters
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
# --------------------------------------------------------


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
    """Encodes frames to MP4 supporting MoviePy 2.x, MoviePy 1.x, and imageio."""
    # Method 1: MoviePy 2.x
    try:
        from moviepy import ImageSequenceClip
        clip_obj = ImageSequenceClip(frames, fps=fps)
        clip_obj.write_videofile(video_path, logger=None)
        return True
    except (ImportError, AttributeError):
        pass

    # Method 2: MoviePy 2.x direct subpackage
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip_obj = ImageSequenceClip(frames, fps=fps)
        clip_obj.write_videofile(video_path, logger=None)
        return True
    except (ImportError, AttributeError):
        pass

    # Method 3: MoviePy 1.x legacy
    try:
        from moviepy.editor import ImageSequenceClip
        clip_obj = ImageSequenceClip(frames, fps=fps)
        clip_obj.write_videofile(video_path, logger=None)
        return True
    except (ImportError, AttributeError):
        pass

    # Method 4: imageio-ffmpeg
    try:
        import imageio
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264")
        for f in frames:
            writer.append_data(f)
        writer.close()
        return True
    except Exception as e:
        print(f"[WARNING] Video encoding fallback failed: {e}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint directory containing actor.pt")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID to attach evaluation videos and metrics to")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--config_path", type=str, default="../../FlashSAC/configs", help="Path to Hydra configs directory")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--step", type=int, default=None, help="The exact interaction step for W&B checkpoint logging")
    parser.add_argument("--project", type=str, default="dexmachina_clean", help="WandB project name")
    parser.add_argument("--entity", type=str, default="schrapsi-kit", help="WandB entity name")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_path)
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Instantiate Evaluation Environments via Native FlashSAC Builder
    if hasattr(cfg, "env"):
        cfg.env.num_train_envs = 1
        cfg.env.num_eval_envs = 1
        cfg.env.num_record_envs = 1

    train_env, eval_env, record_env = create_envs(**cfg.env)
    target_env = eval_env if eval_env is not None else train_env
    uenv = getattr(target_env, "env", getattr(target_env, "base_env", getattr(target_env, "unwrapped", target_env)))

    demo_state_tensor = get_demo_state_tensor(uenv, device)

    # 2. Build Agent with Matching Observation Space
    agent = create_agent(
        observation_space=target_env.observation_space,
        action_space=target_env.action_space,
        env_info={},
        cfg=cfg.agent,
    )

    print(f"[INFO] Loading checkpoint weights from: {args.checkpoint_path}")
    agent.load(args.checkpoint_path)

    ep_len = getattr(uenv, "max_episode_length", 200)

    # Unclamp object to evaluate under unassisted physics
    if hasattr(uenv, "objects") and hasattr(uenv, "object_names") and len(uenv.object_names) > 0:
        obj = uenv.objects[uenv.object_names[0]]
        if getattr(obj, "actuated", False):
            obj.set_joint_gains(0.0, 0.0, force_range=0.0)
    else:
        obj = None

    # Setup output directory
    ckpt_dir = os.path.abspath(args.checkpoint_path)
    step_tag = f"step{args.step}" if args.step is not None else "eval"
    out_dir = os.path.join(ckpt_dir, f"{step_tag}_results")
    os.makedirs(out_dir, exist_ok=True)

    # 3. Evaluation & Video Rollout Loop
    print(f"[INFO] Running evaluation rollout ({ep_len} steps)...")
    reset_res = target_env.reset()
    obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res

    if hasattr(uenv, "start_recording"):
        uenv.start_recording()
        uenv.max_video_frames = int(ep_len * 2)

    manual_renders = []
    if hasattr(uenv, "_floating_camera") and uenv._floating_camera is not None:
        try:
            r_out = uenv._floating_camera.render()
            frame = r_out[0] if isinstance(r_out, tuple) else r_out
            sanitized = sanitize_frame(frame)
            if sanitized is not None:
                manual_renders.append(sanitized)
        except Exception:
            pass

    eval_data = defaultdict(list)
    accumulated_rewards = 0.0

    # Initial transition dictionary required by FlashSAC
    transition = {"next_observation": obs}

    for i in range(ep_len):
        with torch.no_grad():
            env_step = uenv.episode_length_buf.cpu().numpy()[0] if hasattr(uenv, "episode_length_buf") else i

            actions = agent.sample_actions(interaction_step=0, prev_transition=transition, training=False)
            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu().numpy()
            else:
                actions = np.asarray(actions)

            step_res = target_env.step(actions)
            if len(step_res) == 5:
                obs, rew, dones, truncs, infos = step_res
            else:
                obs, rew, dones, infos = step_res

            # Update transition for next action sampling
            transition = {"next_observation": obs}

            if isinstance(rew, torch.Tensor):
                accumulated_rewards += rew.mean().item()
            elif isinstance(rew, np.ndarray):
                accumulated_rewards += float(np.mean(rew))
            else:
                accumulated_rewards += float(rew)

            if obj is not None:
                obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
                if obj_arti.ndim == 1:
                    obj_arti = obj_arti.unsqueeze(-1)
                obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
                eval_data["obj_state"].append(obj_state.detach().cpu().numpy())

            if demo_state_tensor is not None:
                demo_idx = min(env_step, len(demo_state_tensor) - 1)
                eval_data["demo_state"].append(demo_state_tensor[demo_idx].detach().cpu().numpy())

            rew_dict = getattr(uenv, "rew_dict", {})
            for key in ["pos_dist", "rot_dist", "arti_dist"]:
                if key in rew_dict:
                    val = rew_dict[key]
                    eval_data[key].append(val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val))

            if i % 2 == 0 and hasattr(uenv, "_floating_camera") and uenv._floating_camera is not None:
                try:
                    r_out = uenv._floating_camera.render()
                    frame = r_out[0] if isinstance(r_out, tuple) else r_out
                    sanitized = sanitize_frame(frame)
                    if sanitized is not None:
                        manual_renders.append(sanitized)
                except Exception:
                    pass

    ep_return = accumulated_rewards
    eval_data = {k: np.stack(v) for k, v in eval_data.items()}
    out_file = os.path.join(out_dir, "eval_ep0.npy")
    np.save(out_file, eval_data)

    # 4. Save MP4 Video
    last_video_path = None
    recorded_frames = uenv.get_recorded_frames(wait_for_max=False) if hasattr(uenv, "get_recorded_frames") else None
    source_frames = recorded_frames if (recorded_frames is not None and len(recorded_frames) > 0) else manual_renders

    if source_frames and len(source_frames) > 0:
        video_path = os.path.join(out_dir, "eval_video.mp4")
        clean_renders = [sanitize_frame(r) for r in source_frames if sanitize_frame(r) is not None]

        if len(clean_renders) > 0:
            fps = int(1.0 / uenv.dt / 2) if hasattr(uenv, "dt") and uenv.dt > 0 else 30
            success = write_video_file(clean_renders, video_path, fps=fps)
            if success:
                last_video_path = video_path
                print(f"[SUCCESS] Saved evaluation video to {video_path}")

    # 5. Compute ADD and AUC-ADD Metrics
    stats = None
    try:
        from dexmachina.eval.compute_add import compute_for_eval
        clip_name = getattr(cfg.env, "clip", "box-30-230")
        obj_name = clip_name.split("-")[0]
        stats_path = os.path.join(out_dir, "add_stats.json")
        cache_sub_verts = os.path.join(dexmachina_dir, "dexmachina", "eval", "sub_verts")
        compute_for_eval(
            eval_path=out_file,
            obj_name=obj_name,
            out_name="add.npy",
            stats_name="add_stats.json",
            overwrite=True,
            output_cache=False,
            cache_dir=cache_sub_verts if os.path.exists(cache_sub_verts) else "dexmachina/eval/sub_verts",
        )
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            print(f"[RESULTS] ADD-AUC: {stats.get('auc', {}).get('overall')} | ADD: {stats.get('mean_add', {}).get('overall')}")
    except Exception as e:
        print(f"[WARNING] Could not compute ADD metrics: {e}")

    # 6. Upload All Metrics & Video to WandB
    if args.wandb_run_id:
        print(f"[INFO] Connecting to WandB run {args.wandb_run_id} ({args.entity}/{args.project})...")
        wandb.init(id=args.wandb_run_id, project=args.project, entity=args.entity, resume="allow")

        # Convert interaction step to environment steps (step * 1024 envs)
        num_envs = getattr(cfg.env, "num_train_envs", 1024)
        env_step_num = (args.step * num_envs) if args.step is not None else None

        log_payload = {
            "eval/return": ep_return,
            "eval/ep_length": float(ep_len),
        }
        if stats is not None:
            if "auc" in stats and "overall" in stats["auc"]:
                log_payload["eval/auc_add"] = stats["auc"]["overall"]
            if "mean_add" in stats and "overall" in stats["mean_add"]:
                log_payload["eval/add"] = stats["mean_add"]["overall"]

        add_npy_path = os.path.join(out_dir, "add.npy")
        if os.path.exists(add_npy_path):
            try:
                add_data = np.load(add_npy_path, allow_pickle=True).item()
                succ_2cm_list = [np.mean(add_data[part] < 0.02) for part in add_data.keys()]
                succ_5cm_list = [np.mean(add_data[part] < 0.05) for part in add_data.keys()]
                log_payload["eval/success_2cm"] = float(np.mean(succ_2cm_list))
                log_payload["eval/success_5cm"] = float(np.mean(succ_5cm_list))
                print(
                    f"[RESULTS] Success (2cm): {log_payload['eval/success_2cm'] * 100:.2f}% | Success (5cm): {log_payload['eval/success_5cm'] * 100:.2f}%")
            except Exception as e:
                print(f"[WARNING] Could not compute threshold success rates: {e}")

        if last_video_path and os.path.exists(last_video_path):
            cap_text = f"Checkpoint Step: {args.step}" if args.step is not None else "Eval Video"
            fps = int(1.0 / uenv.dt / 2) if hasattr(uenv, "dt") and uenv.dt > 0 else 30
            log_payload["eval_video"] = wandb.Video(last_video_path, fps=fps, format="mp4", caption=cap_text)
            print(f"[INFO] Attached video to WandB payload.")

        if args.step is not None:
            log_payload["checkpoint_step"] = args.step
            log_payload["global/env_step"] = env_step_num

        if log_payload:
            wandb.log(log_payload)
            print(f"[SUCCESS] Logged evaluation payload to WandB: {list(log_payload.keys())}")

        wandb.finish()

    target_env.close()
    print(f"[SUCCESS] Evaluation completed for checkpoint {args.checkpoint_path}!")


if __name__ == "__main__":
    main()