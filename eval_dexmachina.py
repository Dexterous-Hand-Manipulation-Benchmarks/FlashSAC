import os
import sys

# Force global initialization of Genesis
import genesis as gs

import argparse
import torch
import numpy as np
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

from flash_rl.agents import create_agent
from flash_rl.envs import create_envs
from flash_rl.evaluation import evaluate, record_video

# --- THE ENGINE IGNITION BYPASS ---
from dexmachina.envs.base_env import BaseEnv

_original_init = BaseEnv.__init__


def _mocked_init(self, env_cfg, *args, **kwargs):
    if type(env_cfg).__name__ == "DictConfig":
        OmegaConf.set_struct(env_cfg, False)

    # 1. Force Genesis to boot the visualizer
    if 'scene_kwargs' not in env_cfg:
        env_cfg['scene_kwargs'] = {}
    env_cfg['scene_kwargs']['use_visualizer'] = True

    # 2. Tell Dexmachina to build the camera
    env_cfg['record_video'] = True
    env_cfg['render_camera'] = 'front'

    # 3. Supply the camera parameters so Dexmachina doesn't drop it
    if 'camera_kwargs' not in env_cfg or not env_cfg['camera_kwargs']:
        env_cfg['camera_kwargs'] = {
            'front': {
                'res': (256, 256),
                'pos': (0.0, -1.6, 2.2),
                'lookat': (0.0, -0.1, 1.2),
                'fov': 30
            }
        }

    _original_init(self, env_cfg, *args, **kwargs)


BaseEnv.__init__ = _mocked_init


# ----------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to saved agent checkpoint directory")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID to attach evaluation videos to")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--config_path", type=str, default="../../FlashSAC/configs",
                        help="Path to Hydra configs directory")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--step", type=int, default=None, help="The exact interaction step to log this video at in W&B")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_path)
    print(f"[INFO] Loading Hydra configs from: {config_dir}")

    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    if args.wandb_run_id:
        wandb_project = cfg.logger.project if (
                    hasattr(cfg, "logger") and hasattr(cfg.logger, "project")) else "dexmachina"
        wandb.init(id=args.wandb_run_id, project=wandb_project, entity="schrapsi-kit", resume="allow")

    print(f"[INFO] Building environment: {cfg.env.env_name}")
    train_env, eval_env, record_env = create_envs(**cfg.env)

    _, env_info = eval_env.reset()

    agent = create_agent(
        observation_space=eval_env.observation_space,
        action_space=eval_env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    print(f"[INFO] Loading checkpoint weights from: {args.checkpoint_path}")
    agent.load(args.checkpoint_path)

    print("[INFO] Running evaluation rollouts...")
    eval_info = evaluate(agent, eval_env, num_episodes=args.num_episodes, env_type=cfg.env.env_type)
    print(f"[RESULTS] Eval Metrics: {eval_info}")

    print("[INFO] Rendering evaluation videos...")
    video_info = record_video(agent, record_env, num_episodes=1, env_type=cfg.env.env_type)

    if "video" in video_info and wandb.run is not None:
        video_array = video_info["video"][0]

        # W&B Encoder Safety Fix
        if video_array.dtype != np.uint8:
            if video_array.max() <= 1.01:
                video_array = video_array * 255.0
            video_array = np.clip(video_array, 0, 255).astype(np.uint8)

        # Add a caption instead of forcing the global step axis backward
        cap_text = f"Checkpoint Step: {args.step}" if args.step else "Eval Video"
        log_data = {"eval/video": wandb.Video(video_array, fps=30, format="gif", caption=cap_text)}

        if args.step is not None:
            log_data["global/env_step"] = args.step
            eval_info["global/env_step"] = args.step

        wandb.log(log_data, step=args.step)
        print(f"[INFO] Successfully uploaded video for {cap_text} to WandB!")

    if wandb.run is not None:
        wandb.log(eval_info, step=args.step)
        wandb.finish()

    eval_env.close()
    record_env.close()


if __name__ == "__main__":
    main()