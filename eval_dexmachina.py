import os
import sys

# Force global initialization of Genesis to prevent local circular import crashes
import genesis as gs

import argparse
import torch
import numpy as np
import wandb
import hydra
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs import create_envs
from flash_rl.evaluation import evaluate, record_video

# --- CAMERA INJECTION ---
# Intercept BaseEnv initialization to force the camera on, bypassing FlashSAC's stubborn config loader
from dexmachina.envs.base_env import BaseEnv

_original_init = BaseEnv.__init__


def _mocked_init(self, env_cfg, *args, **kwargs):
    env_cfg['record_video'] = True
    _original_init(self, env_cfg, *args, **kwargs)


BaseEnv.__init__ = _mocked_init


# -------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to saved agent checkpoint directory")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID to attach evaluation videos to")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--config_path", type=str, default="../../FlashSAC/configs",
                        help="Path to Hydra configs directory")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()

    # Convert config path to absolute path to avoid Hydra relative path issues
    config_dir = os.path.abspath(args.config_path)
    print(f"[INFO] Loading Hydra configs from: {config_dir}")

    # Initialize Hydra config using absolute path
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    OmegaConf.set_struct(cfg, False)

    # Inject it exactly where Dexmachina's BaseEnv looks for it
    if hasattr(cfg.env, "env_cfg"):
        cfg.env.env_cfg.record_video = True
    else:
        cfg.env.record_video = True

    OmegaConf.set_struct(cfg, True)

    # Resume WandB logging under team entity 'schrapsi-kit'
    if args.wandb_run_id:
        wandb_project = cfg.logger.project if (
                    hasattr(cfg, "logger") and hasattr(cfg.logger, "project")) else "dexmachina"
        wandb.init(
            id=args.wandb_run_id,
            project=wandb_project,
            entity="schrapsi-kit",
            resume="allow",
        )

    print(f"[INFO] Building environment: {cfg.env.env_name}")
    train_env, eval_env, record_env = create_envs(**cfg.env)

    # Obtain env_info dynamically from reset
    _, env_info = eval_env.reset()

    # Create Agent and Load Weights
    agent = create_agent(
        observation_space=eval_env.observation_space,
        action_space=eval_env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    print(f"[INFO] Loading checkpoint weights from: {args.checkpoint_path}")
    agent.load(args.checkpoint_path)

    # 1. Compute Standard Return / Success Metrics
    print("[INFO] Running evaluation rollouts...")
    eval_info = evaluate(agent, eval_env, num_episodes=args.num_episodes, env_type=cfg.env.env_type)
    print(f"[RESULTS] Eval Metrics: {eval_info}")

    # 2. Generate and Log Videos
    print("[INFO] Rendering evaluation videos...")
    video_info = record_video(agent, record_env, num_episodes=1, env_type=cfg.env.env_type)

    if "video" in video_info and wandb.run is not None:
        video_array = video_info["video"][0]

        # --- W&B Encoder Safety Fix ---
        if video_array.dtype != np.uint8:
            if video_array.max() <= 1.01:
                video_array = video_array * 255.0
            video_array = np.clip(video_array, 0, 255).astype(np.uint8)
        # ------------------------------

        wandb.log({"eval_video": wandb.Video(video_array, fps=30, format="mp4")})
        print("[INFO] Successfully uploaded video to WandB!")

    if wandb.run is not None:
        wandb.log(eval_info)
        wandb.finish()

    eval_env.close()
    record_env.close()


if __name__ == "__main__":
    main()