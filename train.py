import os
import sys

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "float32"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# Force global initialization of Genesis to prevent local circular import crashes
import genesis as gs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOGGING_DIR = os.path.join(PROJECT_ROOT, "logging")

for p in [LOGGING_DIR, PROJECT_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


import argparse
import random
from datetime import datetime
from typing import Optional

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.common import create_logger
from flash_rl.envs import create_envs
from flash_rl.evaluation import evaluate, record_video
from flash_rl.types import Tensor
from wandb_logger import UniversalLogger


def run(args: argparse.Namespace) -> None:
    ###############################
    # configs
    ###############################
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    # eval resolver
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)

    # initialize config
    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)

    # --- DYNAMIC LOGGER INSTALLATION ---
    hand_name = getattr(cfg.env, "hand", "unknown_hand") if hasattr(cfg, "env") else "unknown_hand"
    clip_name = getattr(cfg.env, "clip", "unknown_clip") if hasattr(cfg, "env") else "unknown_clip"
    num_envs = getattr(cfg, "num_train_envs", 1024)
    seed = getattr(cfg, "seed", 0)

    UniversalLogger.install(
        algo="flashsac",
        hand=str(hand_name),
        clip=str(clip_name),
        seed=int(seed),
        num_envs=int(num_envs),
    )
    # -----------------------------------

    ###############################
    # seeding / configuration
    ###############################

    # Set random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    #############################
    # envs
    #############################
    train_env, eval_env, record_env = create_envs(**cfg.env)

    observation_space = train_env.observation_space
    action_space = train_env.action_space

    #############################
    # agent
    #############################

    _, env_info = train_env.reset()
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    #############################
    # train
    #############################

    logger = create_logger(cfg)

    # load model if given
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    save_path_resolved = cfg.save_path.replace("TIMESTAMP", datetime.now().strftime("%m%d-%H%M%S"))
    save_path_base = script_dir + "/" + save_path_resolved
    if cfg.agent_load_path is not None:
        load_path = os.path.join(script_dir, cfg.agent_load_path)
        agent.load(load_path)
    if cfg.buffer_load_path is not None:
        load_path = os.path.join(script_dir, cfg.buffer_load_path)
        agent.load_replay_buffer(load_path)

    # initial evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
    video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
    logger.update_metric(**eval_info)
    logger.update_metric(**video_info)
    logger.log_metric(step=0)
    logger.reset()

    # start training
    observations, env_infos = train_env.reset()
    actions: Optional[Tensor] = None
    transition: Optional[dict[str, Tensor]] = None
    update_counter = 0
    update_info = {}

    current_epoch = -1
    max_epochs = 5000

    base_env = getattr(train_env, "env", getattr(train_env, "base_env", train_env))
    target_obj = getattr(base_env, "object", None)

    print("\n" + "=" * 60)
    print("[PRE-FLIGHT CHECK] Verifying Curriculum State:")
    print(f"  -> Object Actuated:       {getattr(target_obj, 'actuated', False)}")
    print(f"  -> use_curriculum Flag:   {getattr(base_env, 'use_curriculum', False)}")
    print(f"  -> Curriculum Controller: {type(getattr(base_env, 'curriculum', None))}")
    print("=" * 60 + "\n")

    if not getattr(base_env, "use_curriculum", False) is True:
        print("\n\n[FATAL] Training aborted at Step 0: base_env.use_curriculum is FALSE!\n")
    if getattr(base_env, "curriculum", None) is None:
        print("\n\n[FATAL] Training aborted at Step 0: base_env.curriculum is None!\n")
    import wandb

    # Initialize a persistent WandB table for transition milestones
    curriculum_history_table = wandb.Table(
        columns=[
            "Step",
            "Epoch",
            "kp",
            "kv",
            "fr",
            "kp_lower",
            "kv_lower",
            "fr_lower",
            "Task Rew",
            "Ep Len",
        ]
    )
    last_logged_gains = None
    printed = False

    for interaction_step in tqdm.tqdm(range(1, int(cfg.num_interaction_steps + 1)), smoothing=0.1, mininterval=0.5):
        # 1. Update Dexmachina Curriculum
        if hasattr(base_env, "set_curriculum") and getattr(base_env, "use_curriculum", False) is True:
            epoch = int((interaction_step * max_epochs) / cfg.num_interaction_steps)
            if epoch > current_epoch:
                res = base_env.set_curriculum(epoch)
                if epoch <= 2:
                    print(f"[CURRICULUM ACTIVE] Advanced to Epoch {epoch} (reset_tracker={res})")
                current_epoch = epoch
        elif not printed:
            print("Tried to set curriculum, curriculum not implemented.")
            printed=True
        # using env steps simplifies the comparison with the performance reported in the paper.
        env_step = interaction_step * cfg.num_train_envs

        # collect data. use random actions until agent.can_start_training()
        if agent.can_start_training() and transition is not None:
            actions = agent.sample_actions(interaction_step, prev_transition=transition, training=True)
        else:
            #actions = train_env.action_space.sample()
            # Ensure random actions for all parallel envs.
            actions = np.array([train_env.action_space.sample() for _ in range(cfg.num_train_envs)])

        assert actions is not None
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=observations.device, dtype=torch.float32)

        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(actions)

        # Log online tracking and success rate metrics
        metric_payload = {}
        for k in ["eval/success_2cm", "eval/success_5cm", "env/pos_dist", "env/rot_dist", "env/arti_dist",
                  "env/keypoint_dist", "env/wrist_pdist", "env/contact_force"]:
            if k in env_infos:
                metric_payload[k] = env_infos[k]

        if metric_payload:
            logger.update_metric(**metric_payload)

        if "episode_info" in env_infos:
            logger.update_metric(**env_infos["episode_info"])

        if isinstance(next_observations, torch.Tensor):
            next_buffer_observations = next_observations.clone()
        else:
            next_buffer_observations = next_observations.copy()
        for env_idx in range(cfg.num_train_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_obs"][env_idx]

        if "episode_info" in env_infos:
            logger.update_metric(**env_infos["episode_info"])

        transition = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_buffer_observations,
        }
        agent.process_transition(transition)
        transition["next_observation"] = next_observations
        observations = next_observations

        if agent.can_start_training():
            # update network
            # updates_per_interaction_step can be below 1.0
            update_counter += cfg.updates_per_interaction_step
            while update_counter >= 1:
                update_info = agent.update()
                logger.update_metric(**update_info)
                update_counter -= 1

            # evaluation
            if cfg.evaluation_per_interaction_step and interaction_step % cfg.evaluation_per_interaction_step == 0:
                eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
                logger.update_metric(**eval_info)

            # metrics
            if cfg.metrics_per_interaction_step and interaction_step % cfg.metrics_per_interaction_step == 0:
                metrics_info = agent.get_metrics()
                logger.update_metric(**metrics_info)

            # video recording
            if cfg.recording_per_interaction_step and interaction_step % cfg.recording_per_interaction_step == 0:
                video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
                logger.update_metric(**video_info)

            # logging
            if cfg.logging_per_interaction_step and interaction_step % cfg.logging_per_interaction_step == 0:
                curriculum_logs = {}

                if hasattr(base_env, "curriculum") and base_env.curriculum is not None:
                    curr = base_env.curriculum

                    # 1. Populate current gain parameters (Scalar plots)
                    curr_gains = curr.get_current_gains()
                    raw_gains = {}
                    for k, v in curr_gains.items():
                        val = float(v.mean().item() if hasattr(v, "mean") else v)
                        curriculum_logs[f"env/{k}"] = val
                        raw_gains[k] = val

                    # 2. Populate pre-registered curriculum progress metrics
                    curriculum_logs["env/curriculum_epoch"] = epoch
                    curr_step = getattr(
                        curr,
                        "curriculum_step",
                        getattr(curr, "decay_step", getattr(curr, "step", None)),
                    )
                    if curr_step is not None:
                        curriculum_logs["env/curriculum_step"] = float(curr_step)

                    # 3. Reward gradients
                    if hasattr(curr, "task_grad"):
                        curriculum_logs["env/grad/task"] = float(curr.task_grad)
                    elif hasattr(curr, "grads") and isinstance(curr.grads, dict):
                        for g_name in ["task", "bc", "con", "imi"]:
                            if g_name in curr.grads:
                                curriculum_logs[f"env/grad/{g_name}"] = float(curr.grads[g_name])

                    # 4. Gating thresholds
                    rolling_task_rew = (
                        float(np.mean(curr.task_rew_deque))
                        if hasattr(curr, "task_rew_deque") and len(curr.task_rew_deque) > 0
                        else 0.0
                    )
                    rolling_ep_len = (
                        float(np.mean(curr.ep_lens_deque))
                        if hasattr(curr, "ep_lens_deque") and len(curr.ep_lens_deque) > 0
                        else 0.0
                    )

                    if hasattr(curr, "task_rew_deque") and len(curr.task_rew_deque) > 0:
                        curriculum_logs["env/curriculum_rolling_task_rew"] = rolling_task_rew
                    if hasattr(curr, "ep_lens_deque") and len(curr.ep_lens_deque) > 0:
                        curriculum_logs["env/curriculum_rolling_ep_len"] = rolling_ep_len

                    # 5. Check for Curriculum Update Event
                    if last_logged_gains is None:
                        last_logged_gains = raw_gains.copy()
                    elif raw_gains != last_logged_gains:
                        # Terminal Table Output
                        print("\n" + "=" * 62)
                        print(
                            f" [CURRICULUM UPDATE DETECTED] Epoch {epoch} | Step"
                            f" {interaction_step}"
                        )
                        print("=" * 62)
                        print(f"| {'Parameter':<18} | {'Previous':<18} | {'New Value':<18} |")
                        print(f"|{'-' * 20}|{'-' * 20}|{'-' * 20}|")
                        for k in raw_gains:
                            prev_v = f"{last_logged_gains.get(k, 0.0):.4f}"
                            new_v = f"{raw_gains[k]:.4f}"
                            print(f"| {k:<18} | {prev_v:<18} | {new_v:<18} |")
                        print("=" * 62 + "\n")

                        # WandB Table Logging
                        curriculum_history_table.add_data(
                            interaction_step,
                            epoch,
                            raw_gains.get("kp", 0.0),
                            raw_gains.get("kv", 0.0),
                            raw_gains.get("fr", 0.0),
                            raw_gains.get("kp_lower", 0.0),
                            raw_gains.get("kv_lower", 0.0),
                            raw_gains.get("fr_lower", 0.0),
                            rolling_task_rew,
                            rolling_ep_len,
                        )
                        wandb.log(
                            {"curriculum/update_table": curriculum_history_table}, step=env_step
                        )

                        last_logged_gains = raw_gains.copy()

                logger.update_metric(**curriculum_logs)
                logger.log_metric(step=env_step)
                logger.reset()
            # checkpointing
            if (
                cfg.save_checkpoint_per_interaction_step
                and interaction_step % cfg.save_checkpoint_per_interaction_step == 0
            ):
                save_path = os.path.join(save_path_base, f"step{interaction_step}")
                agent.save(save_path)

            # save buffer
            if cfg.save_buffer_per_interaction_step and interaction_step % cfg.save_buffer_per_interaction_step == 0:
                save_path = os.path.join(save_path_base, f"step{interaction_step}")
                agent.save_replay_buffer(save_path)

    # final evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
    video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
    logger.update_metric(**eval_info)
    logger.update_metric(**video_info)
    logger.log_metric(step=env_step)
    logger.reset()

    train_env.close()
    eval_env.close()
    record_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()
    run(args)