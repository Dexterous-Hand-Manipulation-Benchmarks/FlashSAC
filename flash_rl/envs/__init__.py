import multiprocessing as mp
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from gymnasium.wrappers import RescaleAction, TimeLimit

from ..types import NDArray


def extract_max_episode_steps(env: VectorEnv[NDArray, NDArray, NDArray]) -> int:
    if hasattr(env, "envs"):
        if isinstance(env.envs, (list, tuple)):
            return env.envs[0]._max_episode_steps  # type: ignore
        elif hasattr(env.envs, "_max_episode_steps"):
            return env.envs._max_episode_steps  # type: ignore
        elif hasattr(env.envs, "episode_length"):
            return env.envs.episode_length  # type: ignore
        elif hasattr(env.envs, "max_episode_length"):
            return env.envs.max_episode_length  # type: ignore
        else:
            raise ValueError(f"Unknown env type: {type(env)}")
    elif hasattr(env, "env_fns"):
        dummy_env = env.env_fns[0]()
        max_episode_steps = dummy_env._max_episode_steps
        del dummy_env
        return max_episode_steps  # type: ignore
    else:
        raise ValueError(f"Unknown env type: {type(env)}")


def create_envs(
    env_type: str,
    seed: int,
    env_name: str,
    num_train_envs: int,
    num_eval_envs: int,
    num_record_envs: int,
    rescale_action: bool,
    max_episode_steps: Optional[int],
    **kwargs: Any,
) -> tuple[
    VectorEnv[NDArray, NDArray, NDArray],
    VectorEnv[NDArray, NDArray, NDArray],
    VectorEnv[NDArray, NDArray, NDArray],
]:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    if env_type in ["dmc", "mujoco", "humanoid_bench", "metaworld", "myosuite", "d4rl"]:
        train_env = create_vec_env(
            env_type=env_type, env_name=env_name, seed=seed, num_envs=num_train_envs,
            rescale_action=rescale_action, max_episode_steps=max_episode_steps,
        )
        eval_env = create_vec_env(
            env_type=env_type, env_name=env_name, seed=seed, num_envs=num_eval_envs,
            rescale_action=rescale_action, max_episode_steps=max_episode_steps,
        )
        record_env = create_vec_env(
            env_type=env_type, env_name=env_name, seed=seed, num_envs=num_record_envs,
            rescale_action=rescale_action, max_episode_steps=max_episode_steps,
        )

    elif env_type == "isaaclab":
        from flash_rl.envs.isaaclab import make_isaaclab_env
        assert rescale_action is None, "Unused hyperparameter in IsaacLab."
        assert num_eval_envs is None, "Unused hyperparameter in IsaacLab."
        assert num_record_envs is None, "Unused hyperparameter in IsaacLab."
        train_env = make_isaaclab_env(env_name=env_name, num_envs=num_train_envs, seed=seed)
        eval_env = train_env
        record_env = train_env

    elif env_type == "maniskill":
        from flash_rl.envs.maniskill import make_maniskill_env
        assert rescale_action is None, "Unused hyperparameter in ManiSkill."
        assert max_episode_steps is None, "Unused hyperparameter in ManiSkill."
        train_env = make_maniskill_env(env_name=env_name, num_envs=num_train_envs)
        eval_env = make_maniskill_env(env_name=env_name, num_envs=num_eval_envs)
        record_env = make_maniskill_env(env_name=env_name, num_envs=num_record_envs)

    elif env_type == "mujoco_playground":
        from flash_rl.envs.mujoco_playground import make_mujoco_playground_env
        assert rescale_action is None, "Unused hyperparameter in Mujoco Playground."
        assert "use_domain_randomization" in kwargs
        assert "use_push_randomization" in kwargs
        train_env = make_mujoco_playground_env(
            env_name=env_name, seed=seed, num_envs=num_train_envs, max_episode_steps=max_episode_steps,
            use_domain_randomization=kwargs["use_domain_randomization"], use_push_randomization=kwargs["use_push_randomization"],
        )
        eval_env = make_mujoco_playground_env(
            env_name=env_name, seed=seed, num_envs=num_eval_envs, max_episode_steps=max_episode_steps,
            use_domain_randomization=False, use_push_randomization=False,
        )
        record_env = make_mujoco_playground_env(
            env_name=env_name, seed=seed, num_envs=num_record_envs, max_episode_steps=max_episode_steps,
            use_domain_randomization=False, use_push_randomization=False,
        )

    elif env_type == "genesis":
        from flash_rl.envs.genesis import make_genesis_env
        assert num_eval_envs is None, "Unused hyperparameter in Genesis."
        assert num_record_envs is None, "Unused hyperparameter in Genesis."
        train_env = make_genesis_env(
            env_name=env_name, num_envs=num_train_envs, rescale_action=rescale_action, eval_mode=False,
        )
        eval_env = train_env
        record_env = train_env

    elif env_type == "dexmachina":
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
        from envs.FlashSAC.dexmachina_wrapper import GenesisFlashSACWrapper

        import genesis as gs
        from dexmachina.envs.base_env import BaseEnv
        from dexmachina.envs.constructors import get_common_argparser, get_all_env_cfg, parse_clip_string

        gs.init(backend=gs.gpu, logging_level='warning')
        rl_device = kwargs.get("rl_device", "cuda:0")
        env_seed = kwargs.get("seed", 0)

        def make_dexmachina_env(num_envs, device, hydra_kwargs, seed):
            parser = get_common_argparser()
            args, _ = parser.parse_known_args([])
            args.num_envs = num_envs
            args.seed = seed

            if "hand" in hydra_kwargs:
                args.hand = hydra_kwargs["hand"]

            # Explicitly enforce retarget_name to match training (defaults to 'para')
            args.retarget_name = hydra_kwargs.get("retarget_name", kwargs.get("retarget_name", "para"))

            raw_clip = hydra_kwargs.get("clip", kwargs.get("clip", env_name))
            clip_str = str(raw_clip)

            if "box" in clip_str or "box" in str(env_name):
                clip_str = "box-30-230-s01-01"
            elif "mixer" in clip_str or "mixer" in str(env_name):
                clip_str = "mixer-0-150-s01-01"
            else:
                parts = clip_str.split("-")
                if len(parts) == 3:
                    clip_str = f"{clip_str}-s01-01"
                elif len(parts) != 5:
                    clip_str = "box-30-230-s01-01"

            args.clip = clip_str
            parse_clip_string(clip_str)

            env_kwargs = get_all_env_cfg(args, device=device)
            env_kwargs['env_cfg']['use_rl_games'] = True
            return BaseEnv(**env_kwargs)

        raw_train_env = make_dexmachina_env(num_train_envs, rl_device, kwargs, env_seed)
        train_env = GenesisFlashSACWrapper(raw_train_env, rl_device=rl_device)
        eval_env = train_env
        record_env = train_env

    else:
        raise NotImplementedError

    return train_env, eval_env, record_env


def create_vec_env(
    env_type: str,
    env_name: str,
    num_envs: int,
    seed: int,
    rescale_action: bool,
    max_episode_steps: Optional[int],
) -> VectorEnv[NDArray, NDArray, NDArray]:
    def make_one_env(
        env_type: str,
        env_name: str,
        seed: int,
        rescale_action: bool,
        max_episode_steps: Optional[int],
        **kwargs: Any,
    ) -> gym.Env[NDArray, NDArray]:
        if env_type == "mujoco":
            from flash_rl.envs.mujoco import make_mujoco_env
            env = make_mujoco_env(env_name, seed, **kwargs)
        elif env_type == "d4rl":
            from flash_rl.envs.d4rl import make_d4rl_env
            env = make_d4rl_env(env_name, seed)
        elif env_type == "dmc":
            from flash_rl.envs.dmc import make_dmc_env
            env = make_dmc_env(env_name, seed, **kwargs)
        elif env_type == "humanoid_bench":
            from flash_rl.envs.humanoid_bench import make_humanoid_env
            env = make_humanoid_env(env_name, seed, **kwargs)
        elif env_type == "myosuite":
            from flash_rl.envs.myosuite import make_myosuite_env
            env = make_myosuite_env(env_name, seed, **kwargs)
        elif env_type == "metaworld":
            from flash_rl.envs.metaworld import make_metaworld_env
            env = make_metaworld_env(env_name, seed, **kwargs)
        else:
            raise NotImplementedError

        if rescale_action:
            env = RescaleAction(env, np.float32(-1.0), np.float32(1.0))

        def extract_predefined_max_episode_steps(env: gym.Env[NDArray, NDArray]) -> Optional[int]:
            while True:
                if isinstance(env, TimeLimit):
                    return env._max_episode_steps
                if not hasattr(env, "env"):
                    return None
                env = env.env  # type: ignore

        predefined_max_episode_steps = extract_predefined_max_episode_steps(env)
        if max_episode_steps is None or max_episode_steps <= 0:
            if predefined_max_episode_steps is None:
                raise ValueError("`max_episode_steps` must be either defined in the environment or provided explicitly.")
            max_episode_steps = predefined_max_episode_steps

        env = TimeLimit(env, max_episode_steps)
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env

    env_fns = [
        (
            lambda i=i: make_one_env(
                env_type=env_type, env_name=env_name, seed=seed + i,
                rescale_action=rescale_action, max_episode_steps=max_episode_steps,
            )
        )
        for i in range(num_envs)
    ]
    if len(env_fns) > 1:
        envs = AsyncVectorEnv(env_fns, autoreset_mode="SameStep")
    else:
        envs = SyncVectorEnv(env_fns, autoreset_mode="SameStep")
    return envs


def create_dataset(env_type: str, env_name: str) -> list[dict[str, Any]]:
    if env_type == "d4rl":
        from flash_rl.envs.d4rl import make_d4rl_dataset
        dataset = make_d4rl_dataset(env_name)
    else:
        raise NotImplementedError
    return dataset


def get_normalized_score(env_type: str, env_name: str, unnormalized_score: float) -> float:
    if env_type == "d4rl":
        from flash_rl.envs.d4rl import get_d4rl_normalized_score
        score = get_d4rl_normalized_score(env_name, unnormalized_score)
    else:
        raise NotImplementedError
    return score