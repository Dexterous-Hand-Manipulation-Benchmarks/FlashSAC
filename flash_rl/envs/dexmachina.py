from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from dexmachina.envs.spec import EnvSpec
    from dexmachina.envs.vec_env import DexMachinaVecEnv

IMAGE_NDIM = 3
RGB_CHANNELS = 3
RGBA_CHANNELS = 4
# Float frames whose maximum is at most this are [0, 1] images; the margin absorbs rounding above 1.0.
UNIT_RANGE_MAX = 1.01

# Clip ids are "<object>-<start>-<end>-<subject>-<take>"; a bare frame range gets subject 1, take 1.
CLIP_ID_PARTS = 5
CLIP_RANGE_PARTS = 3
DEFAULT_SUBJECT_TAKE = "s01-01"
BOX_CLIP = "box-30-230-s01-01"
MIXER_CLIP = "mixer-0-150-s01-01"
SHORT_CLIP_NAMES = {"box": BOX_CLIP, "mixer": MIXER_CLIP}
# Retargeting the demos were made with; Hydra passes it from configs/env/dexmachina.yaml.
DEFAULT_RETARGET_NAME = "para"

# Size of the reported runs (NUM_ENV_STEPS and NUM_TRAIN_ENVS in scripts/flashsac/train.sh); runs of another size
# need their own values until eval reads them from the run's config.
REPORTED_ENV_STEPS = 50_000_896
REPORTED_NUM_TRAIN_ENVS = 1024
REPORTED_INTERACTION_STEPS = REPORTED_ENV_STEPS // REPORTED_NUM_TRAIN_ENVS
# The evaluation runs one env per kind, whatever the run it evaluates used.
EVAL_ENV_COUNT_OVERRIDES = ("env.num_train_envs=1", "env.num_eval_envs=1", "env.num_record_envs=1")

# Reward keys dexmachina's RewardModule reads with cfg.get(key, 0.0) (envs/rewards.py) but get_reward_cfg() leaves out.
EXTRA_REWARD_KEYS = frozenset({"bc_wrist_weight"})

# Building the task's reward config does not depend on the device.
REWARD_CFG_DEVICE = "cpu"

RUN_CONFIG_FILENAME = "config.yaml"
# Read by scripts/flashsac/train.sh to log the evaluation into the training run.
WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"

# Names under which the wrapper reports TrackingErrors in its infos; train.py logs them for the training env, so they
# must not use the eval/ prefix of evaluate()'s own success rates.
SUCCESS_2CM_METRIC = "train/success_2cm"
SUCCESS_5CM_METRIC = "train/success_5cm"
POS_DIST_METRIC = "env/pos_dist"
ROT_DIST_METRIC = "env/rot_dist"
ARTI_DIST_METRIC = "env/arti_dist"
KEYPOINT_DIST_METRIC = "env/keypoint_dist"
TRACKING_METRICS = (
    SUCCESS_2CM_METRIC,
    SUCCESS_5CM_METRIC,
    POS_DIST_METRIC,
    ROT_DIST_METRIC,
    ARTI_DIST_METRIC,
    KEYPOINT_DIST_METRIC,
)


@dataclass(frozen=True, slots=True)
class TrackingErrors:
    """Per-env tracking errors and success flags of the last step; None where dexmachina reported none."""

    pos_dist: torch.Tensor | None
    rot_dist: torch.Tensor | None
    arti_dist: torch.Tensor | None
    success_2cm: torch.Tensor | None
    success_5cm: torch.Tensor | None
    keypoint_dist: torch.Tensor | None

    @classmethod
    def from_rew_dict(cls, rew_dict: Mapping[str, torch.Tensor]) -> "TrackingErrors":
        # The only place that knows the keys of dexmachina's rew_dict (envs/rewards.py).
        return cls(
            pos_dist=rew_dict.get("pos_dist"),
            rot_dist=rew_dict.get("rot_dist"),
            arti_dist=rew_dict.get("arti_dist"),
            success_2cm=rew_dict.get("success_2cm"),
            success_5cm=rew_dict.get("success_5cm"),
            keypoint_dist=rew_dict.get("keypoint_dist"),
        )

    def log_means(self) -> dict[str, float]:
        """Means over the envs under FlashSAC's metric names, leaving out what dexmachina did not report."""
        # Success is defined in dexmachina; we only report the flags it provides.
        success_flags = {SUCCESS_2CM_METRIC: self.success_2cm, SUCCESS_5CM_METRIC: self.success_5cm}
        distances = {
            POS_DIST_METRIC: self.pos_dist,
            ROT_DIST_METRIC: self.rot_dist,
            ARTI_DIST_METRIC: self.arti_dist,
            KEYPOINT_DIST_METRIC: self.keypoint_dist,
        }
        means = {name: flags.float().mean().item() for name, flags in success_flags.items() if flags is not None}
        means.update({name: dist.mean().item() for name, dist in distances.items() if dist is not None})
        return means


def compose_eval_config(
    config_dir: str, config_name: str, overrides: Iterable[str]
) -> tuple[DictConfig, int]:
    """Read-only config for evaluating a checkpoint, and the env count its run trained with (for env steps).

    Only the top-level ``num_train_envs`` sets that count: an ``env.num_train_envs=...`` override is replaced by
    the eval overrides.
    """
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(config_name=config_name, overrides=[*overrides, *EVAL_ENV_COUNT_OVERRIDES])
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    num_train_envs = int(cfg.num_train_envs)
    return cfg, REPORTED_NUM_TRAIN_ENVS if num_train_envs <= 1 else num_train_envs


@dataclass(frozen=True, slots=True)
class DexmachinaEnvConfig:
    """FlashSAC's dexmachina settings, read once from the Hydra env kwargs; hand None = dexmachina's default."""

    clip: str
    max_epochs: int
    hand: str | None
    retarget_name: str
    actuate_object: bool
    reward_overrides: Mapping[str, Any] | None

    @classmethod
    def from_hydra(cls, hydra_kwargs: Mapping[str, Any], env_name: str) -> "DexmachinaEnvConfig":
        reward_cfg = hydra_kwargs.get("reward_cfg")
        curriculum = hydra_kwargs.get("curriculum", True)
        if curriculum and not hydra_kwargs.get("actuate_object", True):
            raise ValueError(
                "curriculum is true but actuate_object is false; dexmachina only runs the curriculum on an "
                "actuated object"
            )
        return cls(
            # Hydra's env_name names the clip when the config sets none.
            clip=str(hydra_kwargs.get("clip", env_name)),
            max_epochs=int(hydra_kwargs["max_epochs"]),
            hand=hydra_kwargs.get("hand"),
            retarget_name=hydra_kwargs.get("retarget_name", DEFAULT_RETARGET_NAME),
            # dexmachina's default curriculum decays the actuated object's gains (kp, kv, force range;
            # constructors.py), so the object is actuated whenever the curriculum is on, unless the config says
            # otherwise.
            actuate_object=hydra_kwargs.get("actuate_object", curriculum),
            # Read-only, so the overrides cannot change after they were read.
            reward_overrides=None if reward_cfg is None else MappingProxyType(dict(reward_cfg)),
        )


def curriculum_epoch(interaction_step: int, num_interaction_steps: float, max_epochs: int) -> int:
    """Curriculum epoch of an interaction step when max_epochs are spread evenly over num_interaction_steps."""
    return int(interaction_step * max_epochs / num_interaction_steps)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Takes a channels-last image, drops an alpha channel and scales float frames to uint8."""
    if image.shape[-1] == RGBA_CHANNELS:
        image = image[..., :RGB_CHANNELS]
    if image.dtype != np.uint8:
        if image.max() <= UNIT_RANGE_MAX:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


class GenesisFlashSACWrapper(gym.Env):
    """Gymnasium wrapper around a DexMachinaVecEnv: adds the tracking metrics to the infos and forwards unknown
    attributes to the BaseEnv."""

    def __init__(self, vec_env, rl_device="cuda:0"):
        super().__init__()
        self.vec_env = vec_env
        # The BaseEnv under the vec env: train.py and eval_dexmachina.py read its attributes (curriculum, camera, demo).
        self.env = vec_env.env
        self.sim_device = rl_device
        self._clip_actions = getattr(self.env, "action_clip", 1.0)
        self.num_envs = getattr(self.env, "num_envs", 1)

        # Observation and action dimensions
        self.num_obs = self.env.obs_dim
        self.num_actions = self.env.action_dim
        self.max_episode_steps = getattr(self.env, "max_episode_length", 200)

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-self._clip_actions, high=self._clip_actions, shape=(self.num_actions,), dtype=np.float32
        )

    def _built_from(self) -> "tuple[EnvSpec, dict[str, Any]]":
        if self.vec_env.spec is None or self.vec_env.reward_overrides is None:
            raise ValueError(
                "The vec env was not built with DexMachinaVecEnv.from_spec, so it does not know its EnvSpec. "
                "Build it with from_spec (make_dexmachina_env does) so that the run can save its spec."
            )
        return self.vec_env.spec, self.vec_env.reward_overrides

    @property
    def env_spec(self) -> "EnvSpec":
        """The EnvSpec the env was built from (gym.Env's own `spec` is a different thing)."""
        return self._built_from()[0]

    @property
    def reward_overrides(self) -> dict[str, Any]:
        return self._built_from()[1]

    def reset(self, seed=None, options=None):
        return self.vec_env.reset()

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.sim_device)
        else:
            actions = actions.to(self.sim_device)

        if actions.ndim == 1:
            batch_size = actions.numel() // self.num_actions
            actions = actions.view(batch_size, self.num_actions)

        if actions.shape[0] == 1 and self.num_envs > 1:
            actions = actions.repeat(self.num_envs, 1)

        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        obs, rew, terminated, truncated, vec_info = self.vec_env.step(actions)

        # dexmachina only replaces the rewards of NaN envs (base_env.py:734-736); kept from the previous wrapper.
        if torch.isnan(rew).any():
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)

        infos = {"final_obs": vec_info["final_obs"]}

        if hasattr(self.env, "rew_dict") and isinstance(self.env.rew_dict, dict):
            infos.update(TrackingErrors.from_rew_dict(self.env.rew_dict).log_means())

        for param_name in ["kp", "kv", "fr", "kp_lower", "kv_lower", "fr_lower"]:
            if hasattr(self.env, param_name):
                val = getattr(self.env, param_name)
                infos[f"env/{param_name}"] = val.mean().item() if isinstance(val, torch.Tensor) else float(val)

        infos["episode_info"] = vec_info["log"]

        return obs, rew, terminated, truncated, infos

    def render(self):
        camera = getattr(self.env, "_floating_camera", None)
        if camera is None:
            raise RuntimeError(
                "render() needs a camera, but the DexMachina env was built without one (its record_video option is "
                "off), so every video frame would be black. Set num_record_episodes=0 to train without videos."
            )
        try:
            render_out = camera.render()
        except Exception as e:
            raise RuntimeError(
                "Rendering a video frame with the DexMachina camera failed (cause above), so the video would be "
                "black or incomplete. Fix the cause, or set num_record_episodes=0 to train without videos."
            ) from e
        image = render_out[0] if isinstance(render_out, tuple) else render_out
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = to_uint8_rgb(image)
        if image.ndim == IMAGE_NDIM:
            image = image[np.newaxis, ...]
        return image

    def set_curriculum(self, epoch):
        if hasattr(self.env, "set_curriculum"):
            self.env.set_curriculum(epoch)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def close(self):
        self.vec_env.close()


def full_clip_id(clip: str) -> str:
    """The clip as "<object>-<start>-<end>-<subject>-<take>": short names and bare frame ranges are completed."""
    clip_str = SHORT_CLIP_NAMES.get(clip, clip)
    parts = clip_str.split("-")
    if len(parts) == CLIP_RANGE_PARTS:
        return f"{clip_str}-{DEFAULT_SUBJECT_TAKE}"
    if len(parts) != CLIP_ID_PARTS:
        raise ValueError(
            f"env clip {clip_str!r} is not a dexmachina clip id: it has {len(parts)} '-'-separated parts, "
            f"expected {CLIP_RANGE_PARTS} (object-start-end) or {CLIP_ID_PARTS} (object-start-end-subject-take), "
            f"or one of the short names {sorted(SHORT_CLIP_NAMES)}. Training on another clip than the one asked for "
            f"would go unnoticed. Fix env.clip, e.g. {BOX_CLIP!r}."
        )
    return clip_str


def dexmachina_spec(
    num_envs: int, hydra_kwargs: Mapping[str, Any], seed: int, env_name: str = ""
) -> "tuple[EnvSpec, dict[str, Any]]":
    """The shared benchmark task for this run, and the run's reward overrides (all equal to the task's values).

    A run chooses the hand, clip, seed and env count; a Hydra env or reward value that differs from the task raises.
    """
    # dexmachina is optional: FlashSAC runs other envs without it.
    from dexmachina.envs.benchmark_task import RunChoices, benchmark_spec  # noqa: PLC0415
    from dexmachina.envs.factory import build_env_kwargs  # noqa: PLC0415

    config = DexmachinaEnvConfig.from_hydra(hydra_kwargs, env_name)
    if config.hand is None:
        raise ValueError("env.hand is not set; the hand decides the task, so there is no default. Set env.hand.")

    spec = benchmark_spec(RunChoices(hand=config.hand, clip=full_clip_id(config.clip), seed=seed, num_envs=num_envs))

    requested: dict[str, Any] = {
        "actuate_object": config.actuate_object,
        "max_epochs": config.max_epochs,
        "retarget_name": config.retarget_name,
    }
    differing = {
        name: (value, getattr(spec, name)) for name, value in requested.items() if value != getattr(spec, name)
    }

    rewards = config.reward_overrides
    if rewards is not None:
        task_rewards = {
            **dict.fromkeys(EXTRA_REWARD_KEYS, 0.0),
            **build_env_kwargs(spec, REWARD_CFG_DEVICE, {})["reward_cfg"],
        }
        unknown_keys = sorted(set(rewards) - set(task_rewards))
        if unknown_keys:
            raise ValueError(f"unknown reward_cfg keys: {unknown_keys}")
        differing.update(
            {
                f"reward_cfg.{key}": (value, task_rewards[key])
                for key, value in rewards.items()
                if value != task_rewards[key]
            }
        )
    if differing:
        details = "; ".join(f"{name}: run {run!r}, task {task!r}" for name, (run, task) in differing.items())
        raise ValueError(
            f"the run sets env values that the benchmark task fixes ({details}). Methods are only comparable on "
            "the same task. Remove these overrides (env.<key> or env.reward_cfg.<key>); "
            "to change the task, change BENCHMARK_TASK in dexmachina."
        )

    return spec, {} if rewards is None else dict(rewards)


def make_dexmachina_env(
    num_envs: int, device: str, hydra_kwargs: Mapping[str, Any], seed: int, env_name: str = ""
) -> "DexMachinaVecEnv":
    """Build a DexMachinaVecEnv from FlashSAC's Hydra env kwargs."""
    from dexmachina.envs.vec_env import DexMachinaVecEnv  # noqa: PLC0415

    spec, reward_overrides = dexmachina_spec(num_envs, hydra_kwargs, seed, env_name)
    return DexMachinaVecEnv.from_spec(spec, device, reward_overrides)


def make_dexmachina_eval_env(train_env: GenesisFlashSACWrapper, device: str) -> GenesisFlashSACWrapper:
    """The training env's evaluation twin: its own scene, one unassisted env that never moves the curriculum clock."""
    from dexmachina.envs.factory import make_eval_env  # noqa: PLC0415
    from dexmachina.envs.vec_env import DexMachinaVecEnv, EnvInternals  # noqa: PLC0415

    # The eval env uses dexmachina's default reward, not the run's reward overrides: benchmark eval compares
    # success and ADD across methods.
    eval_env = make_eval_env(train_env.env_spec, device)
    return GenesisFlashSACWrapper(DexMachinaVecEnv(cast(EnvInternals, eval_env)), rl_device=device)


def refuse_resume_with_fixed_schedule(
    env: GenesisFlashSACWrapper, agent_load_path: str | None, buffer_load_path: str | None
) -> None:
    """Raises when a run loads a checkpoint or buffer under the fixed curriculum schedule."""
    if (agent_load_path or buffer_load_path) and env.env_spec.curr_schedule == "fixed":
        raise ValueError(
            "agent_load_path or buffer_load_path is set, but the curriculum schedule is 'fixed': its clock is the "
            "env's count of training steps, which a new run starts at 0, so the gains would restart at their start "
            "value while the loaded agent has trained further. Train from scratch (leave both paths null)."
        )


def check_env_step_budget(env: GenesisFlashSACWrapper, num_env_steps: int) -> None:
    """Raises unless the training loop's env steps reach the task's budget, to within one interaction step."""
    budget = env.env_spec.budget_env_steps
    if budget is None:
        raise ValueError("the env spec has no budget_env_steps; the benchmark task sets it, so build the spec from it.")
    num_envs = env.env_spec.num_envs
    trained = int(num_env_steps / num_envs) * num_envs
    if not budget <= trained < budget + num_envs:
        raise ValueError(
            f"num_env_steps {num_env_steps} trains {trained} env steps (the loop runs int(num_env_steps / "
            f"{num_envs}) interaction steps), but the task's budget is {budget}; a run must reach it within one "
            f"interaction step, so {budget}..{budget + num_envs - 1}. The fixed curriculum schedule is a fraction of "
            f"the budget, so a different length would end it early or never. Set num_env_steps to the smallest "
            f"multiple of {num_envs} that is at least {budget}."
        )


def check_curriculum_saw_the_trained_steps(env: GenesisFlashSACWrapper, num_interaction_steps: int) -> None:
    """Raises unless the env's curriculum clock counted exactly the env steps the training loop ran."""
    curriculum = env.env.curriculum
    trained = num_interaction_steps * env.env_spec.num_envs
    if curriculum.env_steps != trained:
        raise ValueError(
            f"the curriculum counted {curriculum.env_steps} env steps but the training loop ran {trained}. The fixed "
            "schedule is placed on the curriculum's own count, so it did not run as designed. Check that nothing "
            "else steps the training env and that the evaluation env does not share the training env's curriculum."
        )


def save_run_files(cfg: DictConfig, env: GenesisFlashSACWrapper, directory: Path) -> None:
    """Writes what evaluation needs to rebuild the run's env and agent: env_spec.json and the resolved config.yaml."""
    from dexmachina.envs.run_spec import save_run_spec  # noqa: PLC0415

    save_run_spec(directory, env.env_spec, env.reward_overrides)
    OmegaConf.save(cfg, directory / RUN_CONFIG_FILENAME, resolve=True)


def save_wandb_run_id(directory: Path, run_id: str) -> None:
    """Writes the W&B run id next to env_spec.json, so the evaluation of the run's checkpoints can log into that run."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / WANDB_RUN_ID_FILENAME).write_text(f"{run_id}\n")
