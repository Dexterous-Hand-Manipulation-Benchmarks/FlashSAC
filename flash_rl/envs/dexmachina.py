from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, cast

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

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
# Retargeting the demos were made with; Hydra passes it from configs/env/dexmachina.yaml.
DEFAULT_RETARGET_NAME = "para"

# Size of the reported runs (NUM_ENV_STEPS and NUM_TRAIN_ENVS in scripts/flashsac/train.sh); runs of another size
# need their own values until eval reads them from the run's config.
REPORTED_ENV_STEPS = 50_000_896
REPORTED_NUM_TRAIN_ENVS = 1024
REPORTED_INTERACTION_STEPS = REPORTED_ENV_STEPS // REPORTED_NUM_TRAIN_ENVS
# The evaluation runs one env per kind, whatever the run it evaluates used.
EVAL_ENV_COUNT_OVERRIDES = ("env.num_train_envs=1", "env.num_eval_envs=1", "env.num_record_envs=1")

# Keys of dexmachina's reward config (envs/rewards.py) that the factory also maps onto dexmachina's args.
OBJ_POS_BETA_KEY = "obj_pos_beta"
OBJ_ROT_BETA_KEY = "obj_rot_beta"
OBJ_ARTI_BETA_KEY = "obj_arti_beta"
CONTACT_REW_WEIGHT_KEY = "contact_rew_weight"
IMI_REW_WEIGHT_KEY = "imi_rew_weight"
BC_REW_WEIGHT_KEY = "bc_rew_weight"
# Reward keys dexmachina's RewardModule reads with cfg.get (envs/rewards.py) but get_reward_cfg() leaves out.
EXTRA_REWARD_KEYS = frozenset({"bc_wrist_weight"})

# Names under which the wrapper reports TrackingErrors in its infos.
SUCCESS_2CM_METRIC = "eval/success_2cm"
SUCCESS_5CM_METRIC = "eval/success_5cm"
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


@dataclass(frozen=True, slots=True)
class AppliedEnvConfig:
    """What a built dexmachina env uses, to compare with what FlashSAC configured."""

    use_contact_reward: bool
    object_actuated: bool
    use_curriculum: bool
    zero_epoch: int | None
    reward_weights: Mapping[str, Any]
    use_imi_rew: bool
    use_bc_rew: bool
    hand: str

    @classmethod
    def read_from(cls, env: Any, reward_keys: Iterable[str]) -> "AppliedEnvConfig":
        """Reads the values from a built env; reward_keys are the overrides its RewardModule caches."""
        curriculum = env.curriculum
        return cls(
            use_contact_reward=env.use_contact_reward,
            object_actuated=env.object.actuated,
            use_curriculum=env.use_curriculum,
            zero_epoch=None if curriculum is None else curriculum.zero_epoch,
            reward_weights={key: getattr(env.reward_module, key) for key in reward_keys},
            use_imi_rew=env.reward_module.use_imi_rew,
            use_bc_rew=env.reward_module.use_bc_rew,
            hand=env.robots["left"].name,
        )

    @classmethod
    def expected(
        cls,
        config: DexmachinaEnvConfig,
        args: Any,
        reward_cfg: Mapping[str, Any],
        reward_keys: Iterable[str],
    ) -> "AppliedEnvConfig":
        """The values dexmachina should end up with, from the config, its parsed args and the final reward config."""
        return cls(
            use_contact_reward=reward_cfg[CONTACT_REW_WEIGHT_KEY] > 0,
            object_actuated=config.actuate_object,
            use_curriculum=config.actuate_object,
            zero_epoch=max(config.max_epochs - args.num_zero_epoch, 0) if config.actuate_object else None,
            reward_weights={key: reward_cfg[key] for key in reward_keys},
            use_imi_rew=reward_cfg[IMI_REW_WEIGHT_KEY] > 0,
            use_bc_rew=reward_cfg[BC_REW_WEIGHT_KEY] > 0,
            # dexmachina names a robot "<hand>_<side>" (get_default_robot_cfg in envs/robot.py).
            hand=f"{args.hand}_left",
        )

    def mismatches(self, applied: "AppliedEnvConfig") -> list[str]:
        """One "name: configured X, applied Y" line per differing value, reward weights per key."""
        pairs: list[tuple[str, object, object]] = [
            (field.name, getattr(self, field.name), getattr(applied, field.name))
            for field in fields(self)
            if field.name != "reward_weights"
        ]
        pairs += [
            (f"reward_weights.{key}", value, applied.reward_weights[key])
            for key, value in self.reward_weights.items()
        ]
        return [
            f"{name}: configured {configured}, applied {actual}"
            for name, configured, actual in pairs
            if configured != actual
        ]


def verify_applied_config(env: Any, expected: AppliedEnvConfig) -> None:
    """Raises if the built env does not use the values FlashSAC configured."""
    mismatches = expected.mismatches(AppliedEnvConfig.read_from(env, expected.reward_weights))
    if mismatches:
        raise ValueError("dexmachina env does not use the configured values: " + "; ".join(mismatches))


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
    """
    Standardized Dexmachina Gymnasium wrapper for FlashSAC with online GPU tracking metrics,
    NaN safeguards, zero-overhead success rate calculations, and dynamic attribute forwarding.
    """

    def __init__(self, env, rl_device="cuda:0"):
        super().__init__()
        self.env = env
        self.sim_device = rl_device
        self._clip_actions = getattr(self.env, "action_clip", 1.0)
        self.num_envs = getattr(self.env, "num_envs", 1)

        # Observation and action dimensions
        self.num_obs = self.env.obs_dim
        self.num_actions = self.env.action_dim
        self.max_episode_steps = getattr(self.env, "max_episode_length", 200)

        # Buffer to capture true pre-reset terminal observations
        self._terminal_obs = torch.zeros(
            (self.num_envs, self.num_obs), device=self.sim_device, dtype=torch.float32
        )

        # Intercept reset_idx to capture terminal observation before bodies/dofs reset
        orig_reset_idx = self.env.reset_idx

        def hooked_reset_idx(env_idxs, *args, **kwargs):
            if len(env_idxs) > 0:
                with torch.no_grad():
                    term_obs = self.env.get_observations()
                    term_policy = term_obs["policy"] if isinstance(term_obs, dict) else term_obs
                    if not isinstance(term_policy, torch.Tensor):
                        term_policy = torch.tensor(term_policy, device=self.sim_device, dtype=torch.float32)
                    else:
                        term_policy = term_policy.to(self.sim_device, dtype=torch.float32)
                    if torch.isnan(term_policy).any():
                        term_policy = torch.nan_to_num(term_policy, nan=0.0, posinf=0.0, neginf=0.0)
                    self._terminal_obs[env_idxs] = term_policy[env_idxs]
            return orig_reset_idx(env_idxs, *args, **kwargs)

        self.env.reset_idx = hooked_reset_idx

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-self._clip_actions, high=self._clip_actions, shape=(self.num_actions,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs_dict, extras = self.env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.sim_device, dtype=torch.float32)
        else:
            obs = obs.to(self.sim_device, dtype=torch.float32)

        if torch.isnan(obs).any():
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        self._terminal_obs.copy_(obs)

        return obs, extras

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

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.sim_device, dtype=torch.float32)
        if not isinstance(rew, torch.Tensor):
            rew = torch.tensor(rew, device=self.sim_device, dtype=torch.float32)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor(terminated, device=self.sim_device, dtype=torch.bool)
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, device=self.sim_device, dtype=torch.bool)

        # In-flight NaN safeguards
        if torch.isnan(obs).any():
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(rew).any():
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)

        infos = {}
        dones = terminated | truncated
        non_reset_mask = ~dones
        if non_reset_mask.any():
            self._terminal_obs[non_reset_mask] = obs[non_reset_mask]

        infos["final_obs"] = self._terminal_obs.clone()

        # --- High-Speed GPU Online Success & Tracking Error Metrics ---
        if hasattr(self.env, "rew_dict") and isinstance(self.env.rew_dict, dict):
            infos.update(TrackingErrors.from_rew_dict(self.env.rew_dict).log_means())

        # --- Explicitly Forward Gain and Stiffness Parameters for WandB ---
        for param_name in ["kp", "kv", "fr", "kp_lower", "kv_lower", "fr_lower"]:
            if hasattr(self.env, param_name):
                val = getattr(self.env, param_name)
                infos[f"env/{param_name}"] = val.mean().item() if isinstance(val, torch.Tensor) else float(val)

        if "log" in extras:
            infos["episode_info"] = extras["log"]

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
        """Pass-through helper to expose internal BaseEnv attributes seamlessly."""
        return getattr(self.env, name)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def make_dexmachina_env(num_envs, device, hydra_kwargs, seed, env_name=""):
    """Build a Dexmachina BaseEnv from FlashSAC's Hydra env kwargs."""
    from dexmachina.envs.base_env import BaseEnv
    from dexmachina.envs.constructors import get_all_env_cfg, get_common_argparser, parse_clip_string
    from dexmachina.envs.rewards import get_reward_cfg

    config = DexmachinaEnvConfig.from_hydra(hydra_kwargs, env_name)
    parser = get_common_argparser()
    args, _ = parser.parse_known_args([])
    if config.actuate_object and config.max_epochs <= args.num_zero_epoch:
        raise ValueError(
            f"max_epochs ({config.max_epochs}) must exceed num_zero_epoch ({args.num_zero_epoch}); "
            "otherwise dexmachina zeroes the curriculum from the first epoch"
        )
    args.num_envs = num_envs
    args.seed = seed

    if config.hand is not None:
        args.hand = config.hand
    args.actuate_object = config.actuate_object
    # The training loop's curriculum clock uses the same max_epochs, so zero_epoch aligns with the training length.
    args.max_epochs = config.max_epochs

    rewards = config.reward_overrides
    if rewards is not None:
        known_keys = set(cast(Mapping[str, Any], get_reward_cfg()))
        unknown_keys = sorted(set(rewards) - known_keys - EXTRA_REWARD_KEYS)
        if unknown_keys:
            raise ValueError(f"unknown reward_cfg keys: {unknown_keys}")
        # Keys that have an arg go through args: dexmachina derives env_cfg["use_contact_reward"] and the
        # demo/contact data loading from args (envs/constructors.py).
        for k, v in rewards.items():
            if hasattr(args, k):
                setattr(args, k, v)
        if OBJ_POS_BETA_KEY in rewards or OBJ_ROT_BETA_KEY in rewards or OBJ_ARTI_BETA_KEY in rewards:
            args.task_rew_betas = [
                rewards.get(OBJ_POS_BETA_KEY, args.task_rew_betas[0]),
                rewards.get(OBJ_ROT_BETA_KEY, args.task_rew_betas[1]),
                rewards.get(OBJ_ARTI_BETA_KEY, args.task_rew_betas[2]),
            ]

    args.retarget_name = config.retarget_name

    clip_str = config.clip
    if "box" in clip_str or "box" in str(env_name):
        clip_str = BOX_CLIP
    elif "mixer" in clip_str or "mixer" in str(env_name):
        clip_str = MIXER_CLIP
    else:
        parts = clip_str.split("-")
        if len(parts) == CLIP_RANGE_PARTS:
            clip_str = f"{clip_str}-{DEFAULT_SUBJECT_TAKE}"
        elif len(parts) != CLIP_ID_PARTS:
            clip_str = BOX_CLIP

    args.clip = clip_str
    parse_clip_string(clip_str)

    env_kwargs = get_all_env_cfg(args, device=device)
    if rewards is not None:
        # Reward keys without an arg (e.g. force_penalty, task_rew_weight) reach the RewardModule only here.
        env_kwargs["reward_cfg"].update(rewards)
    env_kwargs['env_cfg']['use_rl_games'] = True
    env = BaseEnv(**env_kwargs)
    cached_reward_keys = [] if rewards is None else [k for k in rewards if hasattr(env.reward_module, k)]
    final_reward_cfg = cast(Mapping[str, Any], env_kwargs["reward_cfg"])
    expected = AppliedEnvConfig.expected(config, args, final_reward_cfg, cached_reward_keys)
    verify_applied_config(env, expected)
    return env


# Backwards Compatibility Aliases
DexmachinaWrapper = GenesisFlashSACWrapper
DexmachinaEnv = GenesisFlashSACWrapper
DexMachinaEnv = GenesisFlashSACWrapper
DexMachinaWrapper = GenesisFlashSACWrapper
GenesisWrapper = GenesisFlashSACWrapper
