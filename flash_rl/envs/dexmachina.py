import torch
import numpy as np
import gymnasium as gym


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
            rew_dict = self.env.rew_dict
            pos_dist = rew_dict.get("pos_dist", None)
            rot_dist = rew_dict.get("rot_dist", None)
            arti_dist = rew_dict.get("arti_dist", None)

            if pos_dist is not None and rot_dist is not None:
                succ_2cm = (pos_dist < 0.02) & (rot_dist < 0.20)
                succ_5cm = (pos_dist < 0.05) & (rot_dist < 0.35)
                if arti_dist is not None:
                    succ_2cm = succ_2cm & (arti_dist < 0.10)
                    succ_5cm = succ_5cm & (arti_dist < 0.15)

                infos["eval/success_2cm"] = succ_2cm.float().mean().item()
                infos["eval/success_5cm"] = succ_5cm.float().mean().item()
                infos["env/pos_dist"] = pos_dist.mean().item()
                infos["env/rot_dist"] = rot_dist.mean().item()
                if arti_dist is not None:
                    infos["env/arti_dist"] = arti_dist.mean().item()

            if "keypoint_dist" in rew_dict:
                infos["env/keypoint_dist"] = rew_dict["keypoint_dist"].mean().item()
            if "wrist_pdist" in rew_dict:
                infos["env/wrist_pdist"] = rew_dict["wrist_pdist"].mean().item()
            if "contact_force" in rew_dict:
                infos["env/contact_force"] = rew_dict["contact_force"].mean().item()

        # --- Explicitly Forward Gain and Stiffness Parameters for WandB ---
        for param_name in ["kp", "kv", "fr", "kp_lower", "kv_lower", "fr_lower"]:
            if hasattr(self.env, param_name):
                val = getattr(self.env, param_name)
                infos[f"env/{param_name}"] = val.mean().item() if isinstance(val, torch.Tensor) else float(val)

        if "log" in extras:
            infos["episode_info"] = extras["log"]

        return obs, rew, terminated, truncated, infos

    def render(self):
        if hasattr(self.env, "_floating_camera") and self.env._floating_camera is not None:
            try:
                render_out = self.env._floating_camera.render()
                image = render_out[0] if isinstance(render_out, tuple) else render_out
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()
                if image.shape[-1] == 4:
                    image = image[..., :3]
                if image.dtype != np.uint8:
                    if image.max() <= 1.01:
                        image = image * 255.0
                    image = np.clip(image, 0, 255).astype(np.uint8)
                if image.ndim == 3:
                    image = image[np.newaxis, ...]
                return image
            except Exception:
                pass
        return np.zeros((1, 160, 160, 3), dtype=np.uint8)

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
    from dexmachina.envs.constructors import get_common_argparser, get_all_env_cfg, parse_clip_string

    kwargs = hydra_kwargs
    parser = get_common_argparser()
    args, _ = parser.parse_known_args([])
    args.num_envs = num_envs
    args.seed = seed

    if "hand" in hydra_kwargs:
        args.hand = hydra_kwargs["hand"]

    # Forward actuation/curriculum flags from Hydra kwargs to Dexmachina args
    args.actuate_object = hydra_kwargs.get(
        "actuate_object",
        hydra_kwargs.get("curriculum", True)
    )

    # Enforce max_epochs for curriculum schedule so zero_epoch aligns with training length
    args.max_epochs = int(hydra_kwargs.get("max_epochs", 10000))

    # Forward reward shaping configs if provided in hydra overrides
    reward_cfg_overrides = hydra_kwargs.get("reward_cfg", None)
    if reward_cfg_overrides is not None:
        rcfg_dict = dict(reward_cfg_overrides)
        for k, v in rcfg_dict.items():
            if hasattr(args, k):
                setattr(args, k, v)
        if "obj_pos_beta" in rcfg_dict or "obj_rot_beta" in rcfg_dict or "obj_arti_beta" in rcfg_dict:
            args.task_rew_betas = [
                rcfg_dict.get("obj_pos_beta", args.task_rew_betas[0]),
                rcfg_dict.get("obj_rot_beta", args.task_rew_betas[1]),
                rcfg_dict.get("obj_arti_beta", args.task_rew_betas[2]),
            ]

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
    if reward_cfg_overrides is not None:
        env_kwargs["reward_cfg"].update(dict(reward_cfg_overrides))
        if "contact_rew_weight" in rcfg_dict:
            env_kwargs["env_cfg"]["use_contact_reward"] = float(rcfg_dict["contact_rew_weight"]) > 0
    env_kwargs['env_cfg']['use_rl_games'] = True
    env = BaseEnv(**env_kwargs)
    if reward_cfg_overrides is not None and hasattr(env, "reward_module"):
        for k, v in dict(reward_cfg_overrides).items():
            env.reward_module.cfg[k] = v
            if hasattr(env.reward_module, k):
                setattr(env.reward_module, k, v)
    return env


# Backwards Compatibility Aliases
DexmachinaWrapper = GenesisFlashSACWrapper
DexmachinaEnv = GenesisFlashSACWrapper
DexMachinaEnv = GenesisFlashSACWrapper
DexMachinaWrapper = GenesisFlashSACWrapper
GenesisWrapper = GenesisFlashSACWrapper