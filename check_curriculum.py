import os
import sys

# 1. Force top-level initialization of Genesis to prevent circular import crashes
import genesis as gs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import hydra
import torch
from omegaconf import OmegaConf
from flash_rl.envs import create_envs

OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)

# 2. Compose minimal config using the exact same config dir as train.py
config_dir = os.path.join(SCRIPT_DIR, "configs")
hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
cfg = hydra.compose(config_name="flashSAC_base", overrides=[
    "env=dexmachina",
    "env.hand=allegro_hand",
    "env.clip=box-30-230",
    "env.retarget_name=para",
    "num_train_envs=1",
    "++env.curriculum=true",
])
OmegaConf.resolve(cfg)

# Scale down environment to 1 instance for instant 3-second startup
if hasattr(cfg, "env"):
    cfg.env.num_train_envs = 1
    cfg.env.num_eval_envs = 1
    cfg.env.num_record_envs = 1

print("\n" + "=" * 60)
print("[DIAGNOSTIC] Initializing 1 test environment...")
train_env, _, _ = create_envs(**cfg.env)
base_env = getattr(train_env, "env", getattr(train_env, "base_env", getattr(train_env, "unwrapped", train_env)))

# 3. Gather factual state from the environment
target_obj = getattr(base_env, "object", None)
n_objects = getattr(base_env, "n_objects", 0)
is_actuated = getattr(target_obj, "actuated", False) if target_obj is not None else False
use_curriculum = getattr(base_env, "use_curriculum", False)
curr_instance = getattr(base_env, "curriculum", None)
curr_cfg = getattr(base_env, "curr_cfg", {})

print("\n[DIAGNOSTIC] Environment Inspection Results:")
print(f"  -> base_env.n_objects:          {n_objects}")
print(f"  -> target_obj exists:           {target_obj is not None}")
print(f"  -> target_obj.actuated:         {is_actuated}")
print(f"  -> base_env.use_curriculum:     {use_curriculum}")
print(f"  -> base_env.curriculum pointer: {type(curr_instance)}")
print(f"  -> curr_cfg type / keys:        {list(curr_cfg.keys()) if isinstance(curr_cfg, dict) else type(curr_cfg)}")

if hasattr(target_obj, "kp") and hasattr(target_obj, "kv"):
    kp_val = target_obj.kp
    kv_val = target_obj.kv
    kp_num = kp_val.mean().item() if hasattr(kp_val, "mean") else float(kp_val)
    kv_num = kv_val.mean().item() if hasattr(kv_val, "mean") else float(kv_val)
    print(f"  -> Initial Object Gains:        kp={kp_num}, kv={kv_num}")

# 4. Assert preconditions
assert use_curriculum is True, (
    f"\n\n[ASSERTION FAILED] base_env.use_curriculum is FALSE!\n"
    f"Reason: n_objects={n_objects}, target_obj.actuated={is_actuated}.\n"
    f"BaseEnv requires (n_objects == 1 and target_obj.actuated) to activate Curriculum."
)
assert curr_instance is not None, "[ASSERTION FAILED] base_env.curriculum is None!"

# 5. Unit test: Manually mutate curriculum and verify propagation
print("\n[DIAGNOSTIC] Testing Gain Propagation...")
test_gain = 42.0
base_env.curriculum.curr_gains["kp"] = test_gain
base_env.curriculum.curr_gains_lower["kp"] = test_gain
base_env.curriculum.reset_object_gains()

live_gains = base_env.curriculum.get_current_gains()
print(f"  -> Live curriculum gains: {live_gains}")

assert live_gains["kp"] == test_gain, f"Expected {test_gain}, got {live_gains['kp']}"
print("\n[VERIFIED] Curriculum changes successfully applied and readable!")
print("=" * 60 + "\n")
train_env.close()