from mp_pytorch.basis_gn import ProDMPBasisGenerator
from mp_pytorch.mp import ProDMP
from mp_pytorch.phase_gn import ExpDecayPhaseGenerator

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
from omni.isaac.lab.envs import (
    ManagerBasedRLEnvCfg,
)
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry


import omni.isaac.lab_tasks  # noqa: F401
import alr_isaaclab_tasks.tasks  # noqa: F401

import mprl.util as util
import fancy_gym
import gymnasium as gym


def get_mp(**kwargs):
    assert kwargs["type"] == "prodmp"
    mp_args = kwargs["args"]
    dtype, device = util.parse_dtype_device(mp_args["dtype"], mp_args["device"])
    phase_gn = ExpDecayPhaseGenerator(
        tau=mp_args["tau"],
        delay=mp_args.get("delay", 0.0),
        alpha_phase=mp_args["alpha_phase"],
        learn_tau=mp_args.get("learn_tau", False),
        learn_delay=mp_args.get("learn_delay", False),
        learn_alpha_phase=mp_args.get("learn_alpha_phase", False),
        dtype=dtype,
        device=device,
    )
    basis_gn = ProDMPBasisGenerator(
        phase_generator=phase_gn,
        num_basis=mp_args["num_basis"],
        basis_bandwidth_factor=mp_args["basis_bandwidth_factor"],
        num_basis_outside=mp_args.get("num_basis_outside", 0),
        dt=mp_args["dt"],
        alpha=mp_args["alpha"],
        pre_compute_length_factor=5,
        dtype=dtype,
        device=device,
    )
    prodmp = ProDMP(
        basis_gn=basis_gn,
        num_dof=mp_args["num_dof"],
        auto_scale_basis=mp_args.get("auto_scale_basis", True),
        weights_scale=mp_args.get("weights_scale", 1),
        goal_scale=mp_args.get("goal_scale", 1),
        disable_weights=mp_args.get("disable_weights", False),
        disable_goal=mp_args.get("disable_goal", False),
        relative_goal=mp_args.get("relative_goal", False),
        dtype=dtype,
        device=device,
    )
    return prodmp


def get_override_mp_config(mp_args: dict):
    """
    Customize the fancy gym black box env by overriding the default MP HPs

    Args:
        mp_args: customized MP hyperparameters from experiment config files

    Returns:
        config: customized MP config used to override the default MP config in
        fancy gym
    """
    config = {
        "phase_generator_kwargs": {},
        "basis_generator_kwargs": {},
        "trajectory_generator_kwargs": {},
    }

    # Phase generator kwargs
    if "tau" in mp_args:
        config["phase_generator_kwargs"]["tau"] = mp_args["tau"]
    if "delay" in mp_args:
        config["phase_generator_kwargs"]["delay"] = mp_args["delay"]
    if "learn_tau" in mp_args:
        config["phase_generator_kwargs"]["learn_tau"] = mp_args["learn_tau"]
    if "learn_delay" in mp_args:
        config["phase_generator_kwargs"]["learn_delay"] = mp_args["learn_delay"]
    if "alpha_phase" in mp_args:
        config["phase_generator_kwargs"]["alpha_phase"] = mp_args["alpha_phase"]

    # Basis generator kwargs
    if "num_basis" in mp_args:
        config["basis_generator_kwargs"]["num_basis"] = mp_args["num_basis"]
    if "basis_bandwidth_factor" in mp_args:
        config["basis_generator_kwargs"]["basis_bandwidth_factor"] = mp_args[
            "basis_bandwidth_factor"
        ]
    if "num_basis_outside" in mp_args:
        config["basis_generator_kwargs"]["num_basis_outside"] = mp_args[
            "num_basis_outside"
        ]
    if "alpha" in mp_args:
        config["phase_generator_kwargs"]["alpha"] = mp_args["alpha"]

    # Trajectory generator kwargs
    if "disable_goal" in mp_args:
        config["trajectory_generator_kwargs"]["disable_goal"] = mp_args["disable_goal"]

    if "relative_goal" in mp_args:
        config["trajectory_generator_kwargs"]["relative_goal"] = mp_args[
            "relative_goal"
        ]

    if "auto_scale_basis" in mp_args:
        config["trajectory_generator_kwargs"]["auto_scale_basis"] = mp_args[
            "auto_scale_basis"
        ]

    if "weights_scale" in mp_args:
        config["trajectory_generator_kwargs"]["weights_scale"] = mp_args[
            "weights_scale"
        ]

    if "goal_scale" in mp_args:
        config["trajectory_generator_kwargs"]["goal_scale"] = mp_args["goal_scale"]

    # Black box kwargs
    if "verbose_level" in mp_args:
        config["black_box_kwargs"]["verbose_level"] = mp_args["verbose_level"]

    return config


def make_env(env_id: str, seed: int, rank: int, render: bool, **kwargs):
    """
    Get a function instance for creating a black box environment

    Args:
        env_id: task id in fancy gym
        seed: start seed idx for the environment group
        rank: seed idx in the internal env group
        render: visualize or not
        **kwargs: keyword arguments

    Returns:
        a function instance for creating a black box environment, work for both
        standard gymnasium task and metaworld task

    """

    def _get_env():
        util.set_logger_level("ERROR")
        env = gym.make(id=env_id, render_mode="human" if render else None, **kwargs)
        env.reset(seed=seed + rank)

        return env

    return _get_env


def make_bb_vec_env(
    env_id: str, num_env: int, seed: int, render: bool, mp_args: dict, **kwargs
):
    """
    Create a vectorized black box environment with customized MP hyperparameters
    Args:
        env_id: env id from fancy gym
        num_env: number of environments
        seed: random seed
        render: render mode
        mp_args: customized MP hyperparameters from experiment config files
        **kwargs: keyword arguments

    Returns:
        A vector of black box environments using the gymnasiium library
    """

    if render:
        assert num_env == 1, "Rendering only works with num_env=1"

    # Generate customized MP wrapper
    mp_config_override = get_override_mp_config(mp_args)

    # Vectorized Class, using Stable Baselines3, as the Gymnasium's is buggy
    # Yet, the sb3 version will roll the gymnasium interface back to the old
    # gym one, e.g.:
    # Gymnasium: state, info = env.reset(seed=seed)
    # -> Gym:    state = env.reset()
    # Gymnasium: state, reward, terminated, truncated, info = env.step(action)
    # -> Gym:    state, reward, done, info = env.step(action)

    if "fancy" in env_id:

        env_fns = [
            make_env(
                env_id=env_id,
                seed=seed,
                rank=i,
                render=render,
                mp_config_override=mp_config_override,
                **kwargs,
            )
            for i in range(num_env)
        ]

        vec_env = SubprocVecEnv if num_env > 1 else DummyVecEnv

        env = vec_env(env_fns)

        return env

    elif "Isaac" in env_id:

        """Train with stable-baselines agent."""
        env_cfg = load_cfg_from_registry(env_id.split("/")[1], "env_cfg_entry_point")
        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = num_env

        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = seed
        env_cfg.sim.device = "cuda:0"

        # create isaac environment
        env = gym.make(env_id, cfg=env_cfg, render_mode=None)

        # wrap around environment for stable baselines
        env = Sb3VecEnvWrapper(env, True)

        return env

    else:
        raise Exception("env_id not matching one of the categories")
