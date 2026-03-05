# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of a custom skrl PPO_RNN agent.

This is a project-local version of the ChargeProject custom PPO workflow, adapted for SpiderBotAIProject.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of a custom skrl PPO_RNN agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch"], help="The ML framework used by skrl.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="RL algorithm.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug_vis",
    action="store_true",
    default=False,
    help="Enable command debug visualization markers (spawn points + waypoints).",
)
parser.add_argument("--debug_plot", action="store_true", default=False, help="Show BEV + wall debug plots via matplotlib.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. " f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    raise SystemExit(1)

from skrl.agents.torch.ppo import PPO_RNN
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import SpiderBotAIProject.tasks  # noqa: F401
from SpiderBotAIProject.tasks.manager_based.spiderbot_ai.agents.skrl_custom_ppo_model import SharedRecurrentModel

# config shortcuts
algorithm = args_cli.algorithm.lower()
if args_cli.agent is None:
    agent_cfg_entry_point = (
        "skrl_custom_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_custom_{algorithm}_cfg_entry_point"
    )
else:
    agent_cfg_entry_point = args_cli.agent


def _init_debug_subplots(registry, artists_out):
    """Build subplots from registered debug plots (called once). Returns the new figure."""
    import matplotlib.pyplot as plt

    plots = registry.plots
    n = len(plots)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    import numpy as np

    axes = np.array(axes).flatten() if n > 1 else np.array([axes])
    for idx, (name, entry) in enumerate(plots.items()):
        ax = axes[idx]
        ax.set_title(name)
        if entry[0] == "image":
            data = entry[1].detach().cpu().numpy()
            im = ax.imshow(data, origin="lower", cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.046)
            artists_out[name] = ("image", im)
        elif entry[0] == "scatter":
            sc = ax.scatter([], [], c=[], s=1, cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.set_aspect("equal")
            artists_out[name] = ("scatter", sc, ax)
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    fig.tight_layout()
    return fig


def _update_debug_plots(fig, registry, artists):
    """Update existing debug plot artists with new data."""
    import numpy as np

    for name, entry in registry.plots.items():
        if name not in artists:
            continue
        art = artists[name]
        if art[0] == "image":
            data = entry[1].detach().cpu().numpy()
            art[1].set_data(data)
            art[1].set_clim(data.min(), data.max())
        elif art[0] == "scatter":
            xy = entry[1].detach().cpu().numpy()
            c = entry[2].detach().cpu().numpy()
            art[1].set_offsets(np.column_stack([xy[:, 0], xy[:, 1]]))
            art[1].set_array(c)
            art[2].set_xlim(xy[:, 0].min() - 0.5, xy[:, 0].max() + 0.5)
            art[2].set_ylim(xy[:, 1].min() - 0.5, xy[:, 1].max() + 0.5)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with custom skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    if args_cli.debug_vis:
        env_cfg.commands.waypoint.debug_vis = True

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # checkpoint path
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"])
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env_cfg.log_dir = log_dir
    from SpiderBotAIProject.tasks.manager_based.spiderbot_ai.custom_terrain_gen import ensure_custom_terrain_usd

    ensure_custom_terrain_usd(
        size_x=float(env_cfg.height_map_size_x),
        size_y=float(env_cfg.height_map_size_y),
        meter_per_grid=float(env_cfg.height_map_meter_per_grid),
        seed=int(env_cfg.seed or 42),
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    device = env.device

    models = {}
    models["policy"] = SharedRecurrentModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        num_envs=env.num_envs,
        init_log_std=experiment_cfg["model"]["log_std_init"],
    )
    models["value"] = models["policy"]

    cfg = experiment_cfg["agent"].copy()

    scheduler_cfg = cfg.get("learning_rate_scheduler")
    if isinstance(scheduler_cfg, str):
        scheduler_map = {"KLAdaptiveLR": KLAdaptiveLR}
        try:
            cfg["learning_rate_scheduler"] = scheduler_map[scheduler_cfg]
        except KeyError as exc:
            raise ValueError(f"Unknown learning_rate_scheduler '{scheduler_cfg}'.") from exc

    shaper_scale = cfg.get("rewards_shaper_scale", 1.0)
    cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * shaper_scale

    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=env.num_envs, device=device)

    agent = PPO_RNN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    agent.load(resume_path)
    agent.set_running_mode("eval")
    agent.init()

    obs, _ = env.reset()
    timestep = 0

    # -- debug plot setup --------------------------------------------------
    debug_fig = None
    debug_artists = {}
    _debug_ready = False
    if args_cli.debug_plot:
        import matplotlib.pyplot as plt

        plt.ion()
        base_env = env.unwrapped
        base_env.debug_plot.enabled = True

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)

            if agent._rnn:
                agent._rnn_initial_states["policy"] = agent._rnn_final_states["policy"]
                if agent.policy is not agent.value:
                    agent._rnn_initial_states["value"] = agent._rnn_final_states["value"]

                finished = (terminated | truncated).nonzero(as_tuple=False)
                if finished.numel():
                    for s in agent._rnn_initial_states["policy"]:
                        s[:, finished[:, 0]] = 0
                    if agent.policy is not agent.value:
                        for s in agent._rnn_initial_states["value"]:
                            s[:, finished[:, 0]] = 0

        # -- debug plot update ------------------------------------------
        if args_cli.debug_plot:
            if not _debug_ready and len(base_env.debug_plot.plots) > 0:
                _debug_ready = True
                debug_fig = _init_debug_subplots(base_env.debug_plot, debug_artists)
            if _debug_ready:
                _update_debug_plots(debug_fig, base_env.debug_plot, debug_artists)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
