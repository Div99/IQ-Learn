import gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

from wrappers.atari_wrapper import ScaledFloatFrame, FrameStack, PyTorchFrame
from wrappers.normalize_action_wrapper import check_and_normalize_box_actions
import envs
import numpy as np

# Register all custom envs
envs.register_custom_envs()


def make_atari(env):
    env = AtariWrapper(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, 4)
    return env


def is_atari(env_name):
    return env_name in ['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']


def make_env(args, monitor=True):
    env = gym.make(args.env.name)

    if monitor:
        env = Monitor(env, "gym")

    if is_atari(args.env.name):
        env = make_atari(env)

    # Normalize box actions to [-1, 1]
    env = check_and_normalize_box_actions(env)
    return env
