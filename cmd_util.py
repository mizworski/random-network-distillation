"""
Helpers for scripts like run_atari.py.
"""

import os

from atari_wrappers import make_atari, wrap_deepmind, make_toy_mr
# TODO kc modif
# from gym.wrappers import FlattenDictWrapper
# from mpi4py import MPI
from baselines import logger
from monitor import Monitor
from vec_env import SubprocVecEnv
import gym




class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info


    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def reset_history(self):
        self.env.reset_history()

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, max_episode_steps=4500):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}

    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id, max_episode_steps=max_episode_steps)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            return wrap_deepmind(env, **wrapper_kwargs)

        return _thunk

    # set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def make_toy_mr_env(map_file, num_env, env_size=None, wrapper_kwargs=None, start_index=0, max_episode_steps=300):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}

    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            env = make_toy_mr(map_file, env_size=env_size, max_episode_steps=max_episode_steps)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            # return wrap_deepmind(env, **wrapper_kwargs)
            return env

        return _thunk

    # set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def make_hanoi_env(num_env, n_disks, wrapper_kwargs=None, start_index=0, max_episode_steps=300):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}

    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            env = make_hanoi(n_disks, max_episode_steps=max_episode_steps)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            # return wrap_deepmind(env, **wrapper_kwargs)
            return env

        return _thunk

    # set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def make_hanoi(n_disks, max_episode_steps=300):
    from hanoi import Hanoi
    from gym import wrappers
    env = Hanoi(n_disks)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    return parser
