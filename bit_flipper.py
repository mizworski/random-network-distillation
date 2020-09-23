"""Simple Bit flipper gym.
Based on: https://github.com/RobertTLange/gym-hanoi
"""
from copy import deepcopy

import gym
import numpy as np
from gym import spaces

from graph_distance_logging import GraphDistanceLogger
from toy_mr import HashableNdarray


class BitFlipper(gym.Env):
    """Bit flipper env adhere to Open AI gym template"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_bits=None, reward_for_solved=1.,
                 reward_for_invalid_action=0):
        self.n_bits = n_bits
        self._reward_for_solved = reward_for_solved
        self._reward_for_invalid_action = reward_for_invalid_action
        self.action_space = spaces.Discrete(self.n_bits)
        self.observation_space = \
            spaces.Box(low=0,
                       high=1,
                       shape=(1, 1, self.n_bits),
                       dtype=np.uint8
                       )

        self._current_state = None
        self.goal_state = self.n_bits * (1,)
        self.done = None

        self._visited_states_in_episode = set()
        self._visited_states_in_history = set()

        self.reset()
        self.graph_distance = GraphDistanceLogger(self)

    def step(self, action):
        def flip_bit(bit_i):
            self._current_state = tuple(self._current_state[i] if i != bit_i
                                        else (self._current_state[i] + 1) % 2
                                        for i in range(self.n_bits))

        if self.done:
            raise RuntimeError('Episode has finished. '
                               'Call env.reset() to start a new episode.')

        info = {'invalid_action': False}
        if self.move_allowed(action):
            flip_bit(action)
        else:
            info['invalid_action'] = True

        if self._current_state == self.goal_state:
            reward = self._reward_for_solved
            info['solved'] = True
            self.done = True
        elif info['invalid_action']:
            reward = self._reward_for_invalid_action
        else:
            reward = 0

        obs = np.expand_dims(self.vectorized_obs(), 0)
        obs = np.expand_dims(obs, 0)

        self._visited_states_in_episode.add(self.obs2state(obs))
        self._visited_states_in_history.add(self.obs2state(obs))
        info.update({
            'visited_states_in_episode': len(self._visited_states_in_episode),
            'visited_states_in_history': len(self._visited_states_in_history),
        })

        if hasattr(self, 'graph_distance'):
            self.graph_distance.update_distances(self.obs2state(
                np.array(self.obs2tuple(self.vectorized_obs())))
            )
            info.update(self.graph_distance.result())

        return obs, reward, self.done, info

    def move_allowed(self, action):
        if self._current_state[action] == 1:
            return True
        elif np.all(self._current_state[:action]):
            return True
        else:
            return False

    def clone_state(self):
        return HashableNdarray(np.array(self._current_state, dtype=np.uint8))

    def restore_state(self, state):
        self._current_state = tuple(state.array)
        self.done = self._current_state == self.goal_state

    def obs2tuple(self, obs):
        return tuple(obs)

    @staticmethod
    def state2obs(state):
        return state.array

    @staticmethod
    def obs2state(observation, copy=True):
        if copy:
            observation = deepcopy(observation)
        return HashableNdarray(observation)

    def vectorized_obs(self):
        return np.array(self._current_state)

    def reset(self):
        self._current_state = self.n_bits * (0,)
        self.done = False

        obs = np.expand_dims(self.vectorized_obs(), 0)
        obs = np.expand_dims(obs, 0)
        return obs

    def reset_history(self):
        self._visited_states_in_episode = set()
        self._visited_states_in_history = set()

    def render(self, mode='human'):
        print(self._current_state)
