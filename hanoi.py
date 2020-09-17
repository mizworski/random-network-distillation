"""Simple Hanoi Towers gym.
Based on: https://github.com/RobertTLange/gym-hanoi
"""
from copy import deepcopy

import gym
from gym import spaces
import numpy as np

from hanoi_helpers import HanoiHelper
from toy_mr import HashableNdarray

class Hanoi(gym.Env):
    """Hanoi tower adhere to Open AI gym template"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_disks=None, reward_for_solved=1.,
                 reward_for_invalid_action=0):
        self.n_disks = n_disks
        self._reward_for_solved = reward_for_solved
        self._reward_for_invalid_action = reward_for_invalid_action
        self.action_space = spaces.Discrete(6)
        self.observation_space = \
            spaces.Box(low=0,
                       high=1,
                       shape=(1, 1, self.n_disks * 3),
                       dtype=np.uint8,
                       )
        # print(self.observation_space)
        self._current_state = None
        self.goal_state = self.n_disks * (2,)

        self.done = None
        self.action_lookup = {0 : '(0,1) - top disk of peg 0 to top of peg 1',
                              1 : '(0,2) - top disk of peg 0 to top of peg 2',
                              2 : '(1,0) - top disk of peg 1 to top of peg 0',
                              3 : '(1,2) - top disk of peg 1 to top of peg 2',
                              4 : '(2,0) - top disk of peg 2 to top of peg 0',
                              5 : '(2,1) - top disk of peg 2 to top of peg 1'}

        self._action_to_move = {0: (0, 1), 1: (0, 2), 2: (1, 0),
                          3: (1, 2), 4: (2, 0), 5: (2, 1)}

        self.helper = HanoiHelper(self.n_disks)
        self._all_states = self.helper.generate_all_states()
        self._all_transitions = self.helper.generate_all_transitions()
        self._visited_states_in_episode = set()
        self._visited_states_in_history = set()

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode has finished. '
                               'Call env.reset() to start a new episode.')

        info = {'invalid_action': False}

        move = self._action_to_move[action]

        if self.move_allowed(move):
            disk_to_move = min(self.disks_on_peg(move[0]))
            moved_state = list(self._current_state)
            moved_state[disk_to_move] = move[1]
            self._current_state = tuple(moved_state)
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
        # HACK - make observation an image for RND code
        state = np.expand_dims(self.vectorized_obs(), 0)
        state = np.expand_dims(state, 0)

        info.update(self.compute_metrics())

        return state, reward, self.done, info

    def clone_state(self):
        return HashableNdarray(np.array(self._current_state, dtype=np.uint8))

    def restore_state(self, state):
        self._current_state = tuple(state.array)
        self.done = self._current_state == self.goal_state

    def obs2tuple(self, obs):
        return tuple(np.reshape(obs, (self.n_disks, 3)).argmax(axis=1))

    @staticmethod
    def state2obs(state):
        return state.array

    @staticmethod
    def obs2state(observation, copy=True):
        if copy:
            observation = deepcopy(observation)
        return HashableNdarray(observation)


    def vectorized_obs(self):
        return np.eye(3)[np.array(self._current_state)].flatten()

    def disks_on_peg(self, peg):
        """
        * Inputs:
            - peg: pole to check how many/which disks are in it
        * Outputs:
            - list of disk numbers that are allocated on pole
        """
        return [disk for disk in range(self.n_disks) if
                self._current_state[disk] == peg]

    def move_allowed(self, move):
        """
        * Inputs:
            - move: tuple of state transition (see ACTION_LOOKUP)
        * Outputs:
            - boolean indicating whether action is allowed from state!
        move[0] - peg from which we want to move disc
        move[1] - peg we want to move disc to
        Allowed if:
            * discs_to is empty (no disc of peg) set to true
            * Smallest disc on target pole larger than smallest on prev
        """
        disks_from = self.disks_on_peg(move[0])
        disks_to = self.disks_on_peg(move[1])

        if disks_from:
            return (min(disks_to) > min(disks_from)) if disks_to else True
        else:
            return False

    def reset(self):
        self._current_state = self.n_disks * (0,)
        self._visited_states_in_episode = set()
        self.done = False
        # HACK - make observation an image for RND code
        state = np.expand_dims(self.vectorized_obs(), 0)
        state = np.expand_dims(state, 0)
        return state

    def render(self, mode='human'):
        for peg in range(3):
            print(f'peg {peg}: {self.disks_on_peg(peg)} ', end='')
        print('')

    def compute_visit_freq_table(self, episodes):
        """Computes visit frequency for each state and averages
        over episodes."""
        visited_states = {state: 0 for state in self._all_states}
        visited_sets_episodes = []
        for episode in episodes:
            visited_set = set()
            if episode.solved:
                visited_set.add(self.goal_state)
            states_batch = [self.obs2tuple(obs) for obs in
                            episode.transition_batch.observation]
            for state in states_batch:
                visited_states[state] += 1
                visited_set.add(state)
                self._visited_states_in_history.add(state)
                self._visited_states_in_episode.add(state)
            visited_sets_episodes.append(visited_set)
            if episode.solved:
                visited_states[self.goal_state] += 1

        visited_freq = np.mean([len(x) / 3 ** self.n_disks
                                for x in visited_sets_episodes])
        visited_states = {state: visited_states[state] /
                                 (len(episodes) * 3 ** self.n_disks)
                          for state in self._all_states}
        return visited_states, visited_freq

    def compute_metrics(self):
        """Computes environment related metrics."""
        metrics = {}
        metrics['visited_states_in_episode'] = len(self._visited_states_in_episode) / 3 ** (self.n_disks)
        metrics['visited_states_in_history'] = \
            len(self._visited_states_in_history) / 3 ** (self.n_disks)
        return metrics

    def log_visit_heat_map(self, epoch, episodes, log_detailed_heat_map,
                               metric_logging):

        del log_detailed_heat_map
        visited_states, _ = self.compute_visit_freq_table(episodes)
        heat_map = self.helper.render_heatmap(visited_states)
        metric_logging.log_image(
            f'episode_model/visit_heat_map',
            epoch, heat_map
        )
