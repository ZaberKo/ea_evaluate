from typing import List
import copy
import numpy as np
import ray
import tree
import importlib
from ray.rllib.utils import merge_dicts as _merge_dicts
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.framework import try_import_torch


from typing import Dict, Optional
from ray.rllib.utils.typing import TensorType, TensorStructType
from ray.rllib.policy import Policy
from ray.rllib.evaluation import SampleBatch
torch, _ = try_import_torch()


def merge_dicts(origin, new):
    new = copy.deepcopy(new)
    # origin is deep copied in _merge_dicts()
    return _merge_dicts(origin, new)


def ray_wait(pendings: List):
    '''
        wait all pendings without timeout
    '''
    return ray.wait(pendings, num_returns=len(pendings))[0]


def clone_numpy_weights(x: TensorStructType,):
    def mapping(item):
        if isinstance(item, np.ndarray):
            ret = item.copy()  # copy to avoid sharing (make it writeable)
        else:
            ret = item
        return ret

    return tree.map_structure(mapping, x)


def import_policy_class(policy_name) -> Policy:
    # read policy_class from config
    tmp = policy_name.rsplit('.', 1)
    if len(tmp) == 2:
        module, name = tmp
        return getattr(importlib.import_module(module), name)
    else:
        raise ValueError('`policy_name` is incorrect')



def deepcopy_np_weights(weights: dict):
    new_weights = {}
    for k, v in weights.items():
        new_weights[k] = v.copy()
    return new_weights

import gym
from gym.spaces import Box
from ray.rllib.env.wrappers.atari_wrappers import MonitorEnv



class MyMonitorEnv(MonitorEnv):
    def __init__(self, env=None):
        super().__init__(env)
        self._total_steps=0
        self._done=True

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self._current_reward = 0
        self._num_steps = 0

        # handle horizon case
        if not self._done:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1
            self._done=True


        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1

        self._done=done

        if done:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1

            # self._total_steps = sum(self._episode_lengths)


        return (obs, rew, done, info)