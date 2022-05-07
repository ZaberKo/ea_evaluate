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


from typing import Dict,Optional
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


def summary_result(res: Dict):
    from glom import glom
    train_info = glom(
        res, "info.learner.training.target_actor_0", default=None)
    timesteps_total = glom(train_info, "timesteps_total", default=None)
    timesteps_this_iter = glom(train_info, "timesteps_this_iter", default=None)

    learner_info = glom(
        train_info, "info.learner.default_policy.learner_stats")
    policy_entropy = glom(learner_info, "entropy")
    policy_loss = glom(learner_info, "policy_loss")
    vf_loss = glom(learner_info, "vf_loss")
    lr = glom(learner_info, "cur_lr")
    entropy_coeff = glom(learner_info, "entropy_coeff")

    evaluation = glom(res, "evaluation")
    # evaluation=glom(res,"evaluation.best_actor")
    episode_len_mean = glom(evaluation, "episode_len_mean", default=None)
    episode_reward_mean = glom(evaluation, "episode_reward_mean", default=None)
    episode_hist = glom(evaluation, "hist_stats", default=None)
    iteration = glom(res, "training_iteration")

    buffer_size = glom(res, "info.learner.replay_buffer_size")
    # ea_enable=glom(res,"EA") is None

    return {
        "timesteps_total": timesteps_total,
        "timesteps_this_iter": timesteps_this_iter,
        "policy_entropy": policy_entropy,
        "policy_loss": policy_loss,
        "vf_loss": vf_loss,
        "cur_lr": lr,
        "entropy_coeff": entropy_coeff,
        "episode_len_mean": episode_len_mean,
        "episode_reward_mean": episode_reward_mean,
        "episode_hist": episode_hist,
        # "ea_enable":ea_enable,
        "iteration": iteration,
        "buffer_size": buffer_size,
    }



