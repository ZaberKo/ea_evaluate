#!/usr/bin/env python

import argparse
import collections
import copy
import pickle
import gym

from gym.wrappers import TimeLimit
from ray.rllib.env.wrappers.atari_wrappers import get_wrapper_by_cls
from ray.tune.registry import register_env


import json
import os
from pathlib import Path
import shelve
import numpy as np

import yaml
from utils import import_policy_class,deepcopy_np_weights

from tqdm import tqdm

import ray
import ray.cloudpickle as cloudpickle
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
from ray.rllib.utils.debug import summarize

from ray.rllib.evaluation import RolloutWorker

from mutation import mutate_inplace

from tqdm import trange

EXAMPLE_USAGE = """
Example usage via RLlib CLI:
    rllib evaluate /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example usage via executable:
    ./evaluate.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example usage w/o checkpoint (for testing purposes):
    ./evaluate.py --run PPO --env CartPole-v0 --episodes 500
"""

# Note: if you use any custom models or envs, register them here first, e.g.:
#
# from ray.rllib.examples.env.parametric_actions_cartpole import \
#     ParametricActionsCartPole
# from ray.rllib.examples.model.parametric_actions_model import \
#     ParametricActionsModel
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionsCartPole(10))


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        help="(Optional) checkpoint from which to roll out. "
        "If none given, will use an initial (untrained) Trainer.")

    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's `DQN` or `PPO`), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env",
        type=str,
        help="The environment specifier to use. This could be an openAI gym "
        "specifier (e.g. `CartPole-v0`) or a full class-path (e.g. "
        "`ray.rllib.examples.env.simple_corridor.SimpleCorridor`).")
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run ray in local mode for easier debugging.")
    parser.add_argument(
        "--episodes",
        default=1,
        type=int,
        help="Number of complete episodes to roll out. Rollout will also stop "
        "if `--steps` (timesteps) limit is reached first. A value of 0 means "
        "no limitation on the number of episodes run.")
    parser.add_argument(
        "--config",
        default="evaluate.yml",
        type=str,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Gets merged with loaded configuration from checkpoint file and "
        "`evaluation_config` settings therein.")
    parser.add_argument("--out", default="mutate_record.pkl", help="Output filename.")
    parser.add_argument("--mutate_nums",type=int,default=100)

    return parser


def run(args, parser):
    with open(args.config, "rb") as f:
        myconfig=yaml.safe_load(f)


    # Load configuration from checkpoint file.
    config_path = ""
    if args.checkpoint:
        args.checkpoint=os.path.expanduser(args.checkpoint)
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        # Try parent directory.
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

    # Load the config from pickled.
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = cloudpickle.load(f)
    # If no pkl file found, require command line `--config`.
    else:
        # If no config in given checkpoint -> Error.
        if args.checkpoint:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no `--config` given on command "
                "line!")

        # Use default config for given agent.
        _, config = get_trainer_class(args.run, return_config=True)

    # Make sure worker 0 has an Env.
    config["create_env_on_driver"] = True

    # Merge with `evaluation_config` (first try from command line, then from
    # pkl file).
    evaluation_config = copy.deepcopy(
        myconfig.get("evaluation_config", config.get(
            "evaluation_config", {})))
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings (if not already the same
    # anyways).
    config = merge_dicts(config, myconfig)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    # Make sure we have evaluation workers.
    if not config.get("evaluation_num_workers"):
        config["evaluation_num_workers"] = config.get("num_workers", 0)
    if not config.get("evaluation_duration"):
        config["evaluation_duration"] = 1
    # Hard-override this as it raises a warning by Trainer otherwise.
    # Makes no sense anyways, to have it set to None as we don't call
    # `Trainer.train()` here.
    config["evaluation_interval"] = 1

    # reset evaluation_duratio
    lives=gym.make(args.env).unwrapped.ale.lives()
    config["evaluation_duration"]=args.episodes*lives
    print(f'{args.env} has {lives} lives, set evaluation_duation={config["evaluation_duration"]}')

    # ======== limit atari max timesteps ===========


    env_name=args.env or config["env"]
    max_episode_steps=3600*5 # at most 5min

    def env_creator(env_config):
        env=gym.make(env_name,**env_config)
        timelimit_wrapper=get_wrapper_by_cls(env, TimeLimit)
        new_env=TimeLimit(timelimit_wrapper.env, max_episode_steps=max_episode_steps)
        return new_env

    new_env_name=env_name+f"-TimeLimit{max_episode_steps}"
    register_env(new_env_name,env_creator)

    config["env"]=new_env_name
    args.env=new_env_name

    # ======================================
    # bug fix for ray1.12.0
    import psutil
    psutil_memory_in_bytes = psutil.virtual_memory().total
    ray._private.utils.get_system_memory = lambda: psutil_memory_in_bytes
    # ======================================

    ray.init(local_mode=args.local_mode,num_gpus=0)
    

    # Create the Trainer from config.
    # cls = get_trainable_cls(args.run)
    cls=import_policy_class(args.run)
    trainer = cls(config=config)

    # Load state from checkpoint, if provided.
    if args.checkpoint:
        trainer.restore(args.checkpoint)

    num_episodes = args.episodes
    num_steps=None

    # note: deepcopy to avoid connection to real model weights
    modelweights_ori=deepcopy_np_weights(
        trainer.get_weights()[DEFAULT_POLICY_ID]
    )

    results={}
    # ======== baseline ============
    baseline_result=rollout(trainer,num_steps,num_episodes)

    results["baseline"]=baseline_result
    print(f"baseline: {baseline_result.summary()}")
    print(baseline_result.hist_episode_length)
    print(baseline_result.hist_episode_reward)
    print("="*20)

    # ======== mutation ===========
    for i in trange(args.mutate_nums):
        modelweights=deepcopy_np_weights(modelweights_ori)
        mutate_inplace(modelweights,weight_magnitude=1e7)
        trainer.set_weights({DEFAULT_POLICY_ID:modelweights})

        result=rollout(trainer, num_steps, num_episodes)
        results[f"mutation {i}"]=result
        print(f"mutation {i}: {result.summary()}")
        # print(result.hist_episode_length)
        # print(result.hist_episode_reward)
        print("="*20)

    trainer.stop()

    p=Path(args.out).expanduser()
    if not p.parent.exists():
        p.parent.mkdir(parents=True)
    with p.open(mode="wb") as f:
        pickle.dump(results,f)


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # If num_episodes is set, stop if limit reached.
    if num_episodes and episodes >= num_episodes:
        return False
    # If num_steps is set, stop if limit reached.
    elif num_steps and steps >= num_steps:
        return False
    # Otherwise, keep going.
    return True


def rollout(agent,
            num_steps,
            num_episodes=0,):
    policy_agent_mapping = default_policy_agent_mapping



    # Normal case: Agent was setup correctly with an evaluation WorkerSet,
    # which we will now use to rollout.
    if hasattr(agent, "evaluation_workers") and isinstance(
            agent.evaluation_workers, WorkerSet):
        steps = 0
        episodes = 0
        result=Record()

        while keep_going(steps, num_steps, episodes, num_episodes):
            eval_result = agent.evaluate()["evaluation"]
            # Increase timestep and episode counters.
            eps = agent.config["evaluation_duration"]
            episodes += eps
            steps += eps * eval_result["episode_len_mean"]
            # Print out results and continue.
            print("Episode #{}: reward: {}".format(
                episodes, eval_result["episode_reward_mean"]))
            print(summarize(eval_result))
            result.add(eval_result)
        return result


class Record:
    def __init__(self) -> None:
        self.hist_episode_length=[]
        self.hist_episode_reward=[]
    def add(self,eval_result):
        self.hist_episode_length+=eval_result["hist_stats"]["episode_lengths"]
        self.hist_episode_reward+=eval_result["hist_stats"]["episode_reward"]
    def summary(self):
        return f"episode_len_mean: {np.mean(self.hist_episode_length)}, episode_reward_mean: {np.mean(self.hist_episode_reward)}"

def main():
    parser = create_parser()
    args = parser.parse_args()

    run(args, parser)


if __name__ == "__main__":
    main()
