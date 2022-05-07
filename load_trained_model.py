#%%
import yaml
import cloudpickle
import os
import copy
import ray

from utils import merge_dicts, import_policy_class


checkpoint="/root/workspace/model_data/ImpalaTrainer_2022-04-19_12-12-17/ImpalaTrainer_BreakoutNoFrameskip-v4_e6b7f_00000_0_2022-04-19_12-12-18/checkpoint_000200/checkpoint-200"
env=None
ray.init(num_gpus=0)

#%%
with open("evaluate.yml", "rb") as f:
    myconfig = yaml.safe_load(f)

myconfig.update({
    "num_gpus": 0,
    "num_gpus_per_worker": 0,
    "evaluation_num_workers": 0,
})

# Load configuration from checkpoint file.
config_path = ""
if checkpoint:
    checkpoint = os.path.expanduser(checkpoint)
    config_dir = os.path.dirname(checkpoint)
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
    raise ValueError(
        "Could not find params.pkl in either the checkpoint dir or "
        "its parent directory AND no `--config` given on command "
        "line!")
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
if not env:
    if not config.get("env"):
        raise ValueError("the following arguments are required: env")
    env = config.get("env")
# Make sure we have evaluation workers.
if not config.get("evaluation_num_workers"):
    config["evaluation_num_workers"] = config.get("num_workers", 0)
if not config.get("evaluation_duration"):
    config["evaluation_duration"] = 1
# Hard-override this as it raises a warning by Trainer otherwise.
# Makes no sense anyways, to have it set to None as we don't call
# `Trainer.train()` here.
config["evaluation_interval"] = 1


#%%
# Create the Trainer from config.
# cls = get_trainable_cls(args.run)
cls = import_policy_class("ref.impala.ImpalaTrainer")
trainer = cls(env=env, config=config)

# %%
worker=trainer.workers.local_worker()
policy=worker.get_policy()
model=policy.model
# %%
params=model.state_dict()
print(params.keys())
# %%
params2=policy.get_weights()
params2
# %%
import torch
import numpy as np
for k in params2.keys():
    print(torch.equal(params[k].cpu(),torch.from_numpy(params2[k])))
# %%
import pandas as pd

df=pd.read_csv("/root/workspace/model_data/ImpalaTrainer_2022-04-19_12-12-17/ImpalaTrainer_BreakoutNoFrameskip-v4_e6b7f_00000_0_2022-04-19_12-12-18/progress.csv")
# %%
df.columns
# %%
df.loc[199,"evaluation/episode_reward_mean"]
# %%
