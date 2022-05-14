#%%
import yaml
import cloudpickle
import os
import copy
import ray

from utils import merge_dicts, import_policy_class


checkpoint="/root/workspace/model_data/ImpalaTrainer_2022-05-08_01-25-34/ImpalaTrainer_SpaceInvadersNoFrameskip-v4-TimeLimit40000_b432f_00000_0_2022-05-08_01-25-35/checkpoint_000200/checkpoint-200"
env="SpaceInvadersNoFrameskip-v4"

import psutil
psutil_memory_in_bytes = psutil.virtual_memory().total
ray._private.utils.get_system_memory = lambda: psutil_memory_in_bytes
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
num_params=0
for name,param in model.state_dict().items():
    print(f"{name}: {param.shape} {param.numel()}")
    num_params+=param.numel()
print(f"params: {num_params}")
# %%
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

weights=trainer.get_weights()[DEFAULT_POLICY_ID]
# %%
from mutation import mutate_inplace
def deepcopy(weights:dict):
    new_weights={}
    for k,v in weights.items():
        new_weights[k]=v.copy()
    return new_weights

# import copy
# ori_weights=copy.deepcopy(weights)


new_weights=deepcopy(weights)
new_weights2=deepcopy(weights)

#%%
for name,param in new_weights.items():
    print(name,param.shape)
    param=0
    break


# %%
import numpy as np
for name in weights.keys():
    print(np.array_equal(new_weights[name],new_weights2[name]))
# %%


def is_equal(weights1,weights2):
    for name in weights1.keys():
        flag=np.array_equal(weights1[name],weights2[name])
        if not flag:
            print(f"{name}: {weights1[name].shape} not equal")
            return False
    
    return True