#%%
import ray
from ray.rllib.agents.a3c import A2CTrainer,DEFAULT_CONFIG
from ray.rllib.evaluation import SampleBatch

ray.init(include_dashboard=False,local_mode=False)
#%%
config=DEFAULT_CONFIG.copy()
config['framework']='torch'
config["rollout_fragment_length"]=200
config["num_workers"]=5
# config["create_env_on_driver"]=True
# config['_fake_gpus']=True

from custom_eval import sample
from ray.rllib.evaluation import RolloutWorker
RolloutWorker.sample=sample



trainer=A2CTrainer(config,env="BreakoutNoFrameskip-v4")


#$$


# %%
workers=trainer.workers
batches = ray.get(
    [
        w.sample.remote()
        for i, w in enumerate(workers.remote_workers())
    ])
# %%
