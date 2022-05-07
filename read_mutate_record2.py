#%%
import pickle
from evaluate_with_mutate import Record

def load(path):
    with open(path,"rb") as f:
        obj=pickle.load(f)
    return obj



data=load("mutate_result_200_new_multi_seed_2333.pkl")


#%%
baseline_result=data["baseline"]
baseline_result2=data["baseline2"]
#%%
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,3),dpi=300)
plt.hist(baseline_result.hist_episode_reward,label="baseline1",alpha=0.5,bins=50)
plt.hist(baseline_result2.hist_episode_reward,label="baseline2",alpha=0.5,bins=50)
plt.legend()
plt.show()


# %%
plt.figure(figsize=(10,3),dpi=300)
plt.plot(baseline_result.hist_episode_reward,label="baseline1",alpha=0.8,lw=0.5)
plt.plot(baseline_result2.hist_episode_reward,label="baseline2",alpha=0.8,lw=0.5)
plt.legend()
plt.show()
#%%
plt.figure(figsize=(10,3),dpi=300)
plt.plot(baseline_result.hist_episode_length,label="baseline1",alpha=0.8,lw=0.5)
plt.plot(baseline_result2.hist_episode_length,label="baseline2",alpha=0.8,lw=0.5)
plt.legend()
plt.show()
# %%
print(f"baseline1: {np.mean(baseline_result.hist_episode_reward)}")
print(f"baseline2: {np.mean(baseline_result2.hist_episode_reward)}")
# %%
