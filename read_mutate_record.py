#%%
import pickle
from evaluate_with_mutate import Record
import matplotlib.pyplot as plt
import numpy as np

def load(path):
    with open(path,"rb") as f:
        obj=pickle.load(f)
    return obj


#%%
draw_mode="std"
# draw_mode="minmax"

def get_metrics(result:Record):
    return result.hist_episode_reward

# for it in range(100,901,200):
for it in [10,30,50,70]:
    print(f"iteration: {it}")
    baseline_data=None
    # baseline_data=load(f"eval_result/SpaceInvaders_01_25_without_conv_frac_01/mutate_result_{it}.pkl")
    # data=load(f"eval_result/Qbert_21_32_without_conv_frac_01_new/mutate_result_{it}.pkl")
    data=load(f"eval_result/BeamRider_18_05_without_conv_frac_01_new/mutate_result_{it}.pkl")

    if baseline_data:
        baseline_result=baseline_data["baseline"]
    else:
        baseline_result=data["baseline"]
    
    print(f"baseline: {baseline_result.summary()}")
    print(f"baseline sampled epsiodes: {len(baseline_result.hist_episode_reward)}")
    # print(baseline_result.hist_episode_length)
    # print(baseline_result.hist_episode_reward)
    # print("="*20)
    # for i in range(100):
    #     mutate_result=data[f"mutation {i}"]
    #     print(f"mutation {i}: {mutate_result.summary()}")
    # baseline_result=data["baseline"]
    y_baseline=np.mean(get_metrics(baseline_result))
    y_baseline_std=np.std(get_metrics(baseline_result))
    y_baseline_min=np.min(get_metrics(baseline_result))
    y_baseline_max=np.max(get_metrics(baseline_result))

    y_mutate=[]
    y_mutate_std=[]
    y_mutate_min=[]
    y_mutate_max=[]
    y_mutate_episodes=[]

    mutation_nums=len(data)
    if "baseline" in data:
        mutation_nums-=1

    for i in range(mutation_nums):
        mutate_result=data[f"mutation {i}"]
        y_mutate.append(np.mean(get_metrics(mutate_result)))
        y_mutate_std.append(np.std(get_metrics(mutate_result)))
        y_mutate_min.append(np.min(get_metrics(mutate_result)))
        y_mutate_max.append(np.max(get_metrics(mutate_result)))
        y_mutate_episodes.append(len(get_metrics(mutate_result)))
    y_mutate=np.array(y_mutate)
    y_mutate_std=np.array(y_mutate_std)
    y_mutate_min=np.array(y_mutate_min)
    y_mutate_max=np.array(y_mutate_max)
    y_mutate_episodes=np.array(y_mutate_episodes)


    print(f"mutate sampled episodes: {y_mutate_episodes.mean()}")

    plt.figure(figsize=(15,3),dpi=300)
    plt.plot()
    plt.plot([0,mutation_nums-1],[y_baseline,y_baseline],label="baseline")
    if draw_mode=="std":
        plt.fill_between([0,mutation_nums-1],[y_baseline+y_baseline_std]*2,[y_baseline-y_baseline_std]*2,alpha=0.3)
    elif draw_mode=="minmax":
        plt.fill_between([0,mutation_nums-1],[y_baseline_max]*2,[y_baseline_min]*2,alpha=0.3)
    plt.plot(np.arange(len(y_mutate)),y_mutate,label="mutate")
    if draw_mode=="std":
        plt.fill_between(np.arange(mutation_nums),y_mutate+y_mutate_std,y_mutate-y_mutate_std,alpha=0.3)
    elif draw_mode=="minmax":
        plt.fill_between(np.arange(mutation_nums),y_mutate_max,y_mutate_min,alpha=0.3)

    print("baseline",y_baseline_max,y_baseline_min, y_baseline)
    print("mutate",y_mutate_max.max(),y_mutate_min.min(),y_mutate.mean())

    plt.legend()
    plt.show()


# %%
def get_metrics(result:Record):
    return result.hist_episode_length
data=load(f"eval_result/BeamRider_18_05_without_conv_frac_01_new/mutate_result_50.pkl")
y_mutate=[]
y_mutate_std=[]
y_mutate_min=[]
y_mutate_max=[]
y_mutate_episodes=[]
mutation_nums=len(data)
if "baseline" in data:
    mutation_nums-=1
for i in range(mutation_nums):
    mutate_result=data[f"mutation {i}"]
    y_mutate.append(np.mean(get_metrics(mutate_result)))
    y_mutate_std.append(np.std(get_metrics(mutate_result)))
    y_mutate_min.append(np.min(get_metrics(mutate_result)))
    y_mutate_max.append(np.max(get_metrics(mutate_result)))
    y_mutate_episodes.append(len(get_metrics(mutate_result)))
y_mutate=np.array(y_mutate)
y_mutate_std=np.array(y_mutate_std)
y_mutate_min=np.array(y_mutate_min)
y_mutate_max=np.array(y_mutate_max)
y_mutate_episodes=np.array(y_mutate_episodes)
# %%
