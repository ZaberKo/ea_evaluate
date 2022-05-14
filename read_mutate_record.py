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
# for it in [10,50,100,150,200]:
for it in [100,200,300,400,500,600,700]:
    print(f"iteration: {it}")
    data=load(f"SpaceInvaders_01_25_without_conv_frac_01/mutate_result_{it}.pkl")
    baseline_data=load(f"SpaceInvaders_01_25_without_conv_frac_01/mutate_result_{it}_baseline.pkl")

    baseline_result=baseline_data["baseline"]
    
    print(f"baseline: {baseline_result.summary()}")
    print(f"baseline sampled epsiodes: {len(baseline_result.hist_episode_reward)}")
    # print(baseline_result.hist_episode_length)
    # print(baseline_result.hist_episode_reward)
    # print("="*20)
    # for i in range(100):
    #     mutate_result=data[f"mutation {i}"]
    #     print(f"mutation {i}: {mutate_result.summary()}")
    baseline_result=data["baseline"]
    y_baseline=np.mean(baseline_result.hist_episode_reward)
    y_baseline_std=np.std(baseline_result.hist_episode_reward)
    y_baseline_min=np.min(baseline_result.hist_episode_reward)
    y_baseline_max=np.max(baseline_result.hist_episode_reward)

    y_mutate=[]
    y_mutate_std=[]
    y_mutate_min=[]
    y_mutate_max=[]
    y_mutate_episodes=[]

    mutation_nums=len(data)-1

    for i in range(mutation_nums):
        mutate_result=data[f"mutation {i}"]
        y_mutate.append(np.mean(mutate_result.hist_episode_reward))
        y_mutate_std.append(np.std(mutate_result.hist_episode_reward))
        y_mutate_min.append(np.min(mutate_result.hist_episode_reward))
        y_mutate_max.append(np.max(mutate_result.hist_episode_reward))
        y_mutate_episodes.append(len(mutate_result.hist_episode_reward))
    y_mutate=np.array(y_mutate)
    y_mutate_std=np.array(y_mutate_std)
    y_mutate_min=np.array(y_mutate_min)
    y_mutate_max=np.array(y_mutate_max)
    y_mutate_episodes=np.array(y_mutate_episodes)


    print(f"mutate sampled episodes: {y_mutate_episodes.mean()} std:{y_mutate_episodes.std()} min:{y_mutate_episodes.min()} max:{y_mutate_episodes.max()}")

    assert len(y_mutate)==mutation_nums

    plt.figure(figsize=(15,3))
    plt.plot()
    plt.plot([0,mutation_nums-1],[y_baseline,y_baseline],label="baseline")
    # plt.fill_between([0,99],[y_baseline+y_baseline_std]*2,[y_baseline-y_baseline_std]*2,alpha=0.3)
    plt.fill_between([0,mutation_nums-1],[y_baseline_max]*2,[y_baseline_min]*2,alpha=0.3)
    plt.plot(np.arange(len(y_mutate)),y_mutate,label="mutate")
    # plt.fill_between(np.arange(100),y_mutate+y_baseline_std,y_mutate-y_baseline_std,alpha=0.3)
    plt.fill_between(np.arange(mutation_nums),y_mutate_max,y_mutate_min,alpha=0.3)

    plt.legend()
    plt.show()
# %%
for it in [100,200]:
    data=load(f"SpaceInvaders_01_25_without_conv/mutate_result_{it}.pkl")
    
    baseline_data=load(f"SpaceInvaders_01_25_without_conv/mutate_result_{it}_baseline.pkl")

    baseline_result=baseline_data["baseline"]
    print(f"sampled epsiodes: {len(baseline_result.hist_episode_length)}")
    print(f"baseline: {baseline_result.summary()}")
    # print(baseline_result.hist_episode_length)
    # print(baseline_result.hist_episode_reward)
    # print("="*20)
    # for i in range(100):
    #     mutate_result=data[f"mutation {i}"]
    #     print(f"mutation {i}: {mutate_result.summary()}")
    baseline_result=data["baseline"]
    y_baseline=np.mean(baseline_result.hist_episode_length)
    y_baseline_std=np.std(baseline_result.hist_episode_length)
    y_baseline_min=np.min(baseline_result.hist_episode_length)
    y_baseline_max=np.max(baseline_result.hist_episode_length)

    y_mutate=[]
    y_mutate_std=[]
    y_mutate_min=[]
    y_mutate_max=[]

    mutation_nums=len(data)-1

    for i in range(mutation_nums):
        mutate_result=data[f"mutation {i}"]
        y_mutate.append(np.mean(mutate_result.hist_episode_length))
        y_mutate_std.append(np.std(mutate_result.hist_episode_length))
        y_mutate_min.append(np.min(mutate_result.hist_episode_length))
        y_mutate_max.append(np.max(mutate_result.hist_episode_length))
    y_mutate=np.array(y_mutate)
    y_mutate_std=np.array(y_mutate_std)
    y_mutate_min=np.array(y_mutate_min)
    y_mutate_max=np.array(y_mutate_max)

    plt.figure(figsize=(15,3))
    plt.plot()
    plt.plot([0,mutation_nums-1],[y_baseline,y_baseline],label="baseline")
    # plt.fill_between([0,99],[y_baseline+y_baseline_std]*2,[y_baseline-y_baseline_std]*2,alpha=0.3)
    plt.fill_between([0,mutation_nums-1],[y_baseline_max]*2,[y_baseline_min]*2,alpha=0.3)
    plt.plot(np.arange(len(y_mutate)),y_mutate,label="mutate")
    # plt.fill_between(np.arange(100),y_mutate+y_baseline_std,y_mutate-y_baseline_std,alpha=0.3)
    plt.fill_between(np.arange(mutation_nums),y_mutate_max,y_mutate_min,alpha=0.3)

    plt.legend()
    plt.show()
# %%
