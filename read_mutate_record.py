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
for it in [10,50,100,150]:
    data=load(f"eval_result_hard_20_15/mutate_result_{it}.pkl")

    baseline_result=data["baseline"]
    print(f"baseline: {baseline_result.summary()}")
    # print(baseline_result.hist_episode_length)
    # print(baseline_result.hist_episode_reward)
    # print("="*20)
    # for i in range(100):
    #     mutate_result=data[f"mutation {i}"]
    #     print(f"mutation {i}: {mutate_result.summary()}")
    baseline_result=data["baseline"]
    y_baseline=np.mean(baseline_result.hist_episode_reward)
    y_baseline_std=np.std(baseline_result.hist_episode_reward)

    y_mutate=[]
    y_mutate_std=[]

    for i in range(100):
        mutate_result=data[f"mutation {i}"]
        y_mutate.append(np.mean(mutate_result.hist_episode_reward))
        y_mutate_std.append(np.std(mutate_result.hist_episode_reward))
    y_mutate=np.array(y_mutate)
    y_mutate_std=np.array(y_mutate_std)

    plt.figure(figsize=(15,3))
    plt.plot()
    plt.plot([0,99],[y_baseline,y_baseline],label="baseline")
    plt.fill_between([0,99],[y_baseline+y_baseline_std]*2,[y_baseline-y_baseline_std]*2,alpha=0.3)
    plt.plot(np.arange(len(y_mutate)),y_mutate,label="mutate")
    plt.fill_between(np.arange(100),y_mutate+y_baseline_std,y_mutate-y_baseline_std,alpha=0.3)

    plt.legend()
    plt.show()
# %%
