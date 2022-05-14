#%%
import gym

env=gym.make("SpaceInvadersNoFrameskip-v4")
env.reset()
env.unwrapped
# %%
env.__class__.__mro__
# %%
env.unwrapped.ale.lives()
# %%
