
import gym
import numpy as np
from stable_baselines3 import SAC
from testEnv import TestEnvironment

def create_model(env):
    # modify the action space - all actions will lie in [-1, 1]
    env.action_space = gym.spaces.Box(low=-1, high=1, shape=env.action_space.shape, dtype=np.float32)

    model = SAC("MlpPolicy", env, gamma=0.995, verbose=0)
    model.learn(total_timesteps=10000, log_interval=4)
    # save the model
    model.save("sac_gym_anm_model")

def load_model():
    model = SAC.load("sac_gym_anm_model")
    return model