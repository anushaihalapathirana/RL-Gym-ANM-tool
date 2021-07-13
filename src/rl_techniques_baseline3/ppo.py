
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def create_model_PPO(test_env):
    # parallel environments
    env = make_vec_env(lambda: test_env, n_envs=4)
    model = PPO("MlpPolicy", env, gamma=0.995, verbose=0)
    model.learn(total_timesteps=10000, log_interval=4)
    # save the model
    model.save("ppo_gym_anm_model")


