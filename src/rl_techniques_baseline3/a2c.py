import gym
from testEnv import TestEnvironment
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

def create_model_A2C(test_env):
    # Parallel environments
    env = make_vec_env(lambda: test_env, n_envs=4)

    model = A2C("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    model.save("a2c_gym_anm_model")
