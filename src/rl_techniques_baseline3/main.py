import gym
import os
import numpy as np
from stable_baselines3 import SAC, PPO, A2C
from testEnv import TestEnvironment
from sac import create_model_SAC
from ppo import create_model_PPO
from a2c import create_model_A2C

env = TestEnvironment()

def load_model(algorithm, model_name):
    model = algorithm.load(model_name)
    return model

def run_test(env, model, model_name):
    episode_rewards, episode_lengths, episode_discounted_rewards = [], [], []
    for episode in range(10):
        done = False
        episode_reward = 0.0
        episode_discounted_reward = 0.0
        episode_length = 0
        obs = env.reset()

        # run microgrid for 10000 steps
        for step in range(10000):
            action, new_states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += -reward
            episode_discounted_reward += -reward * (env.gamma ** episode_length)
            episode_length += 1
            if done:
                obs = env.reset()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_discounted_rewards.append(episode_discounted_reward)
        
    mean_reward = np.mean(episode_rewards)
    mean_discounted_reward = np.mean(episode_discounted_rewards)
    std_reward = np.std(episode_rewards)
    std_discounted_reward = np.std(episode_discounted_rewards)
    print("******  ", model_name, "  ******")
    print('mean cost is  %.2f' % mean_reward, 'std_cost %.3f' % std_reward)


isSACModelAvailable = os.path.isfile('sac_gym_anm_model.zip')
isPPOModelAvailable = os.path.isfile('ppo_gym_anm_model.zip')
isA2CModelAvailable = os.path.isfile('a2c_gym_anm_model.zip')

# SAC model
if(not isSACModelAvailable):
    print("SAC Model saved")
    create_model_SAC(env)

# sac_model = load_model(SAC, 'sac_gym_anm_model')
# run_test(env, sac_model, 'SAC')

# PPO model
if(not isPPOModelAvailable):
    create_model_PPO(env)
    print("PPO Model saved")

ppo_model = load_model(PPO,'ppo_gym_anm_model')
run_test(env, ppo_model, 'PPO')

# A2c model
if(not isA2CModelAvailable):
    create_model_A2C(env)
    print("A2C Model saved")

# a2c_model = load_model(A2C,'a2c_gym_anm_model')
# run_test(env, a2c_model, 'A2C')