import gym
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


    for i in range(10):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done = False
        state = None
        episode_reward = 0.0
        episode_discounted_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            episode_discounted_reward += reward * (env.gamma ** episode_length)
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_discounted_rewards.append(episode_discounted_reward)
    mean_reward = np.mean(episode_rewards)
    mean_discounted_reward = np.mean(episode_discounted_rewards)
    std_reward = np.std(episode_rewards)
    std_discounted_reward = np.std(episode_discounted_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_discounted_rewards
    return mean_reward, std_reward, mean_discounted_reward, std_discounted_reward




def run_test(env, model, model_name):
    episode_rewards, episode_lengths, episode_discounted_rewards = [], [], []
    for episode in range(10):
        done = False
        episode_reward = 0.0
        episode_discounted_reward = 0.0
        episode_length = 0
        obs = env.reset()

        # run microgrid for 10000 steps
        for step in range(1000):
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

# SAC model
create_model_SAC(env)
sac_model = load_model(SAC, 'sac_gym_anm_model')
run_test(env, sac_model, 'SAC')

# PPO model
create_model_PPO(env)
ppo_model = load_model(PPO,'ppo_gym_anm_model')
run_test(env, ppo_model, 'PPO')

# A2c model
create_model_A2C(env)
a2c_model = load_model(A2C,'a2c_gym_anm_model')
run_test(env, a2c_model, 'A2C')