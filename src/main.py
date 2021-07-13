import gym
import numpy as np
from stable_baselines3 import SAC
from testEnv import TestEnvironment
from sac import create_model, load_model
from ppo import create_model_PPO, load_model_PPO

env = TestEnvironment()

def run_test(env, model):
    best_score = env.reward_range[0]
    cost_history = []
    number_of_steps = 10000
    cost = 0
    obs = env.reset()
    # run microgrid for 10000 steps
    for step in range(number_of_steps):
        action, new_states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    
        cost += reward

    cost_history.append(cost)
    avg_cost = np.mean(cost_history[-100:])

    print('cost is  %.2f' % -cost, 'trailing 100 games avg %.3f' % avg_cost)

# SAC model
# create_model(env)
# sac_model = load_model()
# run_test(env, sac_model)

# PPO model
# create_model_PPO(env)
ppo_model = load_model_PPO()
run_test(env, ppo_model)