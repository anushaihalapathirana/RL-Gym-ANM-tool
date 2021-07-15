import numpy as np
import os, sys
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from agent import Agent
from testEnv import TestEnvironment


if __name__ == '__main__':
    if len(sys.argv) == 2 and str(sys.argv[1]).lower() == 'default':
        print("environment is default gym_anm ANM6Easy-v0")
        env = gym.make('gym_anm:ANM6Easy-v0')
    else: 
        env = TestEnvironment()
    
    agent = Agent(alpha=0.001, beta=0.001, 
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=100, layer1_size=400, layer2_size=300,
                n_actions=env.action_space.shape[0])
   
    n_games = 100
    
    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        state = env.reset()
        score = 0
        for j in range(100):
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, new_state, done)
            agent.learn()
            if len(sys.argv) == 2 and str(sys.argv[1]).lower() == 'default':
                env.render()
            score += reward
            state = new_state

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.2f' % score, 'trailing 100 games avg %.3f' % avg_score)
