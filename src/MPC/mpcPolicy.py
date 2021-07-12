import gym
import numpy as np
from gym_anm import MPCAgentConstant
from testEnv import TestEnvironment

def run(env, agent):
    env = env
    o = env.reset()

    # Initialize the MPC policy.
    agent = agent

    n_games = 100
    
    best_score = env.reward_range[0]
    score_history = []
    score = 0
    # Run the policy.
    for step in range(n_games):
        state = env.reset()
        action = agent.act(env)
        new_state, reward, done, info = env.step(action)
        print(f'step = {step}, reward = {reward:.3}')
        score += reward
        state = new_state

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    print('score %.2f' % score, 'trailing 100 games avg %.3f' % avg_score)


# if __name__ == '__main__':
#     run()