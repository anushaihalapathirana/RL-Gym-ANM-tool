
import gym
import time

def run():
    env = gym.make('gym_anm:ANM6Easy-v0')
    o = env.reset() # create innitial observation space

    for i in range(10):
        a = env.action_space.sample() # the agent samples random actions from the action space of the ANM6Easy-v0 task for 10 timesteps.
        o, r, done, info = env.step(a)
        env.render()
        time.sleep(0.5)   # otherwise the rendering is too fast for the human eye

        # A terminal state will reach if no solution to the power flow equations is found.
        # power grid has collapsed and is often due to a voltage collapse problem
        if done:
            # Every time a terminal state is reached, the environment gets reset.
            o = env.reset()
    env.close()

if __name__ == '__main__':
    run()


