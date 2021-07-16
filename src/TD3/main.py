import gym
import sys
from testEnv import TestEnvironment
from traintd3 import trainTD3
from testtd3 import testTD3

if __name__ == '__main__':
    if len(sys.argv) == 2 and str(sys.argv[1]).lower() == 'default':
        print("environment is default gym_anm ANM6Easy-v0")
        env = gym.make('gym_anm:ANM6Easy-v0')
    else: 
        env = TestEnvironment()
    trainTD3(env)
    print("*******  Training Done  *************")
    testTD3(env)