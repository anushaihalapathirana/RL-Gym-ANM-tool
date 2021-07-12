import gym
import numpy as np
from gym_anm import MPCAgentConstant
from gym_anm import MPCAgentPerfect
from testEnv import TestEnvironment
from mpcPolicy import run

env = TestEnvironment()

# initialize mpc constant policy agent
mpcConstantPolicyAgent =  MPCAgentConstant(env.simulator, env.action_space, env.gamma,
                             safety_margin=0.96, planning_steps=10)
run(env, mpcConstantPolicyAgent)

