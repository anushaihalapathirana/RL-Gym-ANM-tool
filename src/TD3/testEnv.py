import numpy as np
from gym_anm import ANMEnv
from gym_anm import MPCAgentConstant
from gym_anm import MPCAgentPerfect

"""
A 2-bus power grid with topology:
  
    Slack -----
            |         
          ------------------     
         |     |           |
        load   storage   wind power
"""

"""
microgrid topology
"""
network = {
    'baseMVA': 100,
    'bus': np.array([
    [0, 0, 132, 1., 1.],
    [1, 1, 33, 1.1, 0.9]
]),
    'device': np.array([
    [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None], #slack
    [1, 1, -1, 0.2, 0, -10,  None, None, None, None, None, None, None, None, None], # load
    [2, 1, 3, None, 50, -50, 50, -50, 30, -30, 25, -25, 100, 0, 0.9],# storage
    [3, 1, 2, None, 30, 0, 30, -30, 20, None, 15, -15, None, None, None] # wind power
]),
    'branch': np.array([
    [0, 1,  0.03,  0.022, 0., 25, 1, 0]
])
}


class TestEnvironment(ANMEnv):

    def __init__(self):
        observation = 'state'             # fully observable environment
        K = 1                             # 1 auxiliary variable
        delta_t = 0.25                    # 15min intervals
        gamma = 0.9                       # discount factor
        lamb = 100                        # penalty weighting hyperparameter
        aux_bounds = np.array([[0, 24 / delta_t - 1]])  # bounds on auxiliary variable
        costs_clipping = (1, 100)         # reward clipping parameters
        seed = 1                          # random seed

        super().__init__(network, observation, K, delta_t, gamma, lamb,
                         aux_bounds, costs_clipping, seed)

    def init_state(self):
        n_dev = self.simulator.N_device              # number of devices
        n_des = self.simulator.N_des                 # number of DES units
        n_gen = self.simulator.N_non_slack_gen       # number of non-slack generators
        N_vars = 2 * n_dev + n_des + n_gen + self.K  # size of state vectors - calculated based on number of devices, gen and storage units - integer

        state = np.random.rand(N_vars) 

        return state                                 # random state vector

    def next_vars(self, s_t):
        P_load = -10 * np.random.rand(1)[0]       # load injections  
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))            
        p_max = (50 * np.random.rand(1)[0])     #max generation from non slack generators
        return np.array([P_load, p_max, aux])

