"""
This file gives the template to follow when creating new gym-anm environments.
For more information, see https://gym-anm.readthedocs.io/en/latest/topics/design_new_env.html.
"""


# inherit from ANMEnv super class
from gym_anm import ANMEnv

class CustomEnvironment(ANMEnv):

    def __init__(self):

        """
        network: a Python dictionary that describes the structure and characteristics of the distribution
        network G and the set of electrical devices D.
        """
        network = {'baseMVA': ...,
                   'bus': ...,
                   'device': ...,
                   'branch': ...}  # power grid specs

        """
        obs: a list of tuples corresponding to the variables to include in observation vectors. 
        all in MW units. 
        
        Alternatively, the obs object can be defined as a customized function
        that returns observation vectors when called (i.e., ot = obs(st)), or as a string ’state’.
In the later case, the environment becomes fully observable and observations ot = st are emitted

there are  combinations for the observation parameters are available
        """
        observation = ...          # observation space

        """
        the number of auxiliary variables K in the state vector given by 
        """
        K = ...                    # number of auxiliary variables


        delta_t = ...              # time interval between timesteps
        gamma = ...                # discount factor
        lamb = ...                 # penalty weighting hyperparameter
        aux_bounds = ...           # bounds on auxiliary variable (optional)
        costs_clipping = ...       # reward clipping parameters (optional)
        seed = ...                 # random seed (optional)

        super().__init__(network, observation, K, delta_t, gamma, lamb,
                         aux_bounds, costs_clipping, seed)

    def init_state(self):
        ...

    """
    method that receives the current state vector s(t) and should return the outcomes of the internal variables for timestep t + 1. 
    It must be implemented by the designer of the task, with the only constraint being that it must return a list of |DL| + |DRER| + K values.
    """
    def next_vars(self, s_t):
        ...

    """
    This method is optional and only useful if the observation space is specified as
a callable object. In the latter case, observation_space() should return the (potentially loose) bounds of
the observation space O, so that agents can easily normalize emitted observation vectors.
    """
    def observation_bounds(self):  # optional
        ...

    """
 support rendering of the interactions
between the agent and the new environment. render() should update the visualization every time it gets
called, 
   """
    def render(self, mode='human'):  # optional
        ...

"""
close() should end the rendering process. 
"""
    def close(self):  # optional
        ...
