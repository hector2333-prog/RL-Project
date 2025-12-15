#copied from CV RL course
import numpy as np
import random
import supersuit as ss
from pettingzoo.atari import pong_v3


# Constants
MAX_INT = int(10e6)
TIME_STEP_MAX = 100000


def get_seed(MAX_INT=int(10e6)):
    '''
    Return a random seed between 0 and MAX_INT.
    
    :param MAX_INT: Maximum integer value for the seed.
    :return: A random integer seed.
    '''
    return random.randint(0, MAX_INT)


def make_env(render_mode="rgb_array"):
    '''
    Create a pre-processed Pong environment using SuperSuit wrappers.

    :param render_mode: The render mode for the environment.
    :return: A pre-processed Pong environment.
    '''
    # Create the environment
    env = pong_v3.env(num_players=2, render_mode=render_mode)

    # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    # Pre-process using SuperSuit
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))
    
    # We evaluate here using an AEC environments
    env.reset(seed=get_seed(MAX_INT))
    env.action_space(env.possible_agents[0]).seed(get_seed(MAX_INT))

    return env