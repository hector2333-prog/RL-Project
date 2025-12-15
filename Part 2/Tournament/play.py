#copied from campus virtual RL course
import supersuit as ss
from pettingzoo.atari import pong_v3
from pettingzoo.utils import agent_selector
from stable_baselines3 import PPO, DQN
import numpy as np
import sys
import logging
import importlib

from utils import *
from wrappers import *


if __name__ == "__main__":
    ### Controler
    if len(sys.argv) == 4:
        agent_left_group = sys.argv[1] # left agent
        agent_right_group = sys.argv[2] # right agent
        match_number = sys.argv[3] # number of match
    else:
        print("Usage: play <left_player> <right_player> <num_match>")
        print("Argument List: {}".format(str(sys.argv)))
        sys.exit(0)

    # PARAMS
    TIME_STEP_MAX = 10000#0
    MAX_INT = int(10e6)
    AGENTS_PATH = "./agents/"
    VIDEOS_PATH = "./videos/"
    LOGS_PATH = "./logs/"
    MATCH_NAME = "M{}-{}_{}".format(match_number, agent_left_group, agent_right_group)

    # Debugging: logging.INFO, logging.DEBUG
    log = set_logging(logging.INFO, LOGS_PATH + MATCH_NAME)

    ### Load the agents
    log.info(">>> MATCH {} / 5".format(match_number))

    # create ENV
    env = make_env()

    # LEFT AGENT
    # Dinamically import the left agent module and load the agent
    load_left_agent = getattr(importlib.import_module(agent_left_group +".load_agents"), "load_left_agent")
    agent_left = load_left_agent()

    # RIGHT AGENT
    # Dynamically import the right agent module and load the agent
    load_right_agent = getattr(importlib.import_module(agent_right_group +".load_agents"), "load_right_agent")
    agent_right = load_right_agent()
    
    # initializing rewards
    rewards = {agent: 0 for agent in env.possible_agents}

    # List of images to create a gif
    images = []


    # Start match
    for agent in env.agent_iter():
        # Getting the observation and action space
        obs, reward, termination, truncation, info = env.last()

        # Update the rewards
        for a in env.agents:
            rewards[a] += env.rewards[a]

        # If the game is over, break
        if termination or truncation:
            log.info("Episode terminated after {} frames!".format(len(images)))
            break
        else:
            # Select the action
            if agent == 'first_0':
                # right player
                act, _ = agent_right.predict(obs)
            elif agent == 'second_0':
                # left player
                act, _ = agent_left.predict(obs)
            else:
                log.error("ERROR: agent incorrect ('{}')".format(agent))
                sys.exit(0)

        # Perform the action
        env.step(act)

        # Store an image of the current state of the environment
        images.append(env.render())
        
        # If we have too many images, break (sometimes due to NOT sending the FIRE action)
        if len(images) > TIME_STEP_MAX:
            log.error("Breaking due to too many images ({})".format(len(images)))
            break

    # Close the environment
    env.close()

    # Match rewards
    agent_left_points = rewards['second_0']
    agent_right_points = rewards['first_0']
    winner = agent_left_group if agent_left_points > agent_right_points else agent_right_group
    log.info(">>> Rewards:")
    log.info("    Player LEFT : {} -> {} points".format(agent_left_group, agent_left_points))
    log.info("    Player RIGHT: {} -> {} points".format(agent_right_group, agent_right_points))
    log.info("    ...and the WINNER is group {}!".format(winner))

    # Export a gif video
    create_gif(images, agent_left_group, agent_right_group, match_number, VIDEOS_PATH, MATCH_NAME)

    # Export MP4
    create_mp4(agent_left_group, agent_right_group, match_number, VIDEOS_PATH, MATCH_NAME)