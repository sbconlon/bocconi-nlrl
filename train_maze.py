# External imports
import argparse
import json

import cProfile
import pstats

# Internal imports
from agents.actorcritic import ActorCriticAgent
from envs.maze.maze import MazeEnv
from models.mistral import Mistral
from models.tiny_llama import TinyLlama

if __name__ == "__main__":
    
    
    """
    NOTE - Disabled runtime profiler
    #
    # Runtime profiler
    #
    profiler = cProfile.Profile()
    profiler.enable()
    """

    #
    # Parse CLI arguements
    #
    parser = argparse.ArgumentParser(prog='ActorCriticTraining')
    parser.add_argument('train_config')
    args = parser.parse_args()
    #
    # Open the train configuration file
    #
    with open(args.train_config, 'r') as file:
        train_config = json.load(file)
    #
    # Initialize the environment
    #
    env = MazeEnv(train_config['initial_board'])
    #
    # Open the model configuration file and initialize the model object.
    #
    with open(train_config['model_config'], 'r') as file:
        model_config = json.load(file)
    if model_config['type'] == 'Mistral':
        llm = Mistral(train_config['model_config'])
    elif model_config['type'] == 'TinyLlama':
        llm = TinyLlama(train_config['model_config'])
    else:
        raise ValueError(f"Unrecognized model type: {model_config['type']}")
    #
    # Initialize the agent
    #
    agent = ActorCriticAgent(env, llm, train_config['agent_config'])
    #
    # Run training loop
    #
    agent.train(
        T=train_config['num_train_loops'], 
        N=train_config['num_trajectories_per_loop'], 
        K=train_config['num_sample_trajectories_per_mc']
    )

    """
    NOTE: Disabling runtime profiler
    #
    # Runtime profiler
    #
    # Load the profiler into pstats
    stats = pstats.Stats(profiler)
    profiler.disable()
    # Simplify file paths for readibility
    stats.strip_dirs()
    # Sort by 'tottime percall' (average time per call)
    stats.sort_stats(pstats.SortKey.TIME)
    # Print the top 20 functions
    stats.print_stats(20)
    """