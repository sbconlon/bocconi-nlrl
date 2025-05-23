# External imports
import argparse
import json
import os
from datetime import datetime

import cProfile
import pstats

# Internal imports
from agents.actorcritic import ActorCriticAgent
from envs.negotiation.negotiation import NegotiationEnv
from models.mistral import Mistral
from models.tiny_llama import TinyLlama

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'
    
    # Redirect all output to the log file
    with open(log_file, 'w') as f:
        print(f"Training log will be saved to: {log_file}")
        
        #
        # Runtime profiler
        #
        profiler = cProfile.Profile()
        profiler.enable()

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
        env = NegotiationEnv(train_config['initial_board'])
        #
        # Open the model configuration file and initialize the model object.
        #
        with open(train_config['model_config'], 'r') as file:
            model_config = json.load(file)
        if model_config['type'] == 'Mistral':
            llm = Mistral(train_config['model_config'])
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