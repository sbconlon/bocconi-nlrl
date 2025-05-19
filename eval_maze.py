# External imports
import argparse
import json

# Internal imports
from agents.actorcritic import ActorCriticAgent
from envs.maze.maze import MazeEnv
from models.mistral import Mistral
from models.tiny_llama import TinyLlama

#
# This file is for evaluating a given model on a given maze.
#
if __name__ == "__main__":
    #
    # Parse CLI arguments
    #
    parser = argparse.ArgumentParser(prog='ActorCriticEvaluation')
    parser.add_argument('eval_config')
    args = parser.parse_args()
    #
    # Open the evaluation configuration file
    #
    with open(args.eval_config, 'r') as file:
        eval_config = json.load(file)
    #
    # Initialize the environment
    #
    env = MazeEnv(eval_config['initial_board'])
    #
    # Open the model configuration file and initialize the model object.
    #
    with open(eval_config['model_config'], 'r') as file:
        model_config = json.load(file)
    if model_config['type'] == 'Mistral':
        llm = Mistral(eval_config['model_config'])
    elif model_config['type'] == 'TinyLlama':
        llm = TinyLlama(eval_config['model_config'])
    else:
        raise ValueError(f"Unrecognized model type: {model_config['type']}")
    #
    # Set the model in eval mode.
    #
    #llm.eval()
    #
    # Initialize the agent
    #
    agent = ActorCriticAgent(env, llm, eval_config['agent_config'])
    #
    # Sample one trajectory from the agent.
    #
    # Note: Use epsilon=0 because we want to evaluate the agent.
    #
    env.reset()
    trajectory = agent.rollout(
        [env], 
        max_trajectory_length=eval_config['max_trajectory_length'],
        epsilon=0.,
        action_temp=eval_config['action_temp']
    )
    #
    # Print the trajectory statistics
    #
    agent.update_stats(trajectory)
    agent.print_stat_summary()