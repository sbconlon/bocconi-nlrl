# External imports
import json
import numpy as np
from pathlib import Path
import re

# Internal imports
from models.model import LanguageModel

class LanguagePolicy:

    def __init__(self, llm: LanguageModel, config: str, throw_formatting_errors: bool=False):
        #
        # LLM to query for actions
        #
        self.llm = llm
        #
        # Get the language policy's configuration parameters
        #
        with open(config, 'r') as file:
            self.config = json.load(file)
        #
        # Read the prompts from the files
        #
        self.system_prompt = Path(self.config['system_prompt_file']).read_text(encoding='utf-8')
        self.user_prompt = Path(self.config['user_prompt_file']).read_text(encoding='utf-8')
        #
        # How to handle ill-formatted responses from the llm
        #
        self.throw_formatting_errors = throw_formatting_errors

    #
    # Given a state, return an action from a set of possible actions and
    # a string explaining the reasoning.
    #
    #    Example input - Maze
    #
    #       state = """ #####
    #                   #o  #
    #                   #  x#
    #                   #####
    #               """
    #
    #       actions = {0: "Move up", 1: "Move down", 2: "Move right", 3: "Move left"}
    #
    #     Example output - Maze
    #
    #         action = 2
    #
    #         reason = """
    #           The player is currently standing next to a free space on the right side of 
    #           the maze. The goal square is also located on the right side of the maze. 
    #           Therefore, the best action for the player is to move right, as it will 
    #           bring them closer to the goal square and allow them to explore more of 
    #           the maze.
    #         """
    #
    def get_action(self, states : list[str], action_sets : list[dict]) -> list[str]:
        #
        # Get the prompt batch size
        #
        assert len(states) == len(action_sets), "Got a different number of states than action sets"
        N = len(states) 
        #
        # Get the prompts for each state
        #
        system_prompts = [self.system_prompt] * len(states)
        user_prompts = [self.user_prompt.format(state=states[i], 
                                                actions=action_sets[i]) for i in range(N)]
        #
        # Query the LLM with the given state and actions
        #
        responses = self.llm.generate_response(system_prompts, user_prompts)
        #
        # Log the LLM responses
        #
        for i in range(N):
            print('-------------------', flush=True)
            print('--> LLM Policy', flush=True)
            print(flush=True)
            print('Prompt:')
            print(user_prompts[i], flush=True)
            print(flush=True)
            print('Response:')
            print(responses[i], flush=True)
            print(flush=True)
        #
        # Return the list of responses from the LLM.
        #
        return responses

    #
    # Given a batch of policy targets, update the policy LLM
    #
    # policy_targets = [(state, available actions, policy_target),
    #                    ...]
    #
    def update(self, policy_targets: list[tuple[str, str]]) -> None:
        #
        # Format the targets into a list that can be used to create
        # a Hugging Face dataset object.
        #
        # data = [
        #          {'system_prompt': ..., 'user_prompt': ..., 'response': ...},
        #           ...
        #        ]
        #
        data = [
            {
                'system_prompt': self.system_prompt,
                'user_prompt': self.user_prompt.format(state=state, 
                                                       actions=actions),
                'response': policy_target
            } for state, actions, policy_target in policy_targets
        ]
        #
        # Train the LLM on the policy target data
        #
        self.llm.train(data)