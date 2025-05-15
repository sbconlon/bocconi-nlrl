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
    def get_action(self, states : list[str], action_sets : list[dict]) -> tuple[int, str]:
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
        # Parse the action and resoning from each response
        #
        actions, reasons = [], []
        for i in range(N):
            print('-------------------', flush=True)
            print('--> LLM Policy', flush=True)
            print(flush=True)
            print('Input state:', flush=True)
            print(states[i], flush=True)
            print(flush=True)
            #
            # Extract the action and resoning from the response
            #
            action = self.extract_action_from_response(responses[i], action_sets[i])
            reason = self.extract_reason_from_response(responses[i])
            #
            # Log
            #
            print('Action:', action_sets[i][action], flush=True)
            print(flush=True)
            print('Reason:', flush=True)
            print(reason, flush=True)
            print(flush=True)
            #
            # Save the action and reasoning to the output lists
            #
            actions.append(action)
            reasons.append(reason)
        #
        # Return the list of selected actions and their reasonings.
        #
        return actions, reasons
    
    #
    # Extract the selected action from the LLM response text.
    #
    # Raise an error if the action isn't found or if the extracted
    # action is not in the given actions dictionary.
    #
    def extract_action_from_response(self, response: str, actions: dict[int, str]) -> int:
        #
        # Response must contain this pattern
        #
        # action_match = re.search(r'Best action:\s*(\d+)', response)
        action_match = re.search(r'Your desicion:\s*(\d+)', response)

        if action_match:
            #
            # Success case - match found.
            #
            action = int(action_match.group(1))
        else:
            #
            # Failure case - no match found, raise a value error or pick a random action.
            #
            message_str = f"Missing action. Policy LLM returned an ill-formatted response. Response:\n'{response}'"
            if self.throw_formatting_errors:
                raise ValueError(message_str)
            else:
                action, _ = self.get_random_action(actions)
                print('WARNING: ' + message_str)
        #
        # Check that the found action id is valid.
        #
        # If not, either raise an error or pick a random action.
        #
        if action not in actions.keys():
            message_str = f"Policy LLM selected an invalid action. Got {action}. Expected one of these {actions}\n Response: {response}"
            if self.throw_formatting_errors:
                raise ValueError(message_str)
            else:
                action, _ = self.get_random_action(actions)
                print('WARNING: ' + message_str)
        #
        # Return the action id.
        #
        return action

    #
    # Extract the reasoning from the LLM response text.
    #
    # Raise an error if the reasoning isn't found.
    #
    def extract_reason_from_response(self, response: str) -> str:
        #
        # Response must contain this pattern.
        #
        # reason_match = re.search(r"Reason:\s*\n?(.*)", response, re.DOTALL)
        reason_match = re.search(r"Message:\s*\n?(.*)", response, re.DOTALL)

        if reason_match:
            #
            # Success case - match found, extract the reasoning string.
            #
            reason =  str(reason_match.group(1))
        else:
            #
            # Failure case - no match found, raise a value error or set the
            #                reason to an empty string.
            #
            message_str = f"Missing reasoning. Policy LLM return an ill-formatted response. Response:\n'{response}'"
            if self.throw_formatting_errors:
                raise ValueError(message_str)
            else:
                reason = ''
                print('WARNING: ' + message_str)
        #
        # Return the reasoning string
        #
        return reason

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

    #
    # Given a list of possible actions, select one randomly and give an empty reason.
    #
    def get_random_action(self, actions : dict[int, str]) -> tuple[int, str]:
        return np.random.choice(list(actions.keys())), ''
    
    #
    # Given a state and action, return the response from the LLM.
    #
    def get_message(self, state: str, action: int) -> str:
        #
        # Get the prompt for the given state and action
        #
        system_prompt = self.system_prompt.format(action=action)
        prompt = self.user_prompt.format(state=state, actions=action)
        #
        # Get the response from the LLM
        #
        response = self.llm.generate_response(system_prompt, prompt)
        #
        # Extract the message from the response
        #
        message = self.extract_message_from_response(response)
        #
        # Return the message
        #
        return message
    
    #
    # Extract the message from the LLM response text.
    #
    # Raise an error if the message isn't found.
    #
    def extract_message_from_response(self, response: str) -> str:
        #
        # Response must contain this pattern.
        #
        message_match = re.search(r"Message:\s*\n?(.*)", response, re.DOTALL)

        if message_match:
            #
            # Success case - match found, extract the message string.
            #
            message = str(message_match.group(1))
        else:
            #
            # Failure case - no match found, raise a value error or set the
            #                message to an empty string.
            #
            message_str = f"Missing message. Policy LLM returned an ill-formatted response. Response:\n'{response}'"
            if self.throw_formatting_errors:
                raise ValueError(message_str)
            else:
                message = ''
                print('WARNING: ' + message_str)
        #
        # Return the message string
        #
        return message
    