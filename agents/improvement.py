# External imports
import json
import numpy as np
from pathlib import Path
import re

# Internal import
from models.model import LanguageModel

class ImprovementOperator:
    def __init__(self, llm: LanguageModel, config: str, throw_formatting_errors: bool):
        #
        # LLM to query
        #
        self.llm = llm
        #
        # Get the improvement operator's configuration parameters
        #
        with open(config, 'r') as file:
            self.config = json.load(file)
        #
        # Read the prompts from the files
        #
        self.system_prompt = Path(self.config['system_prompt']).read_text(encoding='utf-8')
        self.improvement_prompt = Path(self.config['improvement_prompt']).read_text(encoding='utf-8')
        self.evaluation_description = Path(self.config['describe_evaluation']).read_text(encoding='utf-8')
        #
        # How to handle ill-formatted responses from the llm
        #
        self.throw_formatting_errors = throw_formatting_errors

    #
    # Given state-aciton pairs and descriptions of their values from the language
    # value function, perform chain-of-thought reasoning to determine the best action.
    #
    # Return the strategic reasoning for what's the best action and why.
    #
    def reason(self, state: list[str], actions: list[int], values: list[str], action_set: dict[int, str]) -> str:
        #
        # Format the state-action pair evaluations into a string that can be plugged into
        # the user prompt.
        #
        evals_text = self.evaluations_to_text(actions, values, action_set)
        #
        # Query the LLM with the state-action pair evaluations
        #
        response = self.llm.generate_response([self.system_prompt],
                                              [self.improvement_prompt.format(state=state,
                                                                             actions=action_set,
                                                                             evaluations=evals_text)])[0]
        #
        # Log
        #
        print('-------------------')
        print('--> LLM Policy Improvement Operator')
        print()
        print('Input state:')
        print(state)
        print()
        print('Input action evaluations:')
        print(evals_text)
        print()
        #
        # Verify the response's formatting by extracting the best action and reasoning.
        #
        action = self.extract_action_from_response(response, action_set)
        reason = self.extract_reason_from_response(response)
        #
        # Log
        #
        print('Action:', action_set[action])
        print()
        print('Reason:')
        print(reason)
        print()
        #
        # Return the response
        #
        return response

    #
    # Format the given state-action pair evaluations into a single string.
    #
    def evaluations_to_text(self, actions: list[int], values: list[str], action_set: dict[int, str]) -> str:
        #
        # Use the evaluations description to format each 
        # action evaluation into text.
        #
        evaluations_text = ""
        for action_id, action_eval in zip(actions, values):
            #
            # Add this evaluation to the text.
            #
            evaluations_text += self.evaluation_description.format(action_id=action_id,
                                                                   action_str=action_set[action_id],
                                                                   evaluation=action_eval)
            evaluations_text += '\n\n'
        #
        # Return the final text containing all the evaluations
        #
        return evaluations_text
    
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
        action_match = re.search(r'Best action:\s*(\d+)', response)
        if action_match:
            #
            # Success case - match found.
            #
            action = int(action_match.group(1))
        else:
            #
            # Failure case - no match found, raise a value error or pick a random action.
            #
            message_str = f"Missing action. Improvement Operator LLM returned an ill-formatted response. Response:\n'{response}'"
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
            message_str = f"Improvement Operator LLM selected an invalid action. Got {action}. Expected one of these {actions}\n Response: {response}"
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
        reason_match = re.search(r"Reason:\s*\n?(.*)", response, re.DOTALL)
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
            message_str = f"Missing reasoning. Improvement Operator LLM return an ill-formatted response. Response:\n'{response}'"
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
    # Given a list of possible actions, select one randomly and give an empty reason.
    #
    def get_random_action(self, actions : dict[int, str]) -> tuple[int, str]:
        return np.random.choice(list(actions.keys())), ''