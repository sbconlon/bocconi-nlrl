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
    # Given the state and the textf action evaluations text,
    # return the formatted improvement prompt.
    #
    def get_improvement_prompt(self, state: str, action_set: dict, evals_text: str) -> str:
        #
        # Discrete action space - use the action set
        #
        if action_set:
            prompt = self.improvement_prompt.format(state=state,
                                                    actions=action_set,
                                                    evaluations=evals_text)
        #
        # Continuous space - don't use the action set (because it is empty)
        #
        else:
            prompt = self.improvement_prompt.format(state=state,
                                                    actions=action_set,
                                                    evaluations=evals_text)
        #
        # Return the prompt
        #
        return prompt

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
        user_prompt = self.get_improvement_prompt(state, action_set, evals_text)
        response = self.llm.generate_response([self.system_prompt],
                                              [user_prompt])[0]
        #
        # Log
        #
        print('-------------------')
        print('--> LLM Policy Improvement Operator')
        print('Prompt:')
        print(user_prompt)
        print()
        print('Response:')
        print(response)
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
        for action, action_eval in zip(actions, values):
            #
            # Add this evaluation to the text.
            #
            if action_set:
                evaluations_text += self.evaluation_description.format(action_id=action,
                                                                       action_str=action_set[action],
                                                                       evaluation=action_eval)
            else:
                evaluations_text += self.evaluation_description.format(action=action,
                                                                       evaluation=action_eval)
            evaluations_text += '\n\n'
        #
        # Return the final text containing all the evaluations
        #
        return evaluations_text