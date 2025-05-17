# External imports
import json
from pathlib import Path
import re

# Internal imports
from models.model import LanguageModel

class LanguageValueFunction:
    def __init__(self, llm: LanguageModel, config: str, throw_formatting_errors: bool=False):
        #
        # LLM to query for values and MC estimates
        #
        self.llm = llm
        #
        # Get the language value function's configuration parameters
        #
        with open(config, 'r') as file:
            self.config = json.load(file)
        #
        # Read the prompts from the files
        #
        self.system_prompt = Path(self.config['system_prompt']).read_text(encoding='utf-8')
        self.value_prompt = Path(self.config['value_prompt']).read_text(encoding='utf-8')
        self.mc_estimate_prompt = Path(self.config['mc_estimate_prompt']).read_text(encoding='utf-8')
        self.transition_description = Path(self.config['describe_transition']).read_text(encoding='utf-8')
        #
        # How to handle ill-formatted responses from the llm
        #
        self.throw_formatting_errors = throw_formatting_errors

    #
    # Given the action set, format the system prompt and return it.
    #
    def get_system_prompt(self, action_set: dict) -> str:
        #
        # If the action space is discrete, use it in the LLM prompt.
        #
        if action_set:
            #
            # Pass the action set to the system prompt so that the LLM knows
            # the avaible set of actions.
            #
            prompt = self.system_prompt.format(actions=action_set.values())
        #
        # Otherwise, the action space is continuous and needs to be described in the
        # prompt itself.
        #
        else:
            #
            # If there isn't an action set given, then the action set
            # must be described in the system prompt itself.
            #
            # For the negotiation game, the LLM knows the action set already.
            #
            prompt = self.system_prompt
        #
        # Return the propmpt
        #
        return prompt
                
    #
    # Given a state-action pair plus the trajectories, format the
    # Monte-Carlo estimate prompt.
    #
    def get_mc_estimate_prompt(self, state: str, action, action_set: dict, traj_text: str) -> str:
        #
        # If we are given a discrete action set, then use it.
        # Otherwise, format the prompt with the raw action.
        #
        action_str = action_set[action] if action_set else action
        #
        # Format the prompt
        #
        prompt = self.mc_estimate_prompt.format(
            state=state, 
            action=action_str, 
            examples=traj_text
        )
        #
        # Return the formatted prompt
        #
        return prompt

    #
    # Given a state-action pair, format the value prompt.
    #
    def get_value_prompt(self, state: str, action, action_set: dict) -> str:
        #
        # If we are given a discrete action set, then use it.
        # Otherwise, format the prompt with the raw action.
        #
        action_str = action_set[action] if action_set else action
        #
        # Format the prompt
        #
        prompt = self.value_prompt.format(state=state, action=action_str)
        #
        # Return the formatted prompt
        #
        return prompt

    #
    # Given state-action pairs and a set of example trajectories, 
    # return the LLM's response estimating the Monte-Carlo value.
    #
    def mc_estimate(self, sa_pairs : list[tuple],  
                          action_sets : list[dict], 
                          trajectory_samples_lst : list[list[str]]) -> str:
        #
        # Get the prompt batch size
        #
        assert len(sa_pairs) == len(action_sets), "Input batch size mismatch."
        assert len(sa_pairs) == len(trajectory_samples_lst), "Input batch size mismatch."
        N = len(sa_pairs)
        #
        # Get the prompts for each state-action pair
        #
        system_prompts, user_prompts = [], []
        for i in range(N):
            #
            # Unpack the information for this sa pair.
            #
            state, action = sa_pairs[i]
            action_set = action_sets[i]
            trajectory_samples = trajectory_samples_lst[i]
            #
            # Describe the given sampled trajectories in text
            #
            traj_text = self.trajectories_to_text(action_set, trajectory_samples)
            #
            # Save the system and user prompts for this sa pair.
            #
            system_prompts.append(self.get_system_prompt(action_set))
            user_prompts.append(self.get_mc_estimate_prompt(state, action, action_set, traj_text)) 
        #
        # Query the LLM with the prompts
        #
        responses = self.llm.generate_response(system_prompts, user_prompts)
        #
        # Verify that each response is formatted correctly.
        #
        for i in range(N):
            #
            # Unpack sa pair info
            #
            state, action = sa_pairs[i]
            action_set = action_sets[i]
            action_str = action_set[action] if action_set else action
            response = responses[i]
            #
            # Log
            #
            print('-------------------', flush=True)
            print('--> LLM MC Estimate', flush=True)
            print('Prompt:', flush=True)
            print(user_prompts[i], flush=True)
            print()
            print('Response:')
            print(responses[i], flush=True)
        #
        # Otherwise, the selected action is valid.
        #
        return responses

    #
    # Given a list of trajectories, return a string describing each trajectory.
    #
    def trajectories_to_text(self, actions : dict, trajectories : list[tuple[str, int, int]]) -> str:
        #
        # Use the tranisition description to format
        # each trajectory sample into text.
        #
        traj_samples_text = ""
        for i, traj_description in enumerate(trajectories, 1):
            #
            # Trajectory header to delineate it from other samples.
            #
            traj_samples_text += f"Example {i}:\n"
            #
            # Add the description of each transition to the trajectory text.
            #
            #
            # Describe this transition in text.
            #
            # NOTE - This reward str needs to be abstracted in the future.
            #        It only applies to the maze environment.
            #
            traj_samples_text += traj_description
        #
        # Return the final description.
        #
        return traj_samples_text
    
    #
    # Given a response from the LLM, extract the value.
    #
    def extract_value_from_response(self, response : str) -> float:
        value_match = re.search(r'Value:\s*([-+]?\d+(?:\.\d+)?)', response)
        if value_match:
            value = float(value_match.group(1))
        else:
            #
            # Fail case - Either throw an error or pick an arbitrary value of zero.
            #
            message_str = f"Missing value. MC Estimate LLM returned an ill-formatted response. Response:\n'{response}'"
            if self.throw_formatting_errors:
                raise ValueError(message_str)
            else:
                value = 0.0
                print('WARNING: ' + message_str)

        return value

    #
    # Given a response form the LLM, extract the reasoning.
    #
    def extract_reason_from_response(self, response : str) -> str:
        reason_match = re.search(r"Reason:\s*\n?(.*)", response, re.DOTALL)
        if reason_match:
            reason =  str(reason_match.group(1))
        else:
            #
            # Fail case - Either throw an error or return an empty string.
            #
            message_str = f"Missing reasoning. MC Estimate LLM return an ill-formatted response. Response:\n'{response}'"
            if self.throw_formatting_errors:
                raise ValueError(message_str)
            else:
                reason = ''
                print('WARNING: ' + message_str)
        return reason

    #
    # Given a batch of target values, update the value function.
    #
    #    target_values = [
    #        (state, action, value),
    #         ...
    #    ]
    #
    #    where:
    #       - state = string describing the state
    #       - action = action id from the environment
    #       - value = string describing the Monte-Carlo estimate of the state-action pair
    #
    def update(self, target_values : list[tuple], action_set : dict[int, str]) -> None:
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
                'system_prompt': self.get_system_prompt(action_set),
                'user_prompt': self.get_value_prompt(state, action, action_set),
                'response': value_target
            } for state, action, value_target in target_values
        ]
        #
        # Train the LLM on the data
        #
        self.llm.train(data)

    #
    # Given a state-action pair, return the value from the value function
    #
    def get_value(self, states: list[str], actions: list[int], action_sets: list[dict[int, str]]) -> str:
        #
        # Get batch prompt size
        #
        assert len(states) == len(actions), "Input list length mismatch"
        assert len(states) == len(action_sets), "Input list length mismatch"
        N = len(states)
        #
        # Get the list of prompts
        #
        system_prompts, user_prompts = [], []
        for i in range(N):
            #
            # Format the system prompt for this state-action pair
            #
            system_prompts.append(self.get_system_prompt(action_sets[i]))
            #
            # Format the user prompt for this state-action pair
            #
            user_prompts.append(self.get_value_prompt(states[i], actions[i], action_sets[i]))
        #
        # Query the LLM with the batch of prompts
        #
        responses = self.llm.generate_response(system_prompts, user_prompts)
        #
        # Log and verify the formatting of each response
        #
        for i in range(N):
            action_str = action_sets[i][actions[i]] if action_sets[i] else actions[i]
            #
            # Log
            #
            print('-------------------')
            print('--> LLM Value function')
            print('Prompt:')
            print(user_prompts[i], flush=True)
            print()
            print('Response:')
            print(responses[i], flush=True)
            print()
        #
        # Return the LLM responses
        #
        return responses