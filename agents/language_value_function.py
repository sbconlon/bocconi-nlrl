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
    # Given state-action pairs and a set of example trajectories, 
    # return the LLM's response estimating the Monte-Carlo value.
    #
    def mc_estimate(self, sa_pairs : list[tuple[str, int]],  
                          action_sets : list[dict[int, str]], 
                          trajectory_samples_lst : list[list[str]]) -> str:
        print('mc_estimate', flush=True)
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
            # Save the system prompt for this sa pair.
            #
            system_prompts.append(self.system_prompt.format(actions=action_set.values()))
            #
            # Save the user prompt for this sa pair.
            #
            user_prompts.append(
                self.mc_estimate_prompt.format(
                    state=state, 
                    action=action_set[action], 
                    examples=traj_text
                )
            )
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
            response = responses[i]
            #
            # Log
            #
            print('-------------------', flush=True)
            print('--> LLM MC Estimate', flush=True)
            print(flush=True)
            print('Input state-action pair:', flush=True)
            print(state, flush=True)
            print(flush=True)
            print('Action:', action_set[action], flush=True)
            print(flush=True)
            #
            # Verify the response's formatting by extracting the value
            # and reasoning.
            #
            value = self.extract_value_from_response(response)
            reason = self.extract_reason_from_response(response)
            #
            # Log
            #
            print('Value:', value, flush=True)
            print(flush=True)
            print('Reason:', flush=True)
            print(reason, flush=True)
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
            traj_samples_text += f"Trajectory example {i}:\n"
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
    def update(self, target_values : list[tuple], actions : dict[int, str]) -> None:
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
                'system_prompt': self.system_prompt.format(actions=actions.values()),
                'user_prompt': self.value_prompt.format(state=state, action=action),
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
            system_prompts.append(
                self.system_prompt.format(actions=action_sets[i].values())
            )
            #
            # Format the user prompt for this state-action pair
            #
            user_prompts.append(
                self.value_prompt.format(state=states[i], 
                                         action=action_sets[i][actions[i]])
            )
        #
        # Query the LLM with the batch of prompts
        #
        responses = self.llm.generate_response(system_prompts, user_prompts)
        #
        # Log and verify the formatting of each response
        #
        for i in range(N):
            #
            # Log
            #
            print('-------------------')
            print('--> LLM Value function')
            print()
            print('Input state-action pair:')
            print(states[i])
            print()
            print('Action:', action_sets[i][actions[i]])
            print()
            #
            # Verify the response's formatting by extracting the value
            # and reasoning.
            #
            value = self.extract_value_from_response(responses[i])
            reason = self.extract_reason_from_response(responses[i])
            #
            # Log
            #
            print('Value:', value)
            print()
            print('Reason:')
            print(reason)
        #
        # Return the LLM responses
        #
        return responses