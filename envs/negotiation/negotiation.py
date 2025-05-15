# External imports
from pathlib import Path
import re

# Internal imports
from envs.environment import Environment
from agents.language_policy import LanguagePolicy
from models.model import LanguageModel as llm

class NegotiationEnv(Environment):
    #
    # board_file - text file containing the starting state for the negotiation
    #
    # Example:
    # item_description: "A 2022 Tesla Model 3 in excellent condition with 15,000 miles. The car has been well-maintained with all service records available. It includes features like autopilot, premium sound system, and glass roof. The battery health is at 98% and the car has never been in any accidents. The original MSRP was $48,000."
    # item_price: 40000
    # chat_log: []
    # actions: -1 to 200 where:
    #   -1: reject
    #   0-100: propose new price (percentage of target price)
    #   101-200: accept other's proposal (100 + their proposal percentage)

    def __init__(self, negotiation_file: str):
        #
        # Save the negotiation file location
        #
        self.negotiation_file = negotiation_file
        #
        # Initialize the negotiation as finished.
        #
        self.terminated = True
        self.item_price = None
        self.chat_log = []
        # Action set is a list of integers from -1 to 200
        # -1: reject
        # 0-100: propose new price
        # 101-200: accept other's proposal (100 + their proposal)
        self.action_set = list(range(-1, 201))
        
        # Track whose turn it is
        self.current_turn = "buyer"  # Start with buyer
        #
        # Initialize the language policy
        #
        self.lang_policy = LanguagePolicy(llm, 
                                          'configs/policy.json'
        )
        #
        # Initialize the environment.
        #
        self.reset() 
    
    #
    # Reset the environment to its initial state
    #
    def reset(self):
        #
        # Read the initial game state from the board file.
        #
        starting_state = Path(self.negotiation_file).read_text(encoding='utf-8')
        #
        # Set the environment to its starting state
        #
        self.set_state(starting_state)
        # Reset terminated flag
        self.terminated = False
        # Reset turn to buyer
        self.current_turn = "buyer"

    #
    # Given a state, set the environment to that state.
    #
    def set_state(self, state : str) -> None:
        #
        # Set the environment state
        #
        self.state = state
        #
        # Set the chat log 
        #
        self.chat_log = self.state.split('\n')[3:]
        #
        # Chat is not over since we are reseting to a state.
        #
        self.terminated = False
        #
        # Update current turn based on last message in chat log
        #
        if self.chat_log:
            last_role = self.chat_log[-1].split(': ')[0]
            self.current_turn = "seller" if last_role == "buyer" else "buyer"
        else:
            self.current_turn = "buyer"

    #
    # Return a list of all valid actions in the environment
    #
    def actions(self) -> list:
        return self.action_set
    
    #
    # Return - True if the game has terminated.
    #        - False otherwise.
    #
    def is_terminal(self):
        return self.terminated
    
    #
    # Apply the given action in the environment, actually updating the chat log.
    # return the resulting next state and the reward gained.
    #
    def act(self, action_id : int) -> tuple[str, int]:
        """
        Apply the given action in the environment.
        action_id: 
            -1: reject
            0-100: propose new price (percentage of target price)
            101-200: accept other's proposal (100 + their proposal percentage)
        """
        assert not self.terminated, "Trying to act in a terminated chat."
        assert -1 <= action_id <= 200, f"Action id must be between -1 and 200, got {action_id}"
        
        if action_id == -1 or action_id >= 101:
            # Reject or accept other's proposal
            self.terminated = True
        else:
            # Propose new price
            pass
       
        message = self.lang_policy.get_message(self.state, action_id)
        
        # Update the chat log with the response
        self.chat_log.append(f"{self.current_turn}: {action_id}: {message}")
        
        # Update the state with the new chat log
        self.state = '\n'.join(self.state.split('\n')[:2] + self.chat_log)
        
        # Switch turns if not terminated
        if not self.terminated:
            self.current_turn = "seller" if self.current_turn == "buyer" else "buyer"
        
        # Calculate reward
        reward = self.calculate_reward(action_id)
        
        return self.state, reward
    
    #
    # Calculate the reward for the given action.
    #
    def calculate_reward(self, action_id: int) -> int:
        """
        Calculate reward for any action during negotiation.
        action_id: 
            -1: reject (reward = -1)
            0-100: propose new price (reward = (1-action)*0.01 for buyer, action*0.01 for seller)
            101-200: accept other's proposal (reward = 1+(1-action)*0.01 for buyer, 1+action*0.01 for seller)
        """
        if action_id == -1:
            return 0  # Reject gets less reward
        
        elif action_id >= 101:
            # Accept other's proposal
            other_proposal = action_id - 100
            if self.current_turn == "buyer":
                return 100 + (100 - other_proposal)  
            else:
                return 100 + other_proposal
        
        else:
            # Propose new price
            if self.current_turn == "buyer":
                return 100 - action_id
            else:
                return action_id
            
    #
    # Return a string containing a detailed description of the negotiation state.
    #
    def describe_state(self) -> str:
        return self.state

    #
    # Given a trajectory, return a string containing a detailed description of the trajectory.
    #
    def describe_trajectory(self, trajectory: list[tuple[str, int, int]]) -> str:
        """Return the complete negotiation state and conversation from a trajectory."""
        self.set_state(trajectory[-1][0])
        return self.state