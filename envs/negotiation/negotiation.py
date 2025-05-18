# External imports
from copy import deepcopy
import numpy as np
from pathlib import Path
import re

# Internal imports
from envs.environment import Environment
from agents.language_policy import LanguagePolicy
from models.model import LanguageModel as llm

class NegotiationEnv(Environment):
    #
    # negotiation_file - text file containing the starting state for the negotiation
    #
    # Starting state
    #
    #   - item_description: "A 2022 Tesla Model 3 in excellent condition with 15,000 miles. 
    #                        The car has been well-maintained with all service records 
    #                        available. It includes features like autopilot, premium sound 
    #                        system, and glass roof. The battery health is at 98% and the 
    #                        car has never been in any accidents. The original MSRP was 
    #                        $48,000."
    #   
    #   - item_value: 40000
    #
    #   - chat_log: []
    #
    # Players
    #
    #   - 'Buyer': seeks to negotiate the lowest price possible.
    #
    #   - 'Seller': seeks to negotiate the highest price possible.
    #
    # Reward
    #
    #   - The negotiation game is a zero-sum game. The reward is the difference
    #     between the sale's price the agents agree on and the item's value.
    #
    #   - If the players decide not to make a deal, then no reward is given.
    #
    # Action
    #
    #   - Each player responds with the following action format:
    #
    #        action = {'price': price, 'message': message}
    #
    #        where
    #
    #           - price = The price the current player is offering
    #
    #           - message = The chat the player is sending to the other player
    #
    #
    def __init__(self, negotiation_file: str, throw_formatting_errors: bool=False):
        #
        # Save the negotiation file location
        #
        self.negotiation_file = negotiation_file
        #
        # Whether to throw an error or not if the LLM response is ill-formatted.
        #
        self.throw_formatting_errors = throw_formatting_errors
        #
        # Initialize the negotiation as finished.
        #
        self.terminated = True
        self.item_price = None
        self.chat_log = []
        self.price_log = []
        #
        # Track whose turn it is.
        #
        # The seller starts by setting the initial price.
        #
        self.current_turn = "seller"
        #
        # Initialize the environment.
        #
        self.reset() 
    
    #
    # Reset the environment to its initial state
    #
    def reset(self):
        #
        # Read the initial state from the negotiation file.
        #
        # NOTE - SEAN: I think it's easier to make the starting state a dictionary
        #              object rather than a string, let me know if you like this change or not.
        #
        # Note: The starting state just needs to give us
        #          - item_description (str)
        #          - item_value (int)
        #
        starting_state = eval(Path(self.negotiation_file).read_text(encoding='utf-8'))
        #
        # Start with an empty chat and price log and the seller's turn to act.
        #
        starting_state['chat_log'] = []
        starting_state['price_log'] = []
        starting_state['current_turn'] = 'seller'
        #
        # Resetting the state so it is not terminated
        #
        starting_state['terminated'] = False
        #
        # Set the environment to its starting state
        #
        self.set_state(starting_state)

    #
    # Given a state, set the environment to that state.
    #
    def set_state(self, state : dict) -> None:
        #
        # Set the environment state
        #
        self.state = deepcopy(state)
        #
        # Check that the input state has the neccessary information
        #
        assert 'item_description' in self.state, "Input state is missing the item description"
        assert 'item_value' in self.state, "Input state is missing the item value"
        #
        # Set the chat log and price log
        #
        # Important note: this is chat_log and price_log store the reference to the
        #                 respective self.state lists. Any item appended to self.chat_log
        #                 or self.price_log also get appended to the lists in self.state
        #
        # NOTE - SEAN: Is this a good idea?
        #
        self.chat_log = self.state['chat_log']
        self.price_log = self.state['price_log']
        #
        # Inherit the terminated state value
        #
        self.terminated = self.state['terminated']
        #
        # Update current turn based on last message in chat log
        #
        self.current_turn = self.state['current_turn']

    #
    # The negotiation game uses a continous action space,
    # so it can not return a complete action set.
    #
    # Return an empty dictionary
    #
    def actions(self) -> dict:
        return {}

    #
    # Return - True if the game has terminated.
    #        - False otherwise.
    #
    # The negotiation game terminates if any of the two conditions are met
    #
    #   * The players agree on a price
    #
    #   * One of the player's terminates the chat
    #
    def is_terminal(self):
        return self.terminated
    
    #
    # Apply the given action in the environment.
    # return the resulting next state and the reward gained.
    #
    # The Negotiation game expects an action in the form of a dictionary
    # where 'price' is an integer price offer and 'message' is a chat message
    # to be sent to the other player.
    #
    # Decoding the price:
    #
    #   * price = -2 --> reject the offer and terminate the game
    #
    #   * price = -1 --> accept the offer and terminate the game
    #
    #   * price >= 0 --> counter offer to the other player
    #
    def act(self, action : dict[int, str]) -> tuple:
        #
        # Unpack the given action
        #
        price, message = action['price'], action['message']
        #
        # Verify the environment state and action
        #
        assert not self.terminated, "Trying to act in a terminated chat."
        assert (price >= -2), f"Price must be greater than -2, got {price}"
        #
        # Special case: The agent offers the same price as what they were
        #               just offered.
        #
        #               Default to accept action.
        #
        if len(self.price_log) > 0 and price == self.price_log[-1]:
            price = -1
        #
        # Game terminates if the player accepts or rejects the offer.
        #
        # The game will also terminate if a player offers what the other
        # player is offering.
        #
        if price == -1 or price == -2:
            #
            # Then set the terminated flag to true
            #
            self.terminated = True
            self.state['terminated'] = True
            #
            # Compute the reward
            #
            # If a sale occurs, then each player is rewarded equal to their profit.
            #   
            #   seller_reward = sale_price - object_value
            #
            #   buyer_reward = object_value - sale_price
            #
            # Note: this is a zero-sum game, seller_reward = -1 * buyer_reward
            #
            reward = self.calculate_reward(price == -1)
        #
        # Otherwise, the player made a counter offer and the negotiation is still active
        #
        else:
            #
            # Update the chat log with the new message
            #
            self.chat_log.append((self.current_turn, message))
            #
            # Update the price log with the new price
            #
            self.price_log.append(price)
            #
            # If no sale has been made yet, then neither player is rewarded.
            #
            reward = 0
            #
            # Switch turns if not terminated
            #
            self.current_turn = "seller" if self.current_turn == "buyer" else "buyer"
            self.state['current_turn'] = self.current_turn
        #
        # Return the updated state and reward
        #
        return deepcopy(self.state), reward
    
    #
    # Calculate the reward for the given action.
    #
    #   sale_result = True if the sale occurs
    #
    #   sale_result = False if the chat terminates without a sale
    #
    def calculate_reward(self, sale_result: bool) -> int:
        #
        # The player's decided not to do a deal.
        #
        # Neither player is punished or rewarded.
        #
        if not sale_result:
            return 0
        #
        # The sale's price is the last price that was offered before
        # the deal was accepted.
        #
        sale_price = self.price_log[-1]
        #
        # The player's reward is their profit (or loss) on the sale
        #
        seller_profit = sale_price - self.state['item_value']
        if self.current_turn == 'seller':
            return seller_profit
        #
        # Zero-sum game so the buyer's profit is -1 * the seller's profit
        #
        else:
            return -1 * seller_profit

    #
    # Return a string containing a detailed description of the negotiation state.
    #
    def describe_state(self) -> str:
        #
        # String to return
        #
        res = ''
        #
        # Start with the LLM's role
        #
        res += f'You are the {self.current_turn}\n\n'
        #
        # Add the item description
        #
        res += f"Seller's item description:\n"
        res += self.state['item_description']
        res += '\n\n'
        #
        # If this is the start of the negotation, then say so.
        #
        if len(self.chat_log) == 0:
            res += 'No offers or messages have been sent yet.\n'
        #
        # Else, list the messages
        #
        else:
            res += 'Logs in chronological order:\n\n'
            for idx in range(len(self.chat_log)):
                #
                # Unpack values
                #
                player, message = self.chat_log[idx]
                offer = self.price_log[idx]
                #
                # Describe the action taken by the player at this step 
                #
                action = 'buy' if player == 'buyer' else 'sell'
                res += f'The {player} offered to {action} for ${offer}.\n'
                res += f'And sent the message: {message}'
                #
                # Add trailing space if needed.
                #
                if idx != len(self.chat_log) - 1:
                    res += '\n\n'
        #
        # Return the description
        #
        return res

    #
    # Because the state contains the chat and price logs, the state
    # description already holds the trajectory information.
    #
    # So here we can just return the state description.
    #
    def describe_trajectory(self, trajectory: list[tuple[dict, dict, int]]) -> str:
        #
        # Start by getting the description of the game in the lead up
        # to the final action.
        #
        final_state = trajectory[-1][0]
        self.set_state(final_state)
        res = self.describe_state()
        #
        # Describe the final action and reward
        #
        player = final_state['current_turn']
        action = trajectory[-1][1]
        reward = trajectory[-1][2]
        #
        # Describe if the offer is accepted.
        #
        if action == -1:
            #
            # Determine which players won or lost the deal.
            #
            profit_or_loss = 'profits' if reward >= 0 else 'loses'
            other_player = 'buyer' if player == 'seller' else 'seller'
            other_profit_or_loss = 'profits' if reward <= 0 else 'loses'
            #
            # Add the resulting profits
            #
            res += f'The {player} accepts the offer and {profit_or_loss} ${reward}.\n'
            res += f'The {other_player} {other_profit_or_loss} ${-1*reward}.'
        #
        # Describe if the offer is rejected.
        #
        elif action == -2:
            res += f'The {player} rejects the deal and terminates the chat.\n'
            res += 'Neither person profited or lost money.'
        #
        # Otherwise, the maximum trajectory lenth was reached.
        #
        else:
            res += 'Too many messages were sent. The other person lost interest and never responded back.'
        #
        # Return the trajectory description
        #
        return res
    
    #
    # Default the price to free and send an empty message.
    #
    # Note: this function is called when the LLM provides an invalid format.
    #
    # NOTE - Should maybe be changed to a random price centered around
    #        the item price with some standard deviation?
    #
    def get_random_action(self) -> tuple[int, str]:
        return  {'price': 0, 'message': ''}, ''
    
    #
    # The expected response pattern from the policy LLM.
    #
    policy_ptrn = r"""Your decision:\s*(.*?)\s*\nMessage:\s*(.*?)\s*\nReason:\s*(.*)"""
    
    #
    # Extract the selected action from the LLM response text.
    #
    # Actions in the negotiation game are dictionaries with two entries:
    #
    #   action = {
    #       'price': (int) offer price,
    #       'message': (str) chat message to the other player 
    #   }
    #
    # Raise an error if the action isn't found or if the extracted
    # action is not in the given actions dictionary.
    #
    def extract_action_from_response(self, response: str) -> int:
        #
        # Response must contain this pattern
        #
        policy_match = re.search(NegotiationEnv.policy_ptrn, response, re.DOTALL)
        if policy_match:
            #
            # Success case - match found.
            #
            price = int(float(policy_match.group(1).strip()))
            message = str(policy_match.group(2).strip())
        else:
            #
            # Failure case - no match found, raise a value error or pick a random action.
            #
            message_str = f"Missing action. Policy LLM returned an ill-formatted response. Response:\n'{response}'"
            if self.throw_formatting_errors:
                raise ValueError(message_str)
            else:
                price, _ = self.get_random_action()
                message = ''
                print('WARNING: ' + message_str)
        #
        # Check that the price is valid.
        #
        # If not, either raise an error or pick a random action.
        #
        if price < -2:
            message_str = f"Policy LLM selected an invalid action. Got {price}. \n Response: {response}"
            if self.throw_formatting_errors:
                raise ValueError(message_str)
            else:
                price, _ = self.get_random_action()
                message = ''
                print('WARNING: ' + message_str)
        #
        # Return the 
        #
        return {'price': price, 'message': message}

    #
    # Extract the reasoning from the LLM response text.
    #
    # Raise an error if the reasoning isn't found.
    #
    def extract_reason_from_response(self, response: str) -> str:
        #
        # Response must contain this pattern
        #
        match = re.search(NegotiationEnv.policy_ptrn, response, re.DOTALL)

        if match:
            #
            # Success case - match found, extract the reasoning string.
            #
            reason =  str(match.group(3).strip())
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