# External imports
import numpy as np
from pathlib import Path
import re

# Internal imports
from envs.environment import Environment

class MazeEnv(Environment):
    #
    # board_file - text file containing the starting state for the maze
    #
    # Example:
    #
    #    #####
    #    #o  #
    #    #   #
    #    #  x#
    #    #####
    #
    #     - The player 'o' starts in square (1, 1).
    #
    #     - The goal 'x' is in the square (3, 3).
    #
    #     - The walls '#' define the boundaries.
    #
    def __init__(self, board_file: str, throw_formatting_errors: bool=False):
        #
        # Save the board file location
        #
        self.board_file = board_file
        #
        # Raise errors if the LLM response is ill-formatted.
        #
        self.throw_formatting_errors = throw_formatting_errors
        #
        # Initialize the game as finished.
        #
        self.terminated = True
        #
        # Set of actions
        #
        self.action_set = {0: 'move up', 1: 'move down', 2: 'move left', 3: 'move right'}
        #
        # Movements associated with each action
        #
        self.movements = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        #
        # Set the characters that represent the player, goal, and walls.
        #
        self.player_char = 'o'
        self.goal_char = 'x'
        self.wall_char = '#'
        self.space_char = ' '
        #
        # Initialize the player and goal positions
        #
        self.player_position = None
        self.goal_position = None
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
        starting_state = Path(self.board_file).read_text(encoding='utf-8')
        #
        # Set the environment to its starting state
        #
        self.set_state(starting_state)

    #
    # Given a state, set the environment to that state.
    #
    def set_state(self, state : str) -> None:
        #
        # Set the environment state
        #
        self.state = state
        #
        # Build the grid
        #
        self.state_to_grid()
        self.add_grid_labels()
        self.grid_to_state() # Save the grid labels in the state.
        #
        # Reset the player and goal positions
        #
        self.player_position = None
        self.goal_position = None
        #
        # Verify the grid is well-formatted and extract the player and
        # goal positions.
        #
        for i in range(1, len(self.grid)):
            for j in range(1, len(self.grid[i])):
                #
                # Check for the player
                #
                if self.grid[i][j] == self.player_char:
                    assert self.player_position is None, "More than one player found in the input board."
                    self.player_position = (i, j)
                #
                # Check for the goal
                #
                elif self.grid[i][j] == self.goal_char:
                    assert self.goal_position is None, "More than one goal found in the input board."
                    self.goal_position = (i, j)
                #
                # Check for invalid chars.
                #
                elif self.grid[i][j] != self.space_char and self.grid[i][j] != self.wall_char:
                    raise ValueError(f"Invalid character in input board: {self.grid[i][j]}")
        #
        # Check that a player and goal position were found
        #
        assert self.player_position is not None, "Player not found in input board."
        assert self.goal_char is not None, "Goal not found in input board."
        #
        # Maze is not over since we are reseting to a state.
        #
        self.terminated = False

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
    # Apply the given action in the environment,
    # return the resulting next state and the reward gained.
    #
    # Note: If the given action moves the player into a wall or out-of-bounds, then
    #       it is treated as no action and the player stays in place.
    #
    def act(self, action_id : int) -> tuple[str, int]:
        #
        # Check that the game is not over.
        #
        assert not self.terminated, "Trying to act in a terminated game."
        #
        # Check that the action id is valid
        #
        assert action_id in self.action_set.keys(), f"Unrecognized action id: {action_id}"
        #
        # Get the position the player is trying to move into.
        #
        new_player_position = (self.player_position[0]+self.movements[action_id][0],
                               self.player_position[1]+self.movements[action_id][1])
        #
        # Check that the new position is valid.
        #
        if (
            (new_player_position[0] < 0 or new_player_position[0] >= len(self.grid)) or
            (new_player_position[1] < 0 or new_player_position[1] >= len(self.grid[new_player_position[0]])) or
            (self.grid[new_player_position[0]][new_player_position[1]] == self.wall_char)
        ):
            new_player_position = self.player_position # Default to a null action
        #
        # Update the grid
        #
        self.grid[new_player_position[0]][new_player_position[1]] = self.player_char
        if new_player_position != self.player_position:
            self.grid[self.player_position[0]][self.player_position[1]] = self.space_char
        self.player_position = new_player_position
        #
        # Update the state
        #
        self.grid_to_state()
        #
        # Check if the player reached the goal
        #
        reward = int(self.player_position == self.goal_position)
        if reward == 1:
            self.terminated = True
        #
        # Return the new state and the observed reward
        #
        return self.state, reward

    #
    # Use the current state representation to form the grid. str -> list[list[str]].
    #
    def state_to_grid(self):
        #
        # Build a grid using a 2d list.
        #
        # Example:
        #
        #   self.grid = [['#', '#', '#', '#', '#'],
        #                ['#', 'o', ' ', ' ', '#'],
        #                ['#', ' ', ' ', ' ', '#'],
        #                ['#', ' ', ' ', 'x', '#'],
        #                ['#', '#', '#', '#', '#']]
        #
        self.grid = [list(line) for line in self.state.split('\n')]

    #
    # Use the current grid to form the state. list[list[str]] -> str.
    #
    def grid_to_state(self):
        self.state = ''.join([''.join(line)+'\n' for line in self.grid])
        self.state = self.state[:-1]
    
    #
    # Add row and column lables to the grid
    #
    # Before:
    #
    # self.grid = [['#', '#', '#', '#', '#'],
    #              ['#', 'o', ' ', ' ', '#'],
    #              ['#', ' ', ' ', ' ', '#'],
    #              ['#', ' ', ' ', 'x', '#'],
    #              ['#', '#', '#', '#', '#']]
    #
    # After:
    #
    # self.grid = [[' ', '0', '1', '2', '3', '4'],
    #              ['0', '#', '#', '#', '#', '#'],
    #              ['1', '#', 'o', ' ', ' ', '#'],
    #              ['2', '#', ' ', ' ', ' ', '#'],
    #              ['3', '#', ' ', ' ', 'x', '#'],
    #              ['4', '#', '#', '#', '#', '#']]
    #
    def add_grid_labels(self) -> None:
        #
        # Return if the grid is already labeled.
        #
        if self.grid[0][1] == '0' and self.grid[1][0] == '0':
            return
        #
        # Label the rows
        #
        grid_width = 0 # Longest row in the grid
        for row_id in range(len(self.grid)):
            grid_width = max(grid_width, len(self.grid[row_id]))
            self.grid[row_id] = [str(row_id)] + self.grid[row_id]
        #
        # Label the columns
        #
        columns = [' '] + [str(col_id) for col_id in range(grid_width)]
        self.grid = [columns] + self.grid
    
    #
    # Return a string containing a detailed description of the game state.
    #
    def describe_state(self) -> str:
        #
        # String to be returned
        #
        res = ''
        #
        # Add the player's position on the board
        #
        py, px = self.player_position[0]-1, self.player_position[1]-1
        res += f'The player is at position ({px}, {py}).\n'
        #
        # Add the goal's position on the board
        #
        gy, gx = self.goal_position[0]-1, self.goal_position[1]-1
        res += f'The goal is at position ({gx}, {gy}).\n\n'
        #
        # Add the maze board
        #
        res += f'Maze grid:\n{self.state}\n\n'
        #
        # Add a description of the player's surroundings
        #
        char_word = {self.goal_char: 'the goal.', self.wall_char: 'a wall', self.space_char: 'a free space.'}
        res += f'Above the player at position ({px}, {py-1}) is {char_word[self.grid[py][px+1]]}\n'
        res += f'Below the player at position ({px}, {py+1}) is {char_word[self.grid[py+2][px+1]]}\n'
        res += f'Right of the player at position ({px+1}, {py}) is {char_word[self.grid[py+1][px+2]]}\n'
        res += f'Left of the player at position ({px-1}, {py}) is {char_word[self.grid[py+1][px]]}\n'
        #
        # Return the result
        #
        return res

    #
    # Given a trajectory, return a string containing a detailed description of the trajectory.
    #
    def describe_trajectory(self, trajectory: list[tuple[str, int, int]]) -> str:
        #
        # String to be returned
        #
        res = ''
        #
        # Set the environment to the start of the trajectory
        #
        initial_state, initial_action, _ = trajectory[0]
        self.set_state(initial_state)
        #
        # Start by describing the starting point.
        #
        py, px = self.player_position[0]-1, self.player_position[1]-1
        res += f'The player starts at position ({px}, {py}) and takes action {initial_action} ({self.action_set[initial_action]})'
        #
        # Describe the subsequent actions
        #
        for state, action, reward in trajectory[1:]:
            #
            # Update the environment to the next state 
            #
            self.set_state(state)
            #
            # Get the updated player coordinates
            #
            new_py, new_px = self.player_position[0]-1, self.player_position[1]-1
            #
            # Record if the player hit a wall.
            #
            if new_px == px and new_py == py:
                res += ' causing the player to hit a wall and stay in place.\n'
            #
            # Otherwise, record the new player location.
            #
            else:
                res += f' moving the player to position ({px}, {py}).\n'
            #
            # Record the next action.
            #
            res += f'Then, the player takes action {action} ({self.action_set[action]})'
        #
        # Record whether or not the goal was reached.
        #
        if trajectory[-1][2] != 1:
            res += f' causing the player to run out of moves.\n'
        else:
            res += f' causing the player to reach the goal!'
        #
        # Return the resulting string
        #
        return res
    
    #
    # Extract the selected action from the LLM response text.
    #
    # Raise an error if the action isn't found or if the extracted
    # action is not in the given actions dictionary.
    #
    def extract_action_from_response(self, response: str) -> int:
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
            message_str = f"Missing action. Policy LLM returned an ill-formatted response. Response:\n'{response}'"
            if self.throw_formatting_errors:
                raise ValueError(message_str)
            else:
                action, _ = self.get_random_action()
                print('WARNING: ' + message_str)
        #
        # Check that the found action id is valid.
        #
        # If not, either raise an error or pick a random action.
        #
        if action not in self.action_set.keys():
            message_str = f"Policy LLM selected an invalid action. Got {action}. Expected one of these {self.action_set}\n Response: {response}"
            if self.throw_formatting_errors:
                raise ValueError(message_str)
            else:
                action, _ = self.get_random_action()
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
    # Given a list of possible actions, select one randomly and give an empty reason.
    #
    def get_random_action(self) -> tuple[int, str]:
        return np.random.choice(list(self.action_set.keys())), ''