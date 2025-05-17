# External imports
from collections import Counter
from copy import deepcopy
import json
import numpy as  np
from pathlib import Path
import time

# Internal imports
from envs.environment import Environment
from agents.improvement import ImprovementOperator
from agents.language_policy import LanguagePolicy
from agents.language_value_function import LanguageValueFunction
from models.model import LanguageModel

class ActorCriticAgent:

    def __init__(self, env : Environment, llm : LanguageModel, agent_config_file : str):
        #
        # Environment
        #
        self.env = env
        #
        # Read agent configuration file
        #
        with open(agent_config_file, 'r') as file:
            self.config = json.load(file)
        #
        # Set the initial starting action temperature parameters
        #
        self.initial_action_temp = self.config['init_action_temp']
        self.temp_factor = self.config['temp_factor'] # Decay the temp by this factor every train iteration
        #
        # Store a reference to the model
        #
        self.llm = llm
        #
        # Core components
        #
        self.lang_policy = LanguagePolicy(llm, 
                                          self.config['policy_config'],
                                          self.config['throw_formatting_errors']
        )
        self.lang_values = LanguageValueFunction(llm, 
                                                 self.config['values_config'],
                                                 self.config['throw_formatting_errors']
        )
        self.improvement_op = ImprovementOperator(llm, 
                                                  self.config['improvement_config'],
                                                  self.config['throw_formatting_errors']
        )
        #
        # Statistics
        #
        self.avg_traj_len = [] # per loop
        self.avg_reward_per_step = [] # per loop
    
    #
    # Main actor-critic training loop
    #
    def train(self, T,  # Number of training iterations 
                    N,  # Number of trajectories per iteration
                    K): # Number of Monte-Carlo trajectories
        #
        # Training hyperparameters
        #
        #   - N_ACTION_SAMPLES - num. of actions to sample to estimate policy action probability
        #
        #   - TOP_N_ACTIONS - select top N most probable actions to perform the policy update.
        #                   - 'all' selects all possible actions in the given state.
        #
        #   - VALUE_BATCH_SIZE - num. of value targets to use per training iteration
        #
        #   - POLICY_BATCH_SIZE - num. of policy targets to use per training iteration
        #
        #   - KEEP_N_ITER_HISTORY - num. of training iterations until a target is evicted from its buffer.
        #
        #
        N_ACTION_SAMPLES = self.config['num_action_samples']
        TOP_N_ACTIONS = self.config['top_n_actions']
        VALUE_BATCH_SIZE = 'all'
        POLICY_BATCH_SIZE = 'all'
        KEEP_N_ITER_HISTORY = 0
        action_temp = self.initial_action_temp
        #
        # Reset agent statistics
        #
        self.reset_stats()
        #
        # Store value targets and policy targets
        #
        value_buffer = []  # [(train_idx, (s, a, v)), ...]
        policy_buffer = [] # [((train_idx, (s, policy target, strategic reasoning)), ...]
        #
        # Main training loop
        #
        for train_idx in range(T): # Main training loop
            start_time = time.time()
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            print('STEP 0: MEASURE MODEL PERFORMANCE', flush=True)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            #
            # Collect a trajectory with zero temperature to evaluate performance without randomness.
            #
            self.env.reset()
            eval_trajectory = self.rollout([self.env], action_temp=0.)
            #
            # Update stats
            #
            self.update_stats(eval_trajectory)
            self.print_stat_summary()
            #
            # Collect trajectories
            #
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            print('STEP 1: COLLECT TRAJECTORIES', flush=True)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            #
            # Log the action sampling temperature
            #
            print('Sampling actions with temperature:', action_temp)
            #
            # Reset game to its starting position
            #
            self.env.reset()
            #
            # Create N copies of the environment
            #
            envs = [deepcopy(self.env) for _ in range(N)]
            #
            # Rollout each enironment and collect the trajectories.
            #
            trajectories = self.rollout(envs, action_temp=action_temp) # [[(s, a, r), ..], ...]            
            #
            # Log per step runtime
            #
            print(f'STEP 1: runtime={round(time.time()-start_time, 1)} sec', flush=True)
            #
            # Build value estimation targets
            #
            #    Compute value estimates for each state-action
            #    pair that was observed during rollouts.
            #
            start_time = time.time()
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            print('STEP 2: COMPUTE VALUE TARGETS', flush=True)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            value_targets = [] # [(s, a, v), ...]
            #
            # For Monte-Carlo estimates, we evaluate the state-action
            # pair using a set of sample trajectories.
            #
            for trajectory in trajectories:
                #
                # Sample K sample trajectories per state-action pair in this trajectory.
                #
                sample_trajectories = [[] for _ in range(len(trajectory))]
                for _ in range(K):
                    #
                    # Start with a fresh list of environments
                    #
                    envs = [deepcopy(self.env) for _ in range(len(trajectory))]
                    #
                    # Set each environment to a state visited in the trajectory
                    #
                    # Progress each environment's state using the associated action
                    # that was taken at that step.
                    #
                    # Record the reward recieved from this action.
                    #
                    sa_pairs, trajectory_seeds, action_sets = [], [], []
                    for step, env in enumerate(envs):
                        #
                        # Get the state and action that was taken at this step.
                        #
                        state, action, _ = trajectory[step]
                        #
                        # Save the state action pair for this transition.
                        #
                        env.set_state(state)
                        state_description = env.describe_state()
                        sa_pairs.append((state_description, action))
                        #
                        # Set the state
                        #
                        env.set_state(state)
                        #
                        # Save the set of valid actions in this state
                        #
                        action_sets.append(env.actions())
                        #
                        # Take the associated action and get the reward
                        #
                        _, reward = env.act(action)
                        #
                        # Add this transition (state, action, reward)
                        # as the first transition for each sampled trajectory.
                        #
                        trajectory_seeds.append([(state, action, reward)])
                    #
                    # Rollout the environments
                    #
                    rollouts = self.rollout(envs, action_temp=action_temp)
                    #
                    # Add the first transition to the front of these rollouts to
                    # get the full sample trajectory.
                    #
                    # Then add the full sampled trajectory to the sample trajectory list.
                    #
                    for sa_idx in range(len(trajectory)):
                        sample_trajectories[sa_idx].append(trajectory_seeds[sa_idx] + rollouts[sa_idx])
                #
                # Get the description for each trajectory
                #
                sample_traj_descriptions = [[self.env.describe_trajectory(sample_trajectories[i][j]) for j in range(len(sample_trajectories[i]))] for i in range(len(sample_trajectories))]
                #
                # Given the example trajectories, evaluate how good or bad
                # taking each action is in the given states.
                #
                values = self.lang_values.mc_estimate(sa_pairs,
                                                      action_sets, 
                                                      sample_traj_descriptions)
                #
                # Save the result.
                #
                for step in range(len(trajectory)):
                    value_targets.append(
                        # (train idx, (state, action, value))
                        (train_idx, (sa_pairs[step][0], sa_pairs[step][1], values[step]))
                    )
            #
            # Save the value targets from this training iteration to the value buffer.
            #
            value_buffer += value_targets
            #
            # Log per step runtime
            #
            print(f'STEP 2: runtime={time.time()-start_time} sec', flush=True)
            start_time = time.time()
            #
            # Update the value function using the value targets
            #
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            print('STEP 3: TRAIN VALUE MODEL', flush=True)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            if VALUE_BATCH_SIZE == 'all':
                value_targets_batch = [value_buffer[idx][1] for idx in range(len(value_buffer))]
            else:
                sample_idxs = np.random.choice(range(len(value_buffer)), size=VALUE_BATCH_SIZE, replace=False)
                value_targets_batch = [value_buffer[idx][1] for idx in sample_idxs]
            self.lang_values.update(value_targets_batch, self.env.actions())
            #
            # Log the per step runtime
            #
            print(f'STEP 3: runtime={time.time()-start_time} sec', flush=True)
            start_time = time.time()
            #
            # Use the updated value function to improve the policy
            #
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            print('STEP 4: COMPUTE POLICY TARGETS')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            for trajectory in trajectories:
                for transition in trajectory:
                    #
                    # For each state in the trajectory...
                    #
                    state = transition[0]
                    #
                    # Set the environment to this state
                    #
                    self.env.set_state(state)
                    #
                    # Get the set of available actions in this state.
                    #
                    # Note: if env has a continuous action set space, then
                    #       all_actions will be an empty dictionary.
                    #
                    all_actions = self.env.actions()
                    #
                    # Get the text description of the state.
                    #
                    state_description = self.env.describe_state() 
                    #
                    # Get a set of actions for the policy improvement operator to consider
                    # for this state.
                    #
                    if N_ACTION_SAMPLES == 'all':
                        #
                        # Consider all available actions
                        #
                        actions = list(all_actions.keys())
                    else:
                        #
                        # Make a list of the state description and action set.
                        #
                        # Note: This is neccessary for prompt batching.
                        #
                        state_descriptions = [state_description] * N_ACTION_SAMPLES
                        action_sets = [self.env.actions()] * N_ACTION_SAMPLES
                        #
                        # Sample actions from the policy to estimate
                        # action probabilities.
                        #
                        # Recall - the language policy does not directly assign a probability 
                        #          distribution over the action space.
                        #
                        policy_responses = self.lang_policy.get_action(state_descriptions, action_sets)
                        #
                        # Extract the actions from the responses from the policy LLM.
                        #
                        sampled_actions = [
                            self.env.extract_action_from_response(response)
                            for response in policy_responses 
                        ]
                        #
                        # Get the top N most frequent actions
                        #
                        # Unless 'all' is set, then use all the sampled actions.
                        #
                        if TOP_N_ACTIONS == 'all':
                            actions = sampled_actions
                        else:
                            actions = list(Counter(sampled_actions).most_common(TOP_N_ACTIONS))
                    #
                    # Get the value estimates for each action in this state
                    #
                    state_descriptions = [state_description] * len(actions)
                    action_sets = [self.env.actions()] * len(actions)
                    values = self.lang_values.get_value(state_descriptions, actions, action_sets)
                    #
                    # Query the language improvement operator to get strategic reasoning text and
                    # a policy target.
                    #
                    policy_target = self.improvement_op.reason(state_description, actions, values, self.env.actions())
                    #
                    # Verify the reasoning response was formatted properly.
                    #
                    action = self.env.extract_action_from_response(policy_target)
                    reason = self.env.extract_reason_from_response(policy_target)
                    #
                    # Store the policy target triplet in the policy buffer.
                    #
                    # Only train on policy targets with non-empty reasoning.
                    #
                    if reason != '':
                        policy_buffer.append((train_idx, (state, all_actions, policy_target)))
            #
            # Log per step runtime
            #
            print(f'STEP 4: runtime={time.time()-start_time} sec', flush=True)
            start_time = time.time()
            #
            # Update the policy using the policy targets
            #
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            print('STEP 5: TRAIN POLICY MODEL')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', flush=True)
            if POLICY_BATCH_SIZE == 'all':
                policy_targets_batch = [policy_buffer[idx][1] for idx in range(len(policy_buffer))]
            else:
                sample_idxs = np.random.choice(range(len(policy_buffer)), size=POLICY_BATCH_SIZE, replace=False)
                policy_targets_batch = [policy_buffer[idx][1] for idx in sample_idxs]
            self.lang_policy.update(policy_targets_batch)
            #
            # Evict old targets
            #
            threshold = train_idx - KEEP_N_ITER_HISTORY
            value_buffer = [target for target in value_buffer if target[0] >= threshold]
            policy_buffer = [target for target in policy_buffer if target[0] >= threshold]
            #
            # Decay the action sampling temperature for the next iteration
            #
            action_temp *= self.temp_factor
            #
            # Save the model
            #
            self.llm.save()
            #
            # Log per step runtime
            #
            print(f'STEP 5: runtime={time.time()-start_time}', flush=True)
        #
        # Print training stat summary
        #
        self.print_stat_summary()

    #
    # Use the agent's policy to rollout the environment
    # state to completion. Return the observed trajectory.
    #
    def rollout(self, envs: list[Environment], max_trajectory_length=5, action_temp: float=0.) -> list[tuple[str, int, str]]:
        #
        # Store the observed transitions
        #
        trajectories = [[] for _ in range(len(envs))]
        #
        # Number of steps taken thus far.
        #
        length = 0
        #
        # Continue to act until the game terminates.
        # Or we hit the maximum allowed trajectory length.
        #
        while not all(env.is_terminal() for env in envs) and length < max_trajectory_length:
            #
            # Get indicies of active environments
            #
            active_idxs = [idx for idx, env in enumerate(envs) if not env.is_terminal()]
            #
            # Get the set of actions available in the current states
            #
            action_sets = [envs[i].actions() for i in active_idxs]
            #
            # Select an action for each active environment
            #
            current_states = [deepcopy(envs[i].state) for i in active_idxs]
            state_descriptions = [envs[i].describe_state() for i in active_idxs]
            #
            # Query the policy LLM
            #
            responses = self.lang_policy.get_action(state_descriptions, action_sets, temp=action_temp)
            #
            # Extract the action and reasoning from each response
            #
            actions, reasons = [], []
            for idx, env_idx in enumerate(active_idxs):
                actions.append(envs[env_idx].extract_action_from_response(responses[idx]))
                reasons.append(envs[env_idx].extract_reason_from_response(responses[idx]))
            #
            # Apply the actions to the environments to collect the
            # rewards and the next states.
            #
            rewards = [envs[env_idx].act(actions[idx])[1] 
                            for idx, env_idx in enumerate(active_idxs)] 
            #
            # Add each transition to its trajectory
            #
            for idx, env_idx in enumerate(active_idxs):
                trajectories[env_idx].append((current_states[idx], actions[idx], rewards[idx]))
            #
            # Increment the length counter
            #
            length += 1
        #
        # Return the observed trajectory
        #
        return trajectories
    
    #
    # Resets the statistics counters
    #
    def reset_stats(self):
        self.avg_traj_len = []
        self.avg_reward_per_step = []

    #
    # Use the given trajectores to update the agent's stats
    #
    def update_stats(self, trajectories: list[tuple[str, int, int]]) -> None:
        #
        # Count avg trajectory length
        #
        self.avg_traj_len.append(np.mean([len(trajectory) for trajectory in trajectories]))
        #
        # Count avg reward per step
        #
        cumm_reward, n_steps = 0, 0
        for traj in trajectories:
            for _, _, reward in traj:
                cumm_reward += reward
                n_steps += 1
        self.avg_reward_per_step.append(cumm_reward / n_steps)
    
    #
    # Print a summary of the stats collected during training.
    #
    def print_stat_summary(self):
        #
        # For each training iteration, print all the stats.
        #
        for it in range(len(self.avg_traj_len)): 
            print('-------------------------------')
            print(f'Train iteration #{it}')
            print(f'  * Avg. trajectory length={self.avg_traj_len[it]}')
            print(f'  * Avg. reward per step={self.avg_reward_per_step[it]}')