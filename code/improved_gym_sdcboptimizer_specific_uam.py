from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import math
import datetime

class RunwayEnv(gym.Env):
    """
    Runway Environment for reinforcement learning-based optimization of runway operations.
    
    This environment simulates a runway configuration optimization problem where an agent needs
    to decide on runway configurations and service rates based on arrival/departure queues,
    weather conditions, and wind states.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, alpha, beta, delta, eta, 
                 data, taf_data, arrival_data,
                 congested_days, u, u_max, v_min, v_max,
                 possible_actions, valid_bool,
                 wind_states, 
                 observation_low, observation_high,
                 p_wc_trans, p_ws_trans,
                 p_wc_error, p_ws_error,
                 chosen_day=None, render_mode=None):
        """
        Initialize the runway environment.
        
        Parameters:
        -----------
        alpha : float
            Weight for arrival queue in reward function
        delta : float
            Weight for configuration switch cost in reward function
        taf_data : pandas.DataFrame
            Training data with forecast information
        data : pandas.DataFrame
            Test data
        congested_days : list
            List of congested days to select from
        u : dict
            Arrival service rates for different states
        v_min : dict
            Minimum departure service rates for different states (lower bounds)
        v_max : dict
            Maximum departure service rates for different states (upper bounds)
        possible_actions : list
            List of all possible actions
        wind_states : list
            Available runway configurations for each wind state
        observation_low : list
            Lower bounds for observation space
        observation_high : list
            Upper bounds for observation space
        p_wc_trans, p_ws_trans : numpy.ndarray
            Transition probabilities for weather condition and wind state
        p_wc_error, p_ws_error : numpy.ndarray
            Error probabilities for weather condition and wind state
        chosen_day : str, optional
            Specific day to simulate, or random if None
        render_mode : str, optional
            Rendering mode, one of None, "human", or "rgb_array"
        """
        self._init_rendering(render_mode = render_mode)
        self.window_size = 512  # Size of PyGame window
        
        # For storing results
        self.results = {}
        
        # Transition probabilities
        self.p_wc_trans = p_wc_trans.copy()
        self.p_wc_error = p_wc_error.copy()
        self.p_ws_trans = p_ws_trans.copy()
        self.p_ws_error = p_ws_error.copy()
        
        # Data
        self.data = data.copy()
        self.taf_data = taf_data.copy()
        self.arrival_data = arrival_data.copy()
        self.days = congested_days
        
        # Service rates and action definitions
        self.u = u.copy()  # Arrival service rates
        self.u_max = u_max
        self.v_min = v_min.copy()  # Minimum departure service rates (lower bounds)
        self.v_max = v_max.copy()  # Maximum departure service rates (upper bounds)
        # Extract global maximum for queue limiting purposes
        self.v_max_global = max([np.max(v_max[key]) for key in v_max if len(v_max[key]) > 0] or [0])
        self.possible_actions = possible_actions.copy()
        self.valid_bool = valid_bool.copy()
        self.wind_states = wind_states.copy()
        
        # Reward function parameters
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.eta = eta
        
        # Select day if not provided
        self.day = self._select_day(chosen_day)
        
        # Define observation and action spaces
        self._define_spaces(observation_low, observation_high)
        
        # Initialize rendering variables
        self._init_rendering(render_mode)
    
    def _select_day(self, chosen_day):
        """Select a day for simulation, either randomly or as specified."""
        if chosen_day is None:
            chosen_day = np.random.choice(self.days)
        print(f'Day {chosen_day} chosen.\n')
        return chosen_day
    
    def _define_spaces(self, observation_low, observation_high):
        """Define observation and action spaces."""
        # Observation space
        observation_low = np.array(observation_low, dtype=np.float32)
        observation_high = np.array(observation_high, dtype=np.float32)
        self.observation_space = spaces.Box(observation_low, observation_high)

        # Action space
        self.action_space = spaces.Discrete(len(self.possible_actions))
    
    def _heuristic_tactical_action(self, obs):
        """
        Generate a tactical action using a heuristic policy.
        
        The heuristic:
        1. If the previous vertiport side is available for the current wind state, keep it;
           otherwise select the available side with the highest arrival capacity.
        2. Serve the largest arrival count u <= min(arrival queue, max arrival capacity)
           whose minimum departure requirement v_min[u] is met by the departure queue (Eq. 12).
        3. Serve departures up to min(departure queue, v_max[u]) (Eq. 13).
        
        Returns:
            tuple: (config, arrivals_to_serve, departures_to_serve)
        """
        # Extract current state
        RC_i = int(obs[0])
        a_i = int(obs[1])
        d_i = int(obs[2])
        wc = int(obs[3])
        ws = int(obs[4])

        # Check available runway configurations
        available_configs = self.wind_states.get(ws, [0])

        if RC_i in available_configs:

            # Keep the previous runway configuration
            RC_f = RC_i
        
        else:

            # Select the available configuration with highest arrival capacity
            
            best_config = 0
            best_arrival_cap = 0
            for config in available_configs:
                if config == 0:
                    arrival_cap = 0
                else:
                    # Get maximum arrival capacity from RCCE
                    if (wc, config) in self.u:
                        u_array = self.u[wc, config]
                        arrival_cap = int(np.max(u_array)) if len(u_array) > 0 else 0
                    else:
                        arrival_cap = 0
                
                if arrival_cap > best_arrival_cap:
                    best_arrival_cap = arrival_cap
                    best_config = config
            
            RC_f = best_config
        
        # Determine arrivals and departures to serve
        if RC_f == 0:
            # No runway available, serve nothing
            a_served = 0
            d_served = 0
        else:
            # Get the RCCE for this configuration and weather condition
            u_vals = self.u[wc, RC_f]
            v_min_vals = self.v_min[wc, RC_f]  # Lower bound departures (indexed by arrival count)
            v_max_vals = self.v_max[wc, RC_f]  # Upper bound departures (indexed by arrival count)
            
            # Find MAXIMUM FEASIBLE arrival count
            # A solution with 'a' arrivals is feasible if:
            #   1. a <= a_i (arrivals in queue)
            #   2. a <= max(u_vals) (arrival capacity allows it)
            #   3. v_min[a] <= d_i (we have enough departures to satisfy minimum required)
            
            max_arr_cap = int(np.max(u_vals)) if len(u_vals) > 0 else 0
            max_possible_a = min(a_i, max_arr_cap, len(v_min_vals) - 1)
            
            # Search from high to low to find maximum feasible arrivals
            a_served = 0  # Default: serve nothing
            for a in range(max_possible_a, -1, -1):
                if int(v_min_vals[a]) <= d_i:  # Feasible: sufficient departures in queue
                    a_served = a
                    break
            
            # For this feasible arrival count, serve maximum departures
            if a_served > 0 or len(v_max_vals) > 0:
                d_cap_max = int(v_max_vals[a_served])
                d_cap_min = int(v_min_vals[a_served])
                # Serve max departures, but never exceed queue and respect minimum
                d_served = min(d_i, d_cap_max)
            else:
                d_served = 0
        
        return [RC_f, a_served, d_served]

    def _init_rendering(self, render_mode):
        """Initialize rendering-related variables."""
        # Validate render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Rendering attributes
        self.window = None
        self.clock = None

    def _get_obs(self):
        """
        Get the current observation.
        
        Returns:
        --------
        numpy.ndarray
            Current observation
        """
        return self.obs
    
    def _get_info(self):
        """
        Get additional information about the current state.
        
        Returns:
        --------
        dict
            Dictionary with additional information
        """
        return {
            'day': self.day,                  # Current day being analyzed
            'runway_configuration': self.obs[0],  # Current runway configuration
            'arrival_queue': self.obs[1],     # Current arrival queue
            'departure_queue': self.obs[2],   # Current departure queue
            'weather_condition': self.obs[3], # Current weather condition
            'wind_state': self.obs[4]        # Current wind state
        }
    
    def compute_cm_reward(self, 
                      RC_prev: int, 
                      RC_t: int, 
                      a_prev: int, 
                      d_prev: int, 
                      a_t: int, 
                      d_t: int,
                      #g_t: int,
                      evaluate:bool,
                      record:bool = True,
                      ) -> float:
        """
        Compute the reward for taking an action from the current state.

        Args:
            RC_prev: Previous runway configuration
            RC_t: New runway configuration
            a_prev: Previous arrival queue
            d_prev: Previous departure queue
            a_t: New arrival queue
            d_t: New departure queue
            evaluate: If True, return the cost in EUR; otherwise the (negative) training reward
            record: If True, accumulate the cost components of the episode
                    (False for the what-if replays in compute_return_reference)

        Returns:
            float: Calculated reward
        """
        # Determine if runway configuration changed
        config_switch = int(RC_t != RC_prev)
        if record:
            self.config_switch.append(config_switch)

        if not evaluate:

            # Calculate queue penalties (negative reward based on squared queue lengths, Eq. 21)
            queue_penalty = -(self.alpha * a_t**2 + d_t**2)
            if record:
                self.queue_cost -= queue_penalty

            # Calculate switch penalty if configuration changed
            # Only penalize switches when there are aircraft in the queue
            switch_penalty = -config_switch * self.delta * (self.alpha * a_prev**2 + d_prev**2)
            if record:
                self.switch_cost -= switch_penalty
        
        else:
            
            # Delay
            
            # # Calculate queue penalties (negative reward based on squared queue lengths)
            # queue_penalty = (a_t**2 + d_t**2)
            # self.queue_cost += queue_penalty

            # # Calculate switch penalty if configuration changed
            # # Only penalize switches when there are aircraft in the queue
            # switch_penalty = config_switch * self.delta * (a_prev**2 + d_prev**2)
            # self.switch_cost += switch_penalty
            
            # # Ground holding cost
            # ground_penalty = g_t * 15
            # self.ground_cost += ground_penalty
            
            # Cost
            
            # Calculate queue penalties (negative reward based on squared queue lengths)
            queue_penalty = (self.alpha * a_t**2 + d_t**2) * 45
            if record:
                self.queue_cost += queue_penalty

            # Calculate switch penalty if configuration changed
            # Only penalize switches when there are aircraft in the queue
            switch_penalty = config_switch * self.delta * (self.alpha * a_prev**2 + d_prev**2) * 45
            if record:
                self.switch_cost += switch_penalty

        return queue_penalty + switch_penalty

    def _generate_initial_state(self, mode='train', evaluate=False):
        """
        Generate the initial state for an episode.
        
        Parameters:
        -----------
        training : bool
            Whether the environment is in training mode
            
        Returns:
        --------
        numpy.ndarray
            Initial state
        """
             
        # Demand
        RC_i = 0
        a_i = self.day_data.loc[self.time_step, 'scheduled_arrivals_3']
        d_i = self.day_data.loc[self.time_step, 'scheduled_departures']
        wc = self.day_data.loc[self.time_step, 'weather_condition']
        ws = self.day_data.loc[self.time_step, 'wind_state']
        
        # Calculate future values safely to avoid NaN from empty slices
        future_time_slice = self.day_data.loc[(self.time_step+1):(len(self.day_data)-1), 'scheduled_arrivals_3']
        future_arr = np.sum(future_time_slice) if len(future_time_slice) > 0 else 0.0
        
        future_time_slice = self.day_data.loc[(self.time_step+1):(len(self.day_data)-1), 'scheduled_departures']
        future_dep = np.sum(future_time_slice) if len(future_time_slice) > 0 else 0.0
        
        future_time_slice = self.day_data.loc[(self.time_step+1):(len(self.day_data)-1), 'weather_condition']
        future_wc = np.mean(future_time_slice) if len(future_time_slice) > 0 else float(wc)
        
        future_time_slice = self.day_data.loc[(self.time_step+1):(len(self.day_data)-1), 'wind_state']
        future_ws = np.mean(future_time_slice) if len(future_time_slice) > 0 else float(ws)
        
        # Initial observation - ensure float32 dtype
        self.obs = np.array([
            RC_i,    # Runway configuration (0 = None, 1 = 10R_10L, 2 = 28L_28R)
            a_i,  # Arrival queue (integer)
            d_i,  # Departure queue (integer)
            wc,   # Weather condition (0 = VMC, 1 = IMC, 2 = LIMC)
            ws#,    # Wind state (0 = Both available, 1 = 10R_10L available, 2 = 28L_28R available, 3 = Both unavailable)
            # future_arr, # Future arrival demand
            # future_dep, # Future departure demand
            # future_wc,  # Future weather condition 
            # future_ws,   # Future wind state
            # self.time_step # Current time step
        ], dtype=np.float32)
        self.arr_delayed.append(0)
    
    def compute_return_reference(self, evaluate=False):
        
        # Compute return reference
        total_reward = 0
        step = 0
        
        # Initial state
        RC_i = 0
        a_i = self.day_data.loc[step, 'scheduled_arrivals_3']
        d_i = self.day_data.loc[step, 'scheduled_departures']
        wc = self.day_data.loc[step, 'weather_condition']
        ws = self.day_data.loc[step, 'wind_state']
    
        while step < len(self.day_data):
            
            # Getting actions from heuristic tactical policy
            cm_obs = np.array([RC_i, a_i, d_i, wc, ws])
            cm_action = self._heuristic_tactical_action(cm_obs)
            
            # Extract action components
            RC_f = cm_action[0]         # New runway configuration
            a_served = cm_action[1]  # Number of arrivals to process
            d_served = cm_action[2]  # Number of departures to process
            
            # Calculate new queue lengths after processing
            a_f = a_i - a_served  # New arrival queue
            d_f = d_i - d_served  # New departure queue
            
            # Ensure queues don't go negative (sanity check)
            if a_f < 0 or d_f < 0:
                print('Invalid queues detected in initial state.\n')
            
            # Calculate reward
            cm_reward = self.compute_cm_reward(RC_i, RC_f, a_i, d_i, a_f, d_f, evaluate, record=False)
            total_reward += cm_reward
            
            # Advance step
            step += 1
            
            # Next state
            if step < len(self.day_data):
                RC_i = RC_f
                a_i = a_f + self.day_data.loc[step, 'scheduled_arrivals_3']
                d_i = d_f + self.day_data.loc[step, 'scheduled_departures']
                wc = self.day_data.loc[step, 'weather_condition']
                ws = self.day_data.loc[step, 'wind_state']
        
        return total_reward
    
    def compute_reward(self, g_f, evaluate):
        
        # Calculate current return reference
        last_return = self.last_return
        total_reward = self.compute_return_reference(evaluate)
        self.last_return = total_reward
        
        if not evaluate:
            
            ground_penalty = total_reward - last_return - self.eta * g_f
            self.ground_cost -= ground_penalty
        
        else:
            
            ground_penalty = 0#total_reward - last_return + self.eta * g_f * 15 * 45
            self.ground_cost += ground_penalty
        
        return ground_penalty
    
    def generate_weather_scenario(self, mode):
        
        if mode == 'train':
            
            # Use TAF data
            self.day_data['weather_condition'] = self.day_taf_data['weather_condition'].copy()
            self.day_data['wind_state'] = self.day_taf_data['wind_state'].copy()
            
            # Sample initial weather condition and wind state
            self.day_data.loc[0, 'weather_condition'] = np.random.choice([0, 1, 2], p=self.p_wc_error[self.day_data.loc[0, 'weather_condition']])
            prev_wc = self.day_data.loc[0, 'weather_condition']
            self.day_data.loc[0, 'wind_state'] = np.random.choice([0, 1, 2, 3], p=self.p_ws_error[self.day_data.loc[0, 'wind_state']])
            prev_ws = self.day_data.loc[0, 'wind_state']
            
            for step in range(1,len(self.day_data)):
                
                # Sample next weather condition and wind state with transitions
                p_wc = self.p_wc_error[self.day_data.loc[step, 'weather_condition']] * self.p_wc_trans[prev_wc] + 1e-12
                p_wc = p_wc / np.sum(p_wc)
                p_ws = self.p_ws_error[self.day_data.loc[step, 'wind_state']] * self.p_ws_trans[prev_ws] + 1e-12
                p_ws = p_ws / np.sum(p_ws)
                self.day_data.loc[step, 'weather_condition'] = np.random.choice([0, 1, 2], p=p_wc)
                prev_wc = self.day_data.loc[step, 'weather_condition']
                self.day_data.loc[step, 'wind_state'] = np.random.choice([0, 1, 2, 3], p=p_ws)
                prev_ws = self.day_data.loc[step, 'wind_state']
            
        elif mode == 'deploy':
            
            # Use TAF data
            self.day_data['weather_condition'] = self.day_taf_data['weather_condition'].copy()
            self.day_data['wind_state'] = self.day_taf_data['wind_state'].copy()
        
        elif mode == 'test':
            
            # Use METAR data (same as deploy - deterministic weather)
            self.day_data['weather_condition'] = self.day_data['weather_condition'].copy()
            self.day_data['wind_state'] = self.day_data['wind_state'].copy()
        
        # pass
    
    def reset(self, seed=None, options=None, mode='train', evaluate=False):
        """
        Reset the environment to its initial state.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed
        options : dict, optional
            Additional options
        training : bool, optional
            Whether the environment is in training mode
            
        Returns:
        --------
        tuple
            Initial observation and info
        """
        # Seed the random number generator
        super().reset(seed=seed)
        
        # Initialize data
        self.day_data = self.data.loc[self.data['day'] == self.day].reset_index(drop=True).copy()
        self.day_taf_data = self.taf_data.loc[self.taf_data['day'] == self.day].reset_index(drop=True).copy()
        self.day_arrival_data = self.arrival_data.loc[self.arrival_data['day'] == self.day].reset_index(drop=True).copy()
        
        # Generate stochastic weather scenario
        self.generate_weather_scenario(mode)
        
        # Initialize time step
        self.flight = 0
        self.flight_time_step = self.day_arrival_data.loc[self.flight, 'day_slot']
        self.time_step = 0
        
        # Initialize parameters
        self.configs = []
        self.config_switch = []
        self.arrival_queue = []
        self.departure_queue = []
        self.arrivals_served = []
        self.departures_served = []
        self.imposed_delays = []
        self.arr_delayed = []
        self.flight_plan = []
        self.objective = 0
        self.queue_cost = 0
        self.switch_cost = 0
        self.ground_cost = 0

        # Compute return reference
        self.last_return = self.compute_return_reference(evaluate)
        
        # Generate initial state
        self._generate_initial_state(mode, evaluate)
        
        while self.time_step != self.flight_time_step:
            
            # Extract current state variables
            RC_i = int(self.obs[0])  # Previous runway configuration
            a_i = int(self.obs[1])   # Current arrival queue
            d_i = int(self.obs[2])   # Current departure queue
        
            # Getting actions from heuristic tactical policy
            cm_obs = self.obs[0:5]
            cm_action = self._heuristic_tactical_action(cm_obs)
            
            # Extract action components
            RC_f = cm_action[0]         # New runway configuration
            a_served = cm_action[1]  # Number of arrivals to process
            d_served = cm_action[2]  # Number of departures to process
            
            # Calculate new queue lengths after processing
            a_f = a_i - a_served  # New arrival queue
            d_f = d_i - d_served  # New departure queue
            
            # Ensure queues don't go negative (sanity check)
            if a_f < 0 or d_f < 0:
                print('Invalid queues detected in initial state.\n')
            
            # Calculate reward
            cm_reward = self.compute_cm_reward(RC_i, RC_f, a_i, d_i, a_f, d_f, evaluate)
            
            # Storing results
            self.configs.append(RC_f)
            self.arrival_queue.append(a_f)
            self.departure_queue.append(d_f)
            self.arrivals_served.append(a_served)
            self.departures_served.append(d_served)
            self.objective += cm_reward
            
            # Advance time step
            self.time_step += 1
            
            # Generate next state
            self._generate_next_state(RC_f, a_f, d_f, mode, evaluate)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Ensure observation is float32 and doesn't contain NaN/inf
        observation = np.asarray(observation, dtype=np.float32)
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"Warning: NaN or inf detected in reset observation: {observation}")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _generate_next_state(self, RC_f, a_f, d_f, mode='train', evaluate=False):
        """
        Generate the next state given the current action results.
        
        Parameters:
        -----------
        RC_f : int
            Runway configuration after action
        a_f : int
            Arrival queue after action
        d_f : int
            Departure queue after action
        training : bool
            Whether the environment is in training mode
            
        Returns:
        --------
        numpy.ndarray
            Next state
        """
        
        # Demand calculation
        f_3 = self.day_data.loc[self.time_step, 'scheduled_arrivals_3']
        dep_dem = self.day_data.loc[self.time_step, 'scheduled_departures']
        
        # Calculate new queues
        RC_i = RC_f
        a_i = a_f + f_3
        d_i = d_f + dep_dem
        wc = self.day_data.loc[self.time_step, 'weather_condition']
        ws = self.day_data.loc[self.time_step, 'wind_state']
        
        # Calculate future values safely to avoid NaN from empty slices
        future_time_slice = self.day_data.loc[(self.time_step+1):(len(self.day_data)-1), 'scheduled_arrivals_3']
        future_arr = np.sum(future_time_slice) if len(future_time_slice) > 0 else 0.0
        
        future_time_slice = self.day_data.loc[(self.time_step+1):(len(self.day_data)-1), 'scheduled_departures']
        future_dep = np.sum(future_time_slice) if len(future_time_slice) > 0 else 0.0
        
        future_time_slice = self.day_data.loc[(self.time_step+1):(len(self.day_data)-1), 'weather_condition']
        future_wc = np.mean(future_time_slice) if len(future_time_slice) > 0 else float(wc)
        
        future_time_slice = self.day_data.loc[(self.time_step+1):(len(self.day_data)-1), 'wind_state']
        future_ws = np.mean(future_time_slice) if len(future_time_slice) > 0 else float(ws)
        
        # Initial observation - ensure float32 dtype
        self.obs = np.array([
            RC_i,    # Runway configuration (0 = None, 1 = 10R_10L, 2 = 28L_28R)
            a_i,  # Arrival queue (integer)
            d_i,  # Departure queue (integer)
            wc,   # Weather condition (0 = VMC, 1 = IMC, 2 = LIMC)
            ws#,    # Wind state (0 = Both available, 1 = 10R_10L available, 2 = 28L_28R available, 3 = Both unavailable)
            # future_arr, # Future arrival demand
            # future_dep, # Future departure demand
            # future_wc,  # Future weather condition 
            # future_ws,   # Future wind state
            # self.time_step # Current time step
        ], dtype=np.float32)
        self.arr_delayed.append(0)
        
    def _run_tactical_model(self, evaluate):
        
        # Extract current state variables
        RC_i = int(self.obs[0])  # Previous runway configuration
        a_i = int(self.obs[1])   # Current arrival queue
        d_i = int(self.obs[2])   # Current departure queue
        
        # Getting actions from heuristic tactical policy
        cm_obs = self.obs[0:5]
        cm_action = self._heuristic_tactical_action(cm_obs)
        
        # Extract action components
        RC_f = cm_action[0]         # New runway configuration
        a_served = cm_action[1]  # Number of arrivals to process
        d_served = cm_action[2]  # Number of departures to process
        
        # Calculate new queue lengths after processing
        a_f = a_i - a_served  # New arrival queue
        d_f = d_i - d_served  # New departure queue
        
        # Ensure queues don't go negative (sanity check)
        if a_f < 0 or d_f < 0:
            print('Invalid queues detected during step.\n')
        
        # Calculate reward
        cm_reward = self.compute_cm_reward(RC_i, RC_f, a_i, d_i, a_f, d_f, evaluate)
        
        # Storing results
        self.configs.append(RC_f)
        self.arrival_queue.append(a_f)
        self.departure_queue.append(d_f)
        self.arrivals_served.append(a_served)
        self.departures_served.append(d_served)
        self.objective += cm_reward
        
        return RC_f, a_f, d_f
    
    def step(self, action_index, mode='train', evaluate=False):
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        action_index : int
            Index of the action to take
        training : bool, optional
            Whether the environment is in training mode
            
        Returns:
        --------
        tuple
            Observation, reward, terminated, truncated, info
        """        
        # # Ensure action_index is properly handled - convert various array types to int
        # if isinstance(action_index, np.ndarray):
        #     # Use .item() for both 0-d and 1-d arrays
        #     action_index = int(action_index.item()) if action_index.size > 0 else 0
        # else:
        #     action_index = int(action_index)
            
        # Get selected action from action index
        action = self.possible_actions[action_index]
        g_f = action[0]
        self.imposed_delays.append(g_f)
        current_time_step = self.day_arrival_data.loc[self.flight, 'day_slot']
        if g_f != 0:
            self.day_data.loc[current_time_step, 'scheduled_arrivals_3'] -= 1
            self.day_arrival_data.loc[self.flight, 'day_slot'] += g_f
            new_time_step = self.day_arrival_data.loc[self.flight, 'day_slot']
            self.day_data.loc[new_time_step, 'scheduled_arrivals_3'] += 1
            self.day_arrival_data.loc[self.flight, 'scheduled_departure_time'] += g_f * datetime.timedelta(minutes=15)
            self.day_arrival_data.loc[self.flight, 'scheduled_arrival_time'] += g_f * datetime.timedelta(minutes=15)
            self.day_arrival_data.loc[self.flight, 'scheduled_arrival_3_time'] += g_f * datetime.timedelta(minutes=15)
            self.day_arrival_data.loc[self.flight, 'slot_time'] += g_f * datetime.timedelta(minutes=15)
        
            # Update number of arrivals delayed for the current time step
            self.arr_delayed[-1] += 1
            self.obs[1] -= 1
            #self.obs[5] += 1
            
        # Compute reward
        reward = self.compute_reward(g_f, evaluate)
        self.objective += reward
        
        # Advance flight
        self.flight += 1
        
        # Check if episode has terminated
        terminated = self.flight == len(self.day_arrival_data)
        
        if not terminated:
            
            # Next flight slot
            self.flight_time_step = self.day_arrival_data.loc[self.flight, 'day_slot']
            next_time_step = self.flight_time_step
            
            if next_time_step != current_time_step:

                # Run tactical model
                RC_f, a_f, d_f = self._run_tactical_model(evaluate)
            
                # Advance time step
                self.time_step += 1
            
                # Generate next state
                self._generate_next_state(RC_f, a_f, d_f, mode, evaluate)
                
                # While next state is not at the same time step
                while self.time_step != self.flight_time_step:
                    
                    # Run tactical model
                    RC_f, a_f, d_f = self._run_tactical_model(evaluate)
                    
                    # Advance time step
                    self.time_step += 1
                    
                    # Generate next state
                    self._generate_next_state(RC_f, a_f, d_f, mode, evaluate)
                       
        else:
            
            while self.time_step != len(self.day_data) - 1:
                
                # Run tactical model
                RC_f, a_f, d_f = self._run_tactical_model(evaluate)
                
                # Advance a time step
                self.time_step += 1
                
                # Generate next state
                self._generate_next_state(RC_f, a_f, d_f, mode, evaluate)

            # Run tactical model
            RC_f, a_f, d_f = self._run_tactical_model(evaluate)
            
            # Compile solution
            solution = {
                'objective': self.objective,
                'queue_cost': self.queue_cost,
                'switch_cost': self.switch_cost,
                'ground_cost': self.ground_cost,
                'configs': self.configs,
                'arrival_queue': self.arrival_queue,
                'departure_queue': self.departure_queue,
                'arrivals_served': self.arrivals_served,
                'departures_served': self.departures_served,
                'imposed_delays': self.imposed_delays,
                'arr_delayed': self.arr_delayed,
                'flight_plan': self.day_arrival_data
            }
            self.results[self.day] = solution
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Ensure observation is float32 and doesn't contain NaN/inf
        observation = np.asarray(observation, dtype=np.float32)
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"Warning: NaN or inf detected in step observation: {observation}")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def set_of_actions(self) -> np.ndarray:
        """
        Get the set of valid actions for the current state.
        
        Returns:
            Array of valid actions
        """
            
        # Return valid actions using precomputed boolean mask
        return self.possible_actions[self.valid_bool[self.time_step]]
    
    def action_masks(self) -> np.ndarray:
        """
        Get the action mask for the current state.
        
        Returns:
            Boolean array indicating which actions are valid
        """
        
        # Return boolean mask for valid actions
        return self.valid_bool[self.time_step]
    
    def compute_heuristic_baseline(self) -> float:
        """
        Compute the baseline reward using the heuristic tactical action
        applied to the original flight schedule (no strategic delays).
        
        This baseline represents the performance of the heuristic policy
        without any strategic demand management interventions (action 0).
        The heuristic is applied at each time step based on current state
        and capacity constraints.
        
        Returns:
            float: Total baseline reward for the day
        """
        obs, _ = self.reset(mode='test', evaluate=True)
        baseline_reward = 0.0
        terminated = False
        
        while not terminated:
            # Apply heuristic tactical action with no strategic delay (g_f = 0, action 0)
            obs, reward, terminated, truncated, info = self.step(0, mode='test', evaluate=True)
            baseline_reward += reward
        
        return baseline_reward
    
    def compare_to_baseline(self, day: int) -> Dict[str, Any]:
        """
        Compare optimization results to heuristic baseline for a specific day.
        
        The baseline is computed by applying the heuristic tactical action
        without any strategic delays (action 0) throughout the day.
        
        Args:
            day: Day index to compare
            
        Returns:
            Dictionary with comparison metrics including objectives and improvement
        """
        if day not in self.results:
            raise ValueError(f"Day {day} has not been optimized yet")
        
        # Get optimization results
        opt_results = self.results[day]
        
        # Compute baseline reward using heuristic policy with no strategic delays
        baseline_reward = self.compute_heuristic_baseline()
        
        # Calculate improvement (lower rewards/costs are better)
        improvement = (baseline_reward - opt_results['objective']) / abs(baseline_reward) * 100 if baseline_reward != 0 else 0
        
        # Return comparison metrics
        comparison = {
            'day': day,
            'opt_objective': opt_results['objective'],
            'base_objective': baseline_reward,
            'improvement': improvement,
            'opt_configs': opt_results['configs'],
            'opt_arr_delayed': opt_results['arr_delayed'],
            'opt_arr_queue': opt_results['arrival_queue'],
            'opt_dep_queue': opt_results['departure_queue'],
            'arrivals_served': opt_results['arrivals_served'],
            'departures_served': opt_results['departures_served'],
            'config_switches': opt_results['config_switch'] if 'config_switch' in opt_results else []
        }
        
        return comparison
    
    def visualize_results(self, day: int, figsize=(15, 12), folder=""):
        """
        Visualize optimization results for a specific day.
        
        Args:
            day: Day index to visualize
            figsize: Figure size for the plot
        """
        if day not in self.results:
            raise ValueError(f"Day {day} has not been optimized yet")
        
        # Get results
        comparison = self.compare_to_baseline(day)
        results = self.results[day]
        
        # Create figure
        fig, axs = plt.subplots(5, 1, figsize=figsize)
        
        # Time steps
        time_steps = range(len(comparison['opt_configs']))
        n_time_steps = len(time_steps)
        
        # Plot runway configurations
        axs[0].plot(time_steps, comparison['opt_configs'], 'b-', label='SDCB')
        axs[0].plot(time_steps, comparison['base_configs'], 'r--', label='CM')
        axs[0].set_ylabel('Runway Config')
        axs[0].set_title(f'Day {day} - Comparison (Improvement: {comparison["improvement"]:.2f}%)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot separations
        axs[1].plot(time_steps, comparison['opt_arr_delayed'], 'b-', label='SDCB')
        axs[1].plot(time_steps, comparison['base_arr_delayed'], 'r--', label='CM')
        axs[1].set_ylabel('Separation')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot arrival queues - ensure we only plot the same number of points
        axs[2].plot(time_steps, comparison['opt_arr_queue'][:n_time_steps], 'b-', label='SDCB')
        axs[2].plot(time_steps, comparison['base_arr_queue'][:n_time_steps], 'r--', label='CM')
        axs[2].set_ylabel('Arrival Queue')
        axs[2].legend()
        axs[2].grid(True)
        
        # Plot departure queues - ensure we only plot the same number of points
        axs[3].plot(time_steps, comparison['opt_dep_queue'][:n_time_steps], 'b-', label='SDCB')
        axs[3].plot(time_steps, comparison['base_dep_queue'][:n_time_steps], 'r--', label='CM')
        axs[3].set_ylabel('Departure Queue')
        axs[3].legend()
        axs[3].grid(True)
        
        # Plot arrivals and departures served
        axs[4].plot(time_steps, results['arrivals_served'], 'b-', label='Arrivals Served')
        axs[4].plot(time_steps, results['departures_served'], 'g-', label='Departures Served')
        axs[4].set_xlabel('Time Step')
        axs[4].set_ylabel('Aircraft Served')
        axs[4].legend()
        axs[4].grid(True)
        
        plt.tight_layout()
        plt.savefig(folder + "comparison_results.pdf", format="pdf")
        plt.show()
        
        # Print summary
        print(f"Day {day} Summary:")
        print(f"  Objective: DCB = {comparison['opt_objective']:.2f}, CM = {comparison['base_objective']:.2f}")
        print(f"  Improvement: {comparison['improvement']:.2f}%")
        print("  Cost Breakdown:")
        print(f"    Queue Cost: {results['queue_cost']:.2f}")
        print(f"    Switch Cost: {results['switch_cost']:.2f}")
        print(f"    Ground Cost: {results['ground_cost']:.2f}")
    
    def _init_rendering(self, render_mode):
        """Initialize rendering-related variables."""
        # Validate render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
        # Rendering attributes
        self.window = None
        self.clock = None
        self.screen_shot_count = 0
        self.screen_shot_button = pygame.Rect(10, 10, 120, 30)  # (x, y, width, height)
        
        # Colors for rendering
        self.colors = {
            "background": (240, 240, 240),
            "text": (20, 20, 20),
            "arrival_queue": (70, 130, 180),   # Steel blue
            "departure_queue": (220, 120, 50), # Orange
            "side_A": (50, 200, 50),   # Green
            "side_B": (200, 50, 50),   # Red
            "unavailable": (120, 120, 120),    # Gray
            "button": (100, 100, 240),         # Blue
            "button_hover": (130, 130, 255),   # Light blue
            "button_text": (255, 255, 255)     # White
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        return self._render_frame()
    
    def _render_frame(self):
        """
        Render a single frame of the environment.
        
        Returns:
        --------
        numpy.ndarray or None
            RGB array if render_mode is "rgb_array", otherwise None
        """
        import pygame
        
        # Initialize Pygame if not already done
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Runway Configuration Optimization")
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Create the surface to draw on
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.colors["background"])
        
        # Extract current state
        rc = int(self.obs[0])
        arrivals = int(self.obs[1])
        departures = int(self.obs[2])
        weather = int(self.obs[3])
        wind = int(self.obs[4])
        
        # Render time step and day information
        self._render_text(canvas, f"Day: {self.day}", 20, 50)
        self._render_text(canvas, f"Time Step: {self.time_step}", 20, 80)
        
        # Render weather and wind information
        weather_labels = ["VMC", "IMC", "LIMC"]
        wind_labels = ["All Available", "Side A Available", "Side B Available", "None Available"]
        self._render_text(canvas, f"Weather: {weather_labels[weather]}", 20, 110)
        self._render_text(canvas, f"Wind: {wind_labels[wind]}", 20, 140)
        
        # Render queue information
        self._render_text(canvas, f"Arrival Queue: {arrivals}", 20, 180, self.colors["arrival_queue"])
        self._render_text(canvas, f"Departure Queue: {departures}", 20, 210, self.colors["departure_queue"])
        
        # Render runway configuration
        rc_labels = ["None", "Side A", "Side B"]
        self._render_text(canvas, f"Runway Configuration: {rc_labels[rc]}", 20, 250)
        
        # Draw queue bars
        self._draw_queue_bars(canvas, arrivals, departures)
        
        # Draw runway diagram
        self._draw_runway_diagram(canvas, rc, wind)
        
        # Draw screenshot button
        self._draw_button(canvas)
        
        # Copy canvas to window and handle events if render_mode is "human"
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.screen_shot_button.collidepoint(mouse_pos):
                        self._take_screenshot()
            
            # Maintain frame rate
            self.clock.tick(self.metadata["render_fps"])
        
        # Return canvas as RGB array if render_mode is "rgb_array"
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(canvas).transpose((1, 0, 2))
    
    def _render_text(self, canvas, text, x, y, color=None):
        """
        Render text on the canvas.
        
        Parameters:
        -----------
        canvas : pygame.Surface
            Surface to draw on
        text : str
            Text to render
        x, y : int
            Position to render text
        color : tuple, optional
            RGB color tuple, defaults to text color
        """
        import pygame
        
        if not hasattr(self, 'font'):
            self.font = pygame.font.SysFont(None, 24)
        
        if color is None:
            color = self.colors["text"]
        
        text_surface = self.font.render(text, True, color)
        canvas.blit(text_surface, (x, y))
    
    def _draw_queue_bars(self, canvas, arrivals, departures):
        """
        Draw bars representing queue sizes.
        
        Parameters:
        -----------
        canvas : pygame.Surface
            Surface to draw on
        arrivals : int
            Size of arrival queue
        departures : int
            Size of departure queue
        """
        import pygame
        
        # Constants for bar graph
        bar_width = 40
        max_height = 150
        max_queue = 50  # Scale bar height
        
        # Calculate heights (with max cap)
        arr_height = min(arrivals / max_queue * max_height, max_height)
        dep_height = min(departures / max_queue * max_height, max_height)
        
        # Draw arrival queue bar
        arr_bar = pygame.Rect(
            300, 
            320 - arr_height,
            bar_width, 
            arr_height
        )
        pygame.draw.rect(canvas, self.colors["arrival_queue"], arr_bar)
        pygame.draw.rect(canvas, self.colors["text"], arr_bar, 2)  # outline
        
        # Draw departure queue bar
        dep_bar = pygame.Rect(
            360, 
            320 - dep_height,
            bar_width, 
            dep_height
        )
        pygame.draw.rect(canvas, self.colors["departure_queue"], dep_bar)
        pygame.draw.rect(canvas, self.colors["text"], dep_bar, 2)  # outline
        
        # Draw labels
        self._render_text(canvas, "Arrivals", 290, 330)
        self._render_text(canvas, "Departures", 340, 330)
        
        # Draw horizontal line
        pygame.draw.line(canvas, self.colors["text"], (280, 320), (420, 320), 2)
    
    def _draw_runway_diagram(self, canvas, rc, wind):
        """
        Draw runway diagram showing current configuration.
        
        Parameters:
        -----------
        canvas : pygame.Surface
            Surface to draw on
        rc : int
            Runway configuration
        wind : int
            Wind state
        """
        import pygame
        
        # Constants for runway diagram
        center_x = self.window_size // 2
        center_y = 420
        runway_length = 200
        runway_width = 20
        
        # Draw N/S/E/W indicators
        self._render_text(canvas, "N", center_x - 5, center_y - 110)
        self._render_text(canvas, "S", center_x - 5, center_y + 100)
        self._render_text(canvas, "E", center_x + 100, center_y - 5)
        self._render_text(canvas, "W", center_x - 110, center_y - 5)
        
        # Function to draw a runway with given angle
        def draw_runway(angle, label, color):
            # Calculate endpoints based on angle
            rad_angle = math.radians(angle)
            x1 = center_x - (runway_length / 2) * math.cos(rad_angle)
            y1 = center_y - (runway_length / 2) * math.sin(rad_angle)
            x2 = center_x + (runway_length / 2) * math.cos(rad_angle)
            y2 = center_y + (runway_length / 2) * math.sin(rad_angle)
            
            # Draw runway line
            pygame.draw.line(canvas, color, (x1, y1), (x2, y2), runway_width)
            
            # Draw runway label
            label_x = center_x + (runway_length / 2 + 15) * math.cos(rad_angle)
            label_y = center_y + (runway_length / 2 + 15) * math.sin(rad_angle)
            self._render_text(canvas, label, label_x - 20, label_y - 10, color)
        
        # Draw runways based on configuration and wind state
        import math
        
        # Determine colors based on wind state and configuration
        color_10R_10L = self.colors["unavailable"]
        color_28L_28R = self.colors["unavailable"]
        
        if wind == 0:  # Both available
            color_10R_10L = self.colors["runway_10R_10L"]
            color_28L_28R = self.colors["runway_28L_28R"]
        elif wind == 1:  # Only 10R_10L available
            color_10R_10L = self.colors["runway_10R_10L"]
        elif wind == 2:  # Only 28L_28R available
            color_28L_28R = self.colors["runway_28L_28R"]
            
        # Highlight the active configuration
        if rc == 1:  # 10R_10L active
            color_10R_10L = self.colors["runway_10R_10L"]
            # Draw a highlight around the active runway
            pygame.draw.circle(canvas, color_10R_10L, (center_x, center_y), runway_length // 2 + 10, 3)
        elif rc == 2:  # 28L_28R active
            color_28L_28R = self.colors["runway_28L_28R"]
            # Draw a highlight around the active runway
            pygame.draw.circle(canvas, color_28L_28R, (center_x, center_y), runway_length // 2 + 10, 3)
            
        # Draw the runways
        draw_runway(100, "10R/10L", color_10R_10L)  # ~100 degrees for 10R/10L
        draw_runway(280, "28L/28R", color_28L_28R)  # ~280 degrees for 28L/28R
    
    def _draw_button(self, canvas):
        """
        Draw screenshot button on the canvas.
        
        Parameters:
        -----------
        canvas : pygame.Surface
            Surface to draw on
        """
        import pygame
        
        # Check if mouse is over button
        mouse_pos = pygame.mouse.get_pos() if self.render_mode == "human" else (0, 0)
        button_color = self.colors["button_hover"] if self.screen_shot_button.collidepoint(mouse_pos) else self.colors["button"]
        
        # Draw button
        pygame.draw.rect(canvas, button_color, self.screen_shot_button)
        pygame.draw.rect(canvas, self.colors["text"], self.screen_shot_button, 2)  # Border
        
        # Draw button text
        if not hasattr(self, 'font'):
            self.font = pygame.font.SysFont(None, 24)
        
        text_surface = self.font.render("Screenshot", True, self.colors["button_text"])
        text_rect = text_surface.get_rect(center=self.screen_shot_button.center)
        canvas.blit(text_surface, text_rect)
    
    def _take_screenshot(self):
        """Save current frame as a screenshot."""
        import pygame
        import os
        
        # Create screenshots directory if it doesn't exist
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        
        # Generate filename
        filename = f"screenshots/runway_env_{self.day}_step_{self.time_step}_{self.screen_shot_count}.png"
        self.screen_shot_count += 1
        
        # Save screenshot
        pygame.image.save(self.window, filename)
        print(f"Screenshot saved as {filename}")
