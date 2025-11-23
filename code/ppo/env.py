import torch
import numpy as np

class IntersectionEnv:
    def __init__(self, target_pos, agent_data_list, dt=0.1):
        """
        Custom Environment for Collision Avoidance.
        
        Args:
            target_pos (list or tensor): The target state for the ego vehicle (e.g., [target_y, target_velocity]).
            agent_data_list (list): A list of expert trajectories. Each item is a tensor/array of shape (T, 6).
                                    Format: [x, y, vx, vy, sx, sy]
            dt (float): Time step interval.
        """
        self.dt = dt
        self.target_pos = torch.tensor(target_pos, dtype=torch.float32)
        
        # 1. Dynamics Matrices setup based on proposal equation: s_{t+1} = Fs_t + Ga_t
        # State s = [y, v] (position, velocity)
        # F matrix: [[1, dt], [0, 1]]
        self.F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float32)
        
        # G matrix: [[0.5*dt^2], [dt]]
        self.G = torch.tensor([0.5 * dt**2, dt], dtype=torch.float32)

        self.max_accel = 5.0  # Physical acceleration limit
        
        # Define a hard limit to prevent episodes from running too long if data is outliers
        self.episode_len_cap = 500 
        self.max_steps = self.episode_len_cap # Will be updated dynamically in reset()

        # Store all available agent recordings (expert data)
        self.agent_data_list = agent_data_list 
        self.current_agent_traj = None # The specific trajectory used for the current episode

        self.state = None # Ego state: [y, v]
        self.steps = 0

    def reset(self):
        """
        Resets the environment for a new episode.
        1. Resets ego state.
        2. Randomly selects a new agent trajectory from the dataset.
        3. Sets the max_steps based on the length of the selected trajectory.
        """
        # Initialize ego state [y, v] to zero (or a specific start position)
        self.state = torch.tensor([-1.25, 0.0], dtype=torch.float32) 
        self.steps = 0

        # Randomly sample an agent trajectory (the "recording")
        idx = np.random.randint(len(self.agent_data_list))
        self.current_agent_traj = self.agent_data_list[idx] 
        
        # --- Dynamic Max Steps Calculation ---
        # Set max steps to the length of the recording, but cap it at 500 (50 seconds)
        # to ensure efficient training even with outlier data.
        traj_len = len(self.current_agent_traj)
        self.max_steps = min(traj_len, self.episode_len_cap)

        # Retrieve the first frame of the agent data (t=0)
        # agent_info format: [x, y, vx, vy, sx, sy]
        agent_info = self.current_agent_traj[0] 
        
        return self._get_obs(agent_info)

    def step(self, action):
        """
        Executes one time step.
        
        Args:
            action (tensor or float): The acceleration output from the policy.
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        # --- 1. Ego Dynamics Update (Online Physics) ---
        # Clamp action to physical limits
        accel = torch.clamp(action, -self.max_accel, self.max_accel)
        
        # Apply linear dynamics: s_{t+1} = F * s_t + G * a_t
        self.state = torch.matmul(self.F, self.state) + self.G * accel
        
        # --- 2. Agent Update (Replay from Data) ---
        self.steps += 1
        
        done = False
        
        # Check if we have reached the dynamic max_steps for this specific episode
        if self.steps >= self.max_steps:
            done = True
            # Use the last frame to prevent index out of bounds if we hit the limit
            # (Note: self.max_steps is guaranteed to be <= len(traj))
            # We subtract 1 because indices are 0-based
            safe_idx = min(self.steps, len(self.current_agent_traj) - 1)
            agent_info = self.current_agent_traj[safe_idx]
        else:
            agent_info = self.current_agent_traj[self.steps]
        

        if not isinstance(agent_info, torch.Tensor):
            agent_info = torch.tensor(agent_info, dtype=torch.float32)
        # Parse agent data for reward calculation
        # agent_info: [x, y, vx, vy, sx, sy]
        agent_mu = agent_info[:2]      # Agent position (mean): x, y
        
        # Construct Covariance Matrix. Assuming input sx, sy are standard deviations.
        agent_sigma = torch.diag(agent_info[4:6]**2)    # TODO: Verify if sx, sy are stddev or variance

        # --- 3. Compute Reward ---
        reward = self._compute_reward(self.state, accel, agent_mu, agent_sigma)
        
        return self._get_obs(agent_info), reward, done, {}

    def _get_obs(self, agent_info):
        """
        Constructs the observation vector.
        Combines Ego state (dynamic) and Agent state (from data).
        """
        # Ego info: [y, v]
        ego_obs = self.state 
        
        # Agent info: [x, y, vx, vy, sx, sy]
        if not isinstance(agent_info, torch.Tensor):
            agent_info = torch.tensor(agent_info, dtype=torch.float32)
            
        # Concatenate to form the full state vector
        # Total dims: 2 (ego) + 6 (agent) = 8 dimensions
        obs = torch.cat([ego_obs, agent_info]) 
        
        return obs
        
    def _compute_reward(self, ego_state, action, agent_mu, agent_sigma):
        """
        Calculates the reward based on the proposal.
        Reward = - (Performance Cost + Safety Penalty)
        """
        # --- Part A: Performance/Tracking Cost ---
        # [cite_start]c(s, a) = (s - s*)^T Q (s - s*) + r * a^2
        Q = torch.eye(2) 
        r = 0.1 # Weight for control effort
        
        diff = ego_state - self.target_pos
        # Quadratic cost for state deviation
        state_cost = torch.matmul(diff, torch.matmul(Q, diff))  # TODO: Verify Q structure
        # Quadratic cost for acceleration (comfort/efficiency)
        action_cost = r * (action ** 2)
        
        track_cost = state_cost + action_cost

        # --- Part B: Safety Penalty (Mahalanobis Distance) ---
        # [cite_start]Ego position in 2D plane: u_t = (0.5, y) 
        # Note: Ego x is fixed at 0.5 based on proposal
        u_t = torch.tensor([0.5, ego_state[0]]) 
        
        # Calculate vector difference between Ego and Agent
        delta = u_t - agent_mu
        
        # Calculate inverse covariance
        try:
            inv_sigma = torch.inverse(agent_sigma)
        except RuntimeError:
            # Fallback for numerical stability if sigma is singular
            inv_sigma = torch.eye(2) * 100 

        # [cite_start]Mahalanobis distance squared: delta^T * Sigma^-1 * delta 
        mahalanobis_sq = torch.matmul(delta, torch.matmul(inv_sigma, delta))
        
        # Penalty formulation: Inverse of distance (High penalty when close)
        epsilon = 0.01
        # Using inverse squared distance logic as per typical potential field methods
        safety_penalty = 1.0 / (mahalanobis_sq + epsilon)

        # --- Total Reward ---
        # Weight for safety vs performance
        lambda_param = 500.0 
        
        # [cite_start]We maximize Reward, so we take the negative of the Costs 
        total_reward = - (track_cost + lambda_param * safety_penalty)
        
        return total_reward