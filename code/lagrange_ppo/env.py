import torch
import numpy as np

class IntersectionEnv:
    def __init__(self, target_pos, agent_data_list, dt=0.1):
        """
        Custom Environment for Lagrangian PPO.
        Separates Performance Reward and Safety Cost.
        
        Args:
            target_pos (list): Target state [y, v].
            agent_data_list (list): Expert trajectories.
            dt (float): Time step.
        """
        self.dt = dt
        self.target_pos = torch.tensor(target_pos, dtype=torch.float32)
        
        # Dynamics: s_{t+1} = Fs_t + Ga_t
        self.F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float32)
        self.G = torch.tensor([0.5 * dt**2, dt], dtype=torch.float32)

        # Increased acceleration limit to cover data outliers
        self.max_accel = 5.0  
        
        # Episode settings
        self.episode_len_cap = 500 
        self.max_steps = self.episode_len_cap 

        # Data
        self.agent_data_list = agent_data_list 
        self.current_agent_traj = None 
        self.state = None 
        self.steps = 0

    def reset(self):
        """Resets ego state and selects a new agent trajectory."""
        # 1. Initialize Ego (Fixed start at x=0.5, y=-1.25)
        self.state = torch.tensor([-1.25, 0.0], dtype=torch.float32) 
        self.steps = 0

        # 2. Randomly select an Agent recording
        idx = np.random.randint(len(self.agent_data_list))
        self.current_agent_traj = self.agent_data_list[idx]
        
        # ---------------------------------------------------------
        # [Updated] Collision Synchronization
        # Target: Sync Ego and Agent to arrive at (0.5, 0.0) simultaneously.
        # ---------------------------------------------------------
        
        # Step A: Extract Agent positions (x, y are the first two columns)
        agent_pos = self.current_agent_traj[:, :2] 
        
        # Step B: Calculate distance to the Ego's path center (x=0.5, y=0.0)
        # We look for the moment the Agent crosses or gets closest to Ego's lane.
        # Distance squared = (agent_x - 0.5)^2 + (agent_y - 0.0)^2
        dist_sq_to_conflict = (agent_pos[:, 0] - 0.5)**2 + (agent_pos[:, 1])**2
        
        # Find the time index where Agent is closest to the conflict point
        conflict_idx = np.argmin(dist_sq_to_conflict)
        
        # Step C: Rewind time
        # Ego takes approx 1.6s (16 steps) to accelerate from -1.25 to 0.0.
        # We want to start the episode 1.5s ~ 2.5s BEFORE the conflict happens.
        # This forces the Ego to make a decision (brake or rush) immediately.
        steps_to_rewind = np.random.randint(15, 25) 
        
        start_step = max(0, conflict_idx - steps_to_rewind)
        
        # Set current step
        self.steps = start_step
        
        # ---------------------------------------------------------
        
        # 3. Set max steps
        traj_len = len(self.current_agent_traj)
        frames_left = traj_len - self.steps
        self.max_steps = self.steps + min(frames_left, self.episode_len_cap)

        agent_info = self.current_agent_traj[self.steps] 
        return self._get_obs(agent_info)

    def step(self, action):
        """
        Executes one time step.
        Returns: obs, reward (task), done, info (contains 'cost')
        """
        # --- 1. Data Formatting ---
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        # --- 2. Ego Dynamics Update ---
        # Clamp action to physical limits (max_accel = 1.0 based on data)
        accel = torch.clamp(action, -self.max_accel, self.max_accel)
        
        # [Apply dynamics: s_next = F*s + G*a 
        self.state = torch.matmul(self.F, self.state) + self.G * accel
        self.steps += 1
        
        # --- 3. Termination Logic (The "Success" Check) ---
        done = False
        is_success = False
        
        # Target y position (1.25)
        target_y = self.target_pos[0].item() 
        
        # Condition A: Reached Target (Success!)
        if self.state[0] >= target_y:
            done = True
            is_success = True
            
            # Handle data index safely
            safe_idx = min(self.steps, len(self.current_agent_traj) - 1)
            agent_info = self.current_agent_traj[safe_idx]

        # Condition B: Out of Bounds (Fail)
        # If car goes backward too much or flies away
        elif self.state[0] < -2 or self.state[0] > 2:
            done = True
            # Use last frame data
            safe_idx = min(self.steps, len(self.current_agent_traj) - 1)
            agent_info = self.current_agent_traj[safe_idx]

        # Condition C: Timeout (Fail/Truncated)
        elif self.steps >= self.max_steps:
            done = True
            safe_idx = min(self.steps, len(self.current_agent_traj) - 1)
            agent_info = self.current_agent_traj[safe_idx]
            
        # Normal Step
        else:
            agent_info = self.current_agent_traj[self.steps]
        
        # --- 4. Compute Signals ---
        # Ensure agent_info is tensor
        if not isinstance(agent_info, torch.Tensor):
            agent_info = torch.tensor(agent_info, dtype=torch.float32)

        agent_mu = agent_info[:2]      
        # agent_sigma = torch.diag(agent_info[4:6]**2) 
        raw_std_x = agent_info[4]
        raw_std_y = agent_info[5]
        
        # Enforce a minimum standard deviation of 0.8m (approx half car width + margin)
        safe_std_x = torch.clamp(raw_std_x, min=0.3)
        safe_std_y = torch.clamp(raw_std_y, min=0.3)

        agent_sigma = torch.diag(torch.stack([safe_std_x**2, safe_std_y**2]))
        # Calculate Step Reward and Safety Cost
        reward, cost = self._compute_reward_and_cost(self.state, accel, agent_mu, agent_sigma)
        
        # --- 5. Apply Bonuses and Penalties ---
        
        # BONUS: Large reward for reaching the goal (Success)
        if is_success:
            reward += 50.0 
            
        # PENALTY: Large penalty for going out of bounds
        if self.state[0] < -2.0 or self.state[0] > 2.0:
            reward -= 50.0

        # Pack cost into info for Lag-PPO
        info = {"cost": cost}
        
        return self._get_obs(agent_info), reward, done, info

    def _compute_reward_and_cost(self, ego_state, action, agent_mu, agent_sigma):
        """
        Redesigned Reward Function:
        1. Task Reward: 
           - Progress Reward: Positive reward for moving forward (v > 0).
           - Speed Limit: Penalty for exceeding realistic limits.
           - Comfort: Penalty for high acceleration.
           
        2. Safety Cost: 
           - [cite_start]Bounded Mahalanobis Distance cost[cite: 56].
        """
        ego_y = ego_state[0]
        ego_v = ego_state[1]
        
        # ==========================================
        # Part 1: Performance Reward (Task)
        # ==========================================
        
        # A. Progress Reward (Encourage moving towards goal)
        # +0.2 per step if moving at 1 m/s (dt=0.1)
        r_progress = 2.0 * ego_v * self.dt
        
        # B. Speed Limit Penalty (Simulate realistic constraints)
        # Soft limit at 1.5 m/s. Penalize quadratic error if exceeded.
        v_limit = 0.6
        r_overspeed = 0.0
        if ego_v > v_limit:
            r_overspeed = -10.0 * ((ego_v - v_limit) ** 2)
            
        # C. Comfort/Control Penalty (Reduce jerky movements)
        # Penalty proportional to squared acceleration 
        r_comfort = -0.1 * (action.item() ** 2)
        
        # Total Task Reward
        performance_reward = r_progress + r_overspeed + r_comfort

        # ==========================================
        # Part 2: Safety Cost (Constraint)
        # ==========================================
        # Based on Mahalanobis Distance from proposal
        u_t = torch.tensor([0.5, ego_y]) 
        delta = u_t - agent_mu
        
        try:
            inv_sigma = torch.inverse(agent_sigma)
        except RuntimeError:
            # Fallback for singular matrices
            inv_sigma = torch.eye(2) * 100 

        # Calculate distance squared
        mahalanobis_sq = torch.matmul(delta, torch.matmul(inv_sigma, delta))
        
        # # Cost Formulation: 1 / (dist^2 + epsilon)
        # epsilon = 0.01
        # raw_cost = 1.0 / (mahalanobis_sq + epsilon)
        
        # # Clamp cost to prevent numerical explosion during collisions
        # # Max cost = 100.0
        # safety_cost = torch.clamp(raw_cost, max=100.0).item()
        safety_cost = torch.exp(-0.5 * mahalanobis_sq).item()
        safety_cost = safety_cost * 100.0

        return performance_reward.item(), safety_cost

    def _get_obs(self, agent_info):
        ego_obs = self.state 
        if not isinstance(agent_info, torch.Tensor):
            agent_info = torch.tensor(agent_info, dtype=torch.float32)
        return torch.cat([ego_obs, agent_info]) 
        
    # def _compute_reward_and_cost(self, ego_state, action, agent_mu, agent_sigma):
    #     """
    #     Computes:
    #     1. Task Reward: r(s,a) = -TrackingCost 
    #     2. Safety Cost: c(s,a) = 1 / (MahalanobisDist + eps)
    #     """
    #     # --- 1. Task Reward (Performance) ---
    #     Q = torch.eye(2) 
    #     r = 0.1
    #     diff = ego_state - self.target_pos
        
    #     state_cost = torch.matmul(diff, torch.matmul(Q, diff))
    #     action_cost = r * (action ** 2)
        
    #     # Maximize reward = Minimize cost
    #     # We do NOT include safety terms here for Lag-PPO
    #     performance_reward = - (state_cost + action_cost)

    #     # --- 2. Safety Cost (Constraint) ---
    #     u_t = torch.tensor([0.5, ego_state[0]]) 
    #     delta = u_t - agent_mu
        
    #     try:
    #         inv_sigma = torch.inverse(agent_sigma)
    #     except RuntimeError:
    #         inv_sigma = torch.eye(2) * 100 

    #     mahalanobis_sq = torch.matmul(delta, torch.matmul(inv_sigma, delta))
        
    #     epsilon = 0.01
    #     # Inverse squared distance formulation 
    #     raw_cost = 1.0 / (mahalanobis_sq + epsilon)
        
    #     # Clamp cost for stability (max cost ~ 100)
    #     safety_cost = torch.clamp(raw_cost, max=100.0)

    #     return performance_reward.item(), safety_cost.item()
    def render(self):
        """
        Returns the current state for visualization.
        """
        # Ego info
        ego_pos = [0.5, self.state[0].item()]
        ego_v = self.state[1].item()
        
        # Agent info
        safe_idx = min(self.steps, len(self.current_agent_traj) - 1)
        agent_info = self.current_agent_traj[safe_idx]
        
        agent_pos = [agent_info[0], agent_info[1]]
        # Agent data format: [x, y, vx, vy, ...] -> vx is index 2
        agent_v = agent_info[2] 
        
        return {
            'ego_pos': ego_pos,
            'ego_v': ego_v,
            'agent_pos': agent_pos,
            'agent_v': agent_v
        }