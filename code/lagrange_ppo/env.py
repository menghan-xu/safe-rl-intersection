import torch
import numpy as np

class IntersectionEnv:
    def __init__(self, target_pos, agent_data_list, config, dt=0.1):
        """
        Initialized with a config dictionary for easy hyperparameter tuning.
        """
        self.dt = dt
        self.target_pos = torch.tensor(target_pos, dtype=torch.float32)
        
        # Config
        self.cfg = config
        
        # Dynamics: s_{t+1} = Fs_t + Ga_t
        self.F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float32)
        self.G = torch.tensor([0.5 * dt**2, dt], dtype=torch.float32)

        # Physics Limits
        self.max_accel = self.cfg['max_accel']
        
        # Collision Geometry (Radius sum)
        self.collision_dist = self.cfg['robot_radius'] * 2.0 
        
        self.episode_len_cap = 500 # Max steps per episode (50 seconds)
        
        # Data
        self.agent_data_list = agent_data_list 
        self.current_agent_traj = None 
        self.state = None 
        self.steps = 0          # Index in the expert data
        self.episode_steps = 0  # Steps taken in the current RL episode

    def reset(self):
        """Resets ego state using Collision Synchronization logic."""
        # 1. Initialize Ego with random starting position
        # Sample random position within 15cm radius circle around (0.5, -1.25)
        radius = 0.15  # 15cm in meters
        angle = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))  # Uniform sampling in circle
        
        # Apply offset to starting y-position (x is fixed at 0.5 for ego)
        y_offset = r * np.cos(angle)
        y_start = -1.25 + y_offset
        
        self.state = torch.tensor([y_start, 0.0], dtype=torch.float32)
        
        # Reset counters
        self.steps = 0 
        self.episode_steps = 0

        # 2. Select Agent Recording
        idx = np.random.randint(len(self.agent_data_list))
        self.current_agent_traj = self.agent_data_list[idx]
        
        # 3. Collision Sync
        # Find frame where Agent is closest to Ego's path (x=0.5, y=0.0)
        agent_pos = self.current_agent_traj[:, :2] 
        dist_sq_to_conflict = (agent_pos[:, 0] - 0.5)**2 + (agent_pos[:, 1])**2
        conflict_idx = np.argmin(dist_sq_to_conflict)
        
        # Rewind 15-25 steps (1.5s - 2.5s) to create interaction window
        steps_to_rewind = np.random.randint(15, 25) 
        start_step = max(0, conflict_idx - steps_to_rewind)
        
        # Set the data pointer to the calculated start
        self.steps = start_step

        # Get initial agent info
        safe_idx = min(self.steps, len(self.current_agent_traj) - 1)
        agent_info = self.current_agent_traj[safe_idx] 
        return self._get_obs(agent_info)

    def step(self, action):
        """Step function with 'Freeze Agent' logic if data ends."""
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        # 1. Physics Update
        accel = torch.clamp(action, -self.max_accel, self.max_accel)
        self.state = torch.matmul(self.F, self.state) + self.G * accel
        
        # Increment counters
        self.steps += 1        # Advances data pointer
        self.episode_steps += 1 # Advances episode timer
        
        done = False
        is_success = False
        is_collision = False
        
        target_y = self.target_pos[0].item()
        
        # 2. Get Agent Data (Freeze Logic)
        # If self.steps exceeds the recording length, we clamp the index to the last frame.
        # The agent will appear to stop moving at the end of its trajectory.
        traj_len = len(self.current_agent_traj)
        safe_idx = min(self.steps, traj_len - 1)
        agent_info = self.current_agent_traj[safe_idx]
        
        if not isinstance(agent_info, torch.Tensor):
            agent_info = torch.tensor(agent_info, dtype=torch.float32)

        # 3. Collision Check (Euclidean Distance)
        ego_pos = torch.tensor([0.5, self.state[0]])
        agent_pos = agent_info[:2]
        dist = torch.norm(ego_pos - agent_pos).item()
        
        # --- Termination Logic ---
        if dist < self.collision_dist:
            done = True
            is_collision = True
        elif self.state[0] >= target_y:
            done = True
            is_success = True
        elif self.state[0] < -6.0 or self.state[0] > 25.0: # Out of bounds
            done = True
        # Check timeout based on episode duration, NOT data length
        elif self.episode_steps >= self.episode_len_cap: 
            done = True

        # 4. Compute Reward & Soft Cost
        reward, cost, metrics = self._compute_reward_and_cost(self.state, accel, agent_info)
        
        # 5. Apply Bonuses / Penalties
        bonus = 0.0
        if is_success:
            reward += self.cfg['reward_success']
            bonus = self.cfg['reward_success']
        
        if is_collision:
            reward += self.cfg['reward_collision'] 
            cost = self.cfg['cost_crash'] 
            bonus = self.cfg['reward_collision']
            
        metrics['bonus'] = bonus
        metrics['total_reward'] = reward
        metrics['final_cost'] = cost

        info = {"cost": cost, **metrics}
        
        return self._get_obs(agent_info), reward, done, info

    def _get_obs(self, agent_info):
        ego_obs = self.state 
        if not isinstance(agent_info, torch.Tensor):
            agent_info = torch.tensor(agent_info, dtype=torch.float32)
        return torch.cat([ego_obs, agent_info]) 
        
    def _compute_reward_and_cost(self, ego_state, action, agent_info):
        """
        Computes Task Reward and Soft Safety Cost.
        """
        ego_y = ego_state[0]
        ego_v = ego_state[1]
        
        # --- Task Reward ---
        # 1. Progress
        r_progress = self.cfg['w_progress'] * ego_v * self.dt
        
        r_time = -self.cfg['w_time_penalty'] * self.dt
        # 2. Overspeed
        v_limit = self.cfg['v_limit']
        r_overspeed = 0.0
        if ego_v > v_limit:
            r_overspeed = -self.cfg['w_overspeed'] * ((ego_v - v_limit) ** 2)
            
        # 3. Comfort
        r_comfort = -self.cfg['w_comfort'] * (action.item() ** 2)
        
        performance_reward = r_progress + r_overspeed + r_comfort + r_time

        # --- Safety Cost ---
        xa, ya = agent_info[0], agent_info[1]
        s_xa, s_ya = agent_info[4], agent_info[5]
        xe, ye = 0.5, ego_y
        
        # Conservative Boundary Calculation
        sigma_combined = torch.sqrt(s_xa**2 + s_ya**2)
        safety_radius = self.collision_dist + sigma_combined
        d_conservative_sq = safety_radius**2
        
        # Actual Distance Squared
        d_actual_sq = (xe - xa)**2 + (ye - ya)**2
        
        # Soft Cost
        raw_soft_cost = d_conservative_sq - d_actual_sq
        if isinstance(raw_soft_cost, torch.Tensor):
            raw_soft_cost = raw_soft_cost.item()
            
        soft_cost = self.cfg['cost_scale_mu'] * max(0.0, raw_soft_cost)

        # Metrics for visualization
        metrics = {
            "r_prog": r_progress.item() if isinstance(r_progress, torch.Tensor) else r_progress,
            "r_speed": r_overspeed.item() if isinstance(r_overspeed, torch.Tensor) else r_overspeed,
            "r_comf": r_comfort,
            "d_act": np.sqrt(d_actual_sq.item()) if isinstance(d_actual_sq, torch.Tensor) else np.sqrt(d_actual_sq),
            "d_cons": np.sqrt(d_conservative_sq.item()) if isinstance(d_conservative_sq, torch.Tensor) else np.sqrt(d_conservative_sq)
        }

        return performance_reward.item(), soft_cost, metrics
    
    def render(self):
        """Returns current state for visualization"""
        ego_pos = [0.5, self.state[0].item()]
        ego_v = self.state[1].item()
        
        safe_idx = min(self.steps, len(self.current_agent_traj) - 1)
        agent_info = self.current_agent_traj[safe_idx]
        agent_pos = [agent_info[0], agent_info[1]]
        agent_v = agent_info[2] 
        
        return {
            'ego_pos': ego_pos, 'ego_v': ego_v,
            'agent_pos': agent_pos, 'agent_v': agent_v
        }