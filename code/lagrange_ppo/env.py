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
        
        # Dynamics: s_{t+1} = Fs_t + Ga_t [cite: 33]
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
        self.state = torch.tensor([-1.25, 0.0], dtype=torch.float32)  # TODO: check start pos
        self.steps = 0

        # Sample trajectory
        idx = np.random.randint(len(self.agent_data_list))
        self.current_agent_traj = self.agent_data_list[idx] 
        
        # Set dynamic max steps
        traj_len = len(self.current_agent_traj)
        self.max_steps = min(traj_len, self.episode_len_cap)

        agent_info = self.current_agent_traj[0] 
        return self._get_obs(agent_info)

    def step(self, action):
        """
        Returns: obs, reward (task), done, info (contains 'cost')
        """
        # Ensure action is tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        # 1. Ego Update
        accel = torch.clamp(action, -self.max_accel, self.max_accel)
        self.state = torch.matmul(self.F, self.state) + self.G * accel
        
        # 2. Agent Update
        self.steps += 1
        done = False
        
        # Termination conditions
        if self.steps >= self.max_steps:
            done = True
            safe_idx = min(self.steps, len(self.current_agent_traj) - 1)
            agent_info = self.current_agent_traj[safe_idx]
        # Early stopping if ego goes too far out of bounds (numerical stability)
        elif self.state[0] < -10.0 or self.state[0] > 30.0:
            done = True
            agent_info = self.current_agent_traj[self.steps]
        else:
            agent_info = self.current_agent_traj[self.steps]
        
        # Prepare data for calculation
        if not isinstance(agent_info, torch.Tensor):
            agent_info = torch.tensor(agent_info, dtype=torch.float32)

        agent_mu = agent_info[:2]      
        agent_sigma = torch.diag(agent_info[4:6]**2) 

        # 3. Compute Separate Signals
        reward, cost = self._compute_reward_and_cost(self.state, accel, agent_mu, agent_sigma)
        
        # Out of bounds penalty for the task reward
        if self.state[0] < -10.0 or self.state[0] > 30.0:
            reward -= 500.0

        # Pack cost into info for Lag-PPO
        info = {"cost": cost}
        
        return self._get_obs(agent_info), reward, done, info

    def _get_obs(self, agent_info):
        ego_obs = self.state 
        if not isinstance(agent_info, torch.Tensor):
            agent_info = torch.tensor(agent_info, dtype=torch.float32)
        return torch.cat([ego_obs, agent_info]) 
        
    def _compute_reward_and_cost(self, ego_state, action, agent_mu, agent_sigma):
        """
        Computes:
        1. Task Reward: r(s,a) = -TrackingCost [cite: 52]
        2. Safety Cost: c(s,a) = 1 / (MahalanobisDist + eps) [cite: 55]
        """
        # --- 1. Task Reward (Performance) ---
        Q = torch.eye(2) 
        r = 0.1
        diff = ego_state - self.target_pos
        
        state_cost = torch.matmul(diff, torch.matmul(Q, diff))
        action_cost = r * (action ** 2)
        
        # Maximize reward = Minimize cost
        # We do NOT include safety terms here for Lag-PPO
        performance_reward = - (state_cost + action_cost)

        # --- 2. Safety Cost (Constraint) ---
        u_t = torch.tensor([0.5, ego_state[0]]) 
        delta = u_t - agent_mu
        
        try:
            inv_sigma = torch.inverse(agent_sigma)
        except RuntimeError:
            inv_sigma = torch.eye(2) * 100 

        mahalanobis_sq = torch.matmul(delta, torch.matmul(inv_sigma, delta))
        
        epsilon = 0.01
        # Inverse squared distance formulation [cite: 55]
        raw_cost = 1.0 / (mahalanobis_sq + epsilon)
        
        # Clamp cost for stability (max cost ~ 100)
        safety_cost = torch.clamp(raw_cost, max=100.0)

        return performance_reward.item(), safety_cost.item()