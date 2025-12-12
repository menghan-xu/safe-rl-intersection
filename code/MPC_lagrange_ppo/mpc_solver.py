import casadi as ca
import numpy as np
import torch
class MPCPilot:
    def __init__(self, cfg):
        self.dt = cfg['dt']
        self.N = cfg.get('mpc_horizon', 30) 
        
        # Constraints
        self.v_min = 0.0
        # self.v_max = cfg.get('v_max', 8.0) 
        self.v_max = 2.0
        self.v_limit = cfg['v_limit']
        self.a_min = -cfg['max_accel']
        self.a_max = cfg['max_accel']
        
        # Target Y (Just needs to be past the intersection)
        self.y_target = 1.25
        
        # Dimensions
        self.x_ego_fixed = 0.5
        self.R_ego = cfg['robot_radius']
        self.R_agent = cfg['robot_radius']
        
        # Weights (From YAML)
        self.w_track = cfg.get('mpc_w_track', 1.0)
        self.w_comfort = cfg.get('mpc_w_comfort', 0.01)
        self.w_terminal = cfg.get('mpc_w_terminal', 10.0)
        self.w_safety_slack = cfg.get('mpc_w_safety_slack', 1e9)
        
        # Cache Opti to avoid re-creation overhead (Optional but recommended for speed)
        # For simplicity in training loop, we re-create or keep it simple first.
        # Given training speed matters, we stick to the standard implementation first.
        
    def solve(self, ego_state, agent_future_traj):
        """
        ego_state: [y, v]
        agent_future_traj: List of [x, y, vx, vy, sig_x, sig_y]
        """
        opti = ca.Opti()

        # Variables
        Y = opti.variable(self.N + 1)
        V = opti.variable(self.N + 1)
        A = opti.variable(self.N)
        Slack = opti.variable(self.N + 1)

        y_init = ego_state[0]
        v_init = ego_state[1]

        # Cost
        cost = 0
        for k in range(self.N):
            cost += self.w_track * (V[k] - self.v_limit)**2
            cost += self.w_comfort * A[k]**2
            cost += self.w_safety_slack * Slack[k]**2
            
        opti.minimize(cost)

        # Constraints
        opti.subject_to(Y[0] == y_init)
        opti.subject_to(V[0] == v_init)
        
        for k in range(self.N):
            # Dynamics
            opti.subject_to(Y[k+1] == Y[k] + V[k]*self.dt + 0.5*A[k]*self.dt**2)
            opti.subject_to(V[k+1] == V[k] + A[k]*self.dt)
            
            # Bounds
            opti.subject_to(opti.bounded(self.a_min, A[k], self.a_max))
            opti.subject_to(opti.bounded(self.v_min, V[k+1], self.v_max))
            opti.subject_to(Slack[k] >= 0)

            # Safety (Non-convex)
            # Handle horizon mismatch
            idx = min(k, len(agent_future_traj)-1)
            step_data = agent_future_traj[idx]

            if isinstance(step_data, torch.Tensor):
                step_data = step_data.detach().cpu().numpy()
            elif isinstance(step_data, np.ndarray):
                pass 
            else:
                step_data = np.array(step_data)

            ag_x = float(step_data[0])
            ag_y = float(step_data[1])
            ag_sig_x = float(step_data[4]) 
            ag_sig_y = float(step_data[5])
            
            dist_sq = (self.x_ego_fixed - ag_x)**2 + (Y[k] - ag_y)**2
            sigma_norm = ca.sqrt(ag_sig_x**2 + ag_sig_y**2 + 1e-6) 
            # Add a small margin (0.2) as per previous tuning
            safe_thresh = (self.R_ego + self.R_agent + sigma_norm + 0.2)**2
            
            opti.subject_to(dist_sq >= safe_thresh - Slack[k])

        # Solver
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {'max_iter': 50, 'print_level': 0, 'sb': 'yes'} # Faster iterations for training
        opti.solver('ipopt', p_opts, s_opts)

        try:
            sol = opti.solve()
            return float(sol.value(A[0]))
        except RuntimeError:
            return float(-self.a_max) # Emergency Brake