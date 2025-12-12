import casadi as ca
import numpy as np

class MPCPilot:
    def __init__(self, cfg):
        self.dt = cfg['dt']
        self.N = 30  # Prediction Horizon (20 steps = 2.0 seconds)
        
        # Constraints
        self.v_min = 0.0
        self.v_max = cfg.get('v_max', 2) 
        self.v_limit = cfg['v_limit']      
        self.a_min = -cfg['max_accel']
        self.a_max = cfg['max_accel']
        self.y_target = cfg['target_pos'][0]
        
        # Dimensions for Safety
        self.x_ego_fixed = 0.5
        self.R_ego = 0.3328
        self.R_agent = 0.3328 
        
        # Weights (Tuned to match your RL logic roughly)
        self.w_track = 1.0
        self.w_comfort = 0.01
        self.w_terminal = 5.0
        self.w_safety_slack = 100000000000.0 
        
    def solve(self, ego_state, agent_future_traj):
        """
        ego_state: [y, v]
        agent_future_traj: List/Array of shape (N+1, 6) 
                           Contains [x, y, vx, vy, sigma_x, sigma_y] for the next N steps
        """
        opti = ca.Opti()

        # --- 1. Decision Variables ---
        # State trajectory: y, v
        Y = opti.variable(self.N + 1)
        V = opti.variable(self.N + 1)
        # Control trajectory: a
        A = opti.variable(self.N)
        # Slack variables for safety constraints (soft constraint)
        Slack = opti.variable(self.N + 1)

        # --- 2. Parameters (Current State) ---
        y_init = ego_state[0]
        v_init = ego_state[1]

        # --- 3. Objective Function ---
        # Cost = Tracking + Comfort + Terminal + Safety_Slack_Penalty
        cost = 0
        for k in range(self.N):
            # Tracking cost (v - v_limit)^2
            cost += self.w_track * (V[k] - self.v_limit)**2
            # Comfort cost (a^2)
            cost += self.w_comfort * A[k]**2
            # Soft Constraint Penalty
            cost += self.w_safety_slack * Slack[k]**2
            
        # Terminal cost (reach target)
        # cost += self.w_terminal * (Y[self.N] - self.y_target)**2 
        # (Optional: MPC usually recedes, strict terminal constraint might be too hard)

        opti.minimize(cost)

        # --- 4. Constraints ---
        # Initial State Constraint
        opti.subject_to(Y[0] == y_init)
        opti.subject_to(V[0] == v_init)
        
        for k in range(self.N):
            # A. Dynamics (Double Integrator)
            opti.subject_to(Y[k+1] == Y[k] + V[k]*self.dt + 0.5*A[k]*self.dt**2)
            opti.subject_to(V[k+1] == V[k] + A[k]*self.dt)
            
            # B. State & Actuator Constraints
            opti.subject_to(opti.bounded(self.a_min, A[k], self.a_max))
            opti.subject_to(opti.bounded(self.v_min, V[k+1], self.v_max)) 
            opti.subject_to(Slack[k] >= 0)

            # C. Safety Constraints (The Non-Convex Part)
            # Retrieve agent prediction for step k
            # agent_traj expected format: [x, y, ..., sigma_x, sigma_y]
            # Handle case where horizon is longer than data residue
            idx = min(k, len(agent_future_traj)-1)
            ag_x = agent_future_traj[idx][0]
            ag_y = agent_future_traj[idx][1]
            ag_sig_x = agent_future_traj[idx][4] # index 6 in your readme, but let's adjust index later
            ag_sig_y = agent_future_traj[idx][5]
            
            # d_conservative definition
            dist_sq = (self.x_ego_fixed - ag_x)**2 + (Y[k] - ag_y)**2
            
            # conservative radius (sqrt(sig_x^2 + sig_y^2))
            sigma_norm = ca.sqrt(ag_sig_x**2 + ag_sig_y**2 + 1e-6) 
            safe_thresh = (self.R_ego + self.R_agent + sigma_norm)**2
            
            # Constraint: Dist^2 >= Safe^2 - Slack
            opti.subject_to(dist_sq >= safe_thresh - Slack[k])

        # --- 5. Solver Setup ---
        # IPOPT is the standard interior-point solver for non-convex NLP
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {'max_iter': 100, 'print_level': 0, 'sb': 'yes'} # Silence output
        opti.solver('ipopt', p_opts, s_opts)

        # --- 6. Solve ---
        try:
            sol = opti.solve()
            # Return the first control action
            return sol.value(A[0])
        except RuntimeError:
            print("[MPC Warning] Infeasible! Braking hard.")
            # Fallback: Emergency Brake
            return -self.a_max