import numpy as np
import scipy.linalg

class LQRPilot:
    def __init__(self, dt=0.1, v_target=1.5):
        self.dt = dt
        self.v_target = v_target
        
        # 1. Dynamics: x_{t+1} = A x_t + B u_t
        # State: [y_ego, v_ego]
        self.A = np.array([
            [1.0, self.dt],
            [0.0, 1.0]
        ])
        
        self.B = np.array([
            [0.5 * self.dt**2],
            [self.dt]
        ])
        
        # 2. Costs
        # We only care about velocity tracking. 
        # q_pos = 0 (ignore position), q_vel = 10 (track velocity strictly)
        self.Q = np.array([
            [0.0, 0.0],
            [0.0, 5.0] 
        ])
        # r_accel = 1 (penalty for acceleration/jerk)
        self.R = np.array([[1.0]])
        
        # 3. Solve DARE (Discrete Algebraic Riccati Equation)
        P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        
        # 4. Compute Gain K
        # K = (R + B^T P B)^-1 (B^T P A)
        self.K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        
    def get_action(self, ego_y, ego_v):
        """
        Calculates u = -K(x - x_ref).
        x_ref is dynamically set to [current_y, target_v] to ignore position error.
        """
        # We pretend the target position is exactly where we are, 
        # so the position error is always 0.
        y_err = 0.0 
        v_err = ego_v - self.v_target
        
        error = np.array([[y_err], [v_err]])
        
        u = -self.K @ error
        return u.item()

    def get_action_batch(self, ego_y_batch, ego_v_batch):
        """
        Vectorized version for multiple environments.
        ego_y_batch: shape (N,)
        ego_v_batch: shape (N,)
        """
        # error shape: (2, N)
        v_err = ego_v_batch - self.v_target
        y_err = np.zeros_like(v_err)
        
        error = np.stack([y_err, v_err], axis=0) 
        
        # K shape (1, 2) @ (2, N) -> (1, N)
        u = -self.K @ error
        return u.flatten() # Returns (N,)