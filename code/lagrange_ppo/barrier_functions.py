"""
Barrier Force Functions for Safe RL
Implements simple collision and control barriers with numerical stability.
Based on "Model-Based Safe Reinforcement Learning with Time-Varying State and Control Constraints"
"""

import torch
import numpy as np


class SimpleCollisionBarrier:
    """
    State barrier for collision avoidance.
    Constraint: G(x) = r_safe^2 - d^2 <= 0
    where r_safe includes uncertainty and d is distance to obstacle.
    """
    def __init__(self, safety_radius=0.67, epsilon=0.1):
        """
        Args:
            safety_radius: Base collision radius (2 * robot_radius)
            epsilon: Relaxation factor for numerical stability
        """
        self.safety_radius = safety_radius
        self.eps = epsilon  # Relaxation for stability
    
    def gradient(self, ego_state, agent_pos, agent_sigma):
        """
        Compute barrier gradient for collision avoidance.
        
        Args:
            ego_state: Tensor [y_ego, v_ego] - ego vehicle state
            agent_pos: Tensor [x_agent, y_agent] - obstacle position
            agent_sigma: Tensor [sigma_x, sigma_y] - obstacle uncertainty
            
        Returns:
            Tensor [grad_y] - gradient wrt ego y-position
        """
        # Ego position (x fixed at 0.5)
        if ego_state.dim() > 0:
            y_ego = ego_state[0]
        else:
            y_ego = ego_state
        ego_pos = torch.tensor([0.5, y_ego], dtype=torch.float32, device=ego_state.device)
        
        # Conservative safety radius (includes uncertainty)
        if agent_sigma is not None and agent_sigma.numel() > 0:
            sigma_total = torch.norm(agent_sigma)
        else:
            sigma_total = torch.tensor(0.0, device=ego_state.device)
        r_safe = self.safety_radius + sigma_total
        
        # Distance to agent
        diff = ego_pos - agent_pos[:2]
        d = torch.norm(diff)
        
        # Constraint function: G = r_safe^2 - d^2
        # When G > 0: violation, G < 0: safe
        G = r_safe**2 - d**2
        
        # Only apply barrier force if approaching constraint
        if G < -self.eps:
            return torch.zeros(1, device=ego_state.device)  # Safe, no force
        
        # Gradient of G wrt ego_pos: dG/d(ego_pos) = -2*(ego_pos - agent_pos)
        grad_G = -2.0 * diff
        
        # Barrier gradient: grad_B = -grad_G / G
        # Clamp G for numerical stability
        G_safe = torch.clamp(G, min=-10.0, max=-self.eps)
        grad_B = -grad_G / G_safe
        
        # Return only y-component (ego only moves in y direction)
        return grad_B[1:2]


class SimpleControlBarrier:
    """
    Control barrier for acceleration limits.
    Box constraints: -u_max <= u <= u_max
    """
    def __init__(self, max_accel=3.2, epsilon=0.1):
        """
        Args:
            max_accel: Maximum acceleration magnitude
            epsilon: Relaxation factor for numerical stability
        """
        self.u_max = max_accel
        self.eps = epsilon
    
    def gradient(self, u):
        """
        Compute barrier gradient for control constraints.
        
        Args:
            u: Tensor (scalar or 1D) - control action
            
        Returns:
            Tensor [grad_u] - gradient wrt control
        """
        # Extract scalar value
        if u.dim() == 0:
            u_val = u.item()
        else:
            u_val = u[0].item()
        
        device = u.device if hasattr(u, 'device') else torch.device('cpu')
        
        # Upper bound constraint: G1 = u - u_max <= 0
        if u_val > self.u_max - self.eps:
            G = u_val - self.u_max
            # Barrier gradient: positive force to push down
            return torch.tensor([1.0 / max(-G, self.eps)], device=device)
        
        # Lower bound constraint: G2 = -u - u_max <= 0
        elif u_val < -self.u_max + self.eps:
            G = -u_val - self.u_max
            # Barrier gradient: negative force to push up
            return torch.tensor([-1.0 / max(-G, self.eps)], device=device)
        
        else:
            # Well within bounds, no force needed
            return torch.zeros(1, device=device)


if __name__ == "__main__":
    # Quick test
    print("Testing SimpleCollisionBarrier...")
    collision_barrier = SimpleCollisionBarrier(safety_radius=0.67)
    
    ego_state = torch.tensor([0.0, 0.5])  # y=0, v=0.5
    agent_pos = torch.tensor([0.5, 0.5])  # Close to ego!
    agent_sigma = torch.tensor([0.1, 0.1])
    
    grad = collision_barrier.gradient(ego_state, agent_pos, agent_sigma)
    print(f"Collision barrier gradient: {grad}")
    print(f"Expected: positive (push ego away from agent)")
    
    print("\nTesting SimpleControlBarrier...")
    control_barrier = SimpleControlBarrier(max_accel=3.2)
    
    u_high = torch.tensor([3.1])  # Near upper limit
    grad_high = control_barrier.gradient(u_high)
    print(f"Control at {u_high.item()}: gradient = {grad_high.item()}")
    print(f"Expected: positive (pushes control down)")
    
    u_low = torch.tensor([-3.1])  # Near lower limit
    grad_low = control_barrier.gradient(u_low)
    print(f"Control at {u_low.item()}: gradient = {grad_low.item()}")
    print(f"Expected: negative (pushes control up)")
    
    u_ok = torch.tensor([1.0])  # Safe
    grad_ok = control_barrier.gradient(u_ok)
    print(f"Control at {u_ok.item()}: gradient = {grad_ok.item()}")
    print(f"Expected: ~0 (no force needed)")
