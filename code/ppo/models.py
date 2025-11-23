import torch
from torch import nn
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=0.5, bias_const=0.0):
    """
    Helper function to initialize layer weights with orthogonal initialization.
    This helps PPO converge faster.
    """
    if hasattr(layer, "weight"):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ContinuousActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=5.0):
        """
        Actor-Critic Network for Continuous Control.
        
        Args:
            state_dim (int): Dimension of observation space (8 in your env).
            action_dim (int): Dimension of action space (1 for acceleration).
            max_action (float): The maximum physical acceleration (e.g., 3.0 m/s^2).
        """
        super().__init__()
        self.max_action = max_action
        
        # Shared feature extractor (Trunk)
        # Increased depth (64 -> 64) to handle the interaction between Ego and Agent better.
        # Using Tanh activation is often more stable for continuous control tasks than ReLU.
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh()
        )
        
        # --- Actor Head (Policy) ---
        # Outputs the Mean (mu) of the action distribution
        self.mu_head = layer_init(nn.Linear(64, action_dim), std=0.01)
        
        # Learnable parameter for Log Standard Deviation
        # Independent of state (standard PPO practice)
        self.log_std = nn.Parameter(torch.zeros(action_dim)) 

        # --- Critic Head (Value) ---
        # Estimates the Value function V(s)
        self.value_head = layer_init(nn.Linear(64, 1), std=1.0)

    def action_value(self, state, action=None):
        """
        Forward pass to get action, log_prob, entropy, and value.
        """ 
        # Handle single state inputs (add batch dimension if needed)
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
            
        # 1. Extract features
        x = self.trunk(state)
        
        # 2. Calculate Mean Action
        # We use Tanh to squash output to [-1, 1], then multiply by max_action
        # Result is in range [-max_action, max_action]
        mu = torch.tanh(self.mu_head(x)) * self.max_action
        
        # 3. Calculate Std Dev
        # .exp() ensures standard deviation is always positive
        std = self.log_std.exp()
        
        # 4. Create Normal Distribution
        dist = Normal(mu, std)

        # 5. Sample Action (if not provided)
        if action is None:
            action = dist.sample()

        # 6. Calculate Log Probability and Entropy
        # Sum over action dimensions (essential if action_dim > 1, harmless if = 1)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        # 7. Calculate Value
        value = self.value_head(x).squeeze(-1)

        # Remove batch dimension if input was single state
        if single:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)
            
        return action, log_prob, entropy, value

    @torch.no_grad
    def value(self, state):
        """
        Helper to get just the value (used during GAE calculation).
        """
        x = self.trunk(state)
        return self.value_head(x).squeeze(-1)