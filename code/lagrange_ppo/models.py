import torch
from torch import nn
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=0.5, bias_const=0.0):
    if hasattr(layer, "weight"):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ContinuousActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=5.0):
        super().__init__()
        self.max_action = max_action
        
        # Shared Feature Extractor
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh()
        )
        
        # --- Actor Head (Policy) ---
        self.mu_head = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim)) 
        
        # --- Critic 1: Task Reward Value (V_R) ---
        self.reward_value_head = layer_init(nn.Linear(64, 1), std=1.0)

        # --- Critic 2: Safety Cost Value (V_C) [NEW] ---
        # estimates expected future safety costs
        self.cost_value_head = layer_init(nn.Linear(64, 1), std=1.0)

    def action_value(self, state, action=None):
        """Returns action, log_prob, entropy, V_R, and V_C"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = self.trunk(state)
        
        # Actor
        mu = torch.tanh(self.mu_head(x)) * self.max_action
        std = self.log_std.exp()
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        # Critics
        val_r = self.reward_value_head(x).squeeze(-1)
        val_c = self.cost_value_head(x).squeeze(-1) # Return Cost Value

        return action, log_prob, entropy, val_r, val_c

    @torch.no_grad
    def value(self, state):
        """Returns both (V_R, V_C)"""
        x = self.trunk(state)
        val_r = self.reward_value_head(x).squeeze(-1)
        val_c = self.cost_value_head(x).squeeze(-1)
        return val_r, val_c