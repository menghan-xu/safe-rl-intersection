import torch
from torch import nn
from torch.distributions.normal import Normal
import numpy as np

# Initialize layers with orthogonal weights and constant biases (this is a placeholder)
# Immediately replaced with the trained weight during deployment
def layer_init(layer, std=0.5, bias_const=0.0):
    if hasattr(layer, "weight"):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ContinuousActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=3.2):
        super().__init__()
        self.max_action = max_action
        
        # Shared Feature Extractor
            # input: states [y_ego, v_ego, x_agent, y_agent, vx_agent, vy_agent, sigmax, sigmay]
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)), # first layer -- extract basic features -- weights x input_states + bias 
            nn.Tanh(), # activation, squashes values from be between -1 and 1
            layer_init(nn.Linear(64, 64)), # input = 64 features, output = 64 refined features 
            nn.Tanh() # activation 
        )
        
        # INITIALIZE ACTOR AND CRITIC HEADS 
        # --- Actor Head (Policy) ---
        self.mu_head = layer_init(nn.Linear(64, action_dim), std=0.01) 
        self.log_std = nn.Parameter(torch.zeros(action_dim)) 
        
        # --- Critic 1: Task Reward Value (V_R) ---
        self.reward_value_head = layer_init(nn.Linear(64, 1), std=1.0)

        # --- Critic 2: Safety Cost Value (V_C) ---
        self.cost_value_head = layer_init(nn.Linear(64, 1), std=1.0)

    def action_value(self, state, action=None):
        """Returns action, log_prob, entropy, V_R, and V_C"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = self.trunk(state)
        
        # Actor -- build the policy distribution 
        mu = torch.tanh(self.mu_head(x)) * self.max_action
        std = self.log_std.exp()
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()

        # Sum log probs if action_dim > 1 (here it is 1, so sum changes nothing but is safe)
        log_prob = dist.log_prob(action).sum(-1) # log probability of the action under the policy
        entropy = dist.entropy().sum(-1) # entropy of the policy distribution -- measures the uncertainty in the action selection; negative log expectation
        
        # Critics
        val_r = self.reward_value_head(x).squeeze(-1)
        val_c = self.cost_value_head(x).squeeze(-1)

        return action, log_prob, entropy, val_r, val_c

    @torch.no_grad()
    def value(self, state):
        """Returns both (V_R, V_C)"""
        x = self.trunk(state)
        val_r = self.reward_value_head(x).squeeze(-1)
        val_c = self.cost_value_head(x).squeeze(-1)
        return val_r, val_c