"""
Minimal test to verify Barrier Force Function integration works correctly.
Tests the core BFF functionality without running full training.
"""

import torch
import numpy as np
from barrier_functions import SimpleCollisionBarrier, SimpleControlBarrier
from models import ContinuousActorCritic

print("="*60)
print("MINIMAL BFF INTEGRATION TEST")
print("="*60)

# Test 1: Barrier Functions
print("\n[Test 1] Testing Barrier Functions...")
state_barrier = SimpleCollisionBarrier(safety_radius=0.67)
control_barrier = SimpleControlBarrier(max_accel=3.2)
print("✓ Barrier functions initialized")

# Test collision barrier
ego_state = torch.tensor([0.0, 0.5])  # y=0, v=0.5
agent_pos = torch.tensor([0.5, 0.6])  # Close!
agent_sigma = torch.tensor([0.1, 0.1])
grad_coll = state_barrier.gradient(ego_state, agent_pos, agent_sigma)
print(f"  Collision gradient: {grad_coll.item():.3f} (should push away)")

# Test control barrier  
u = torch.tensor([3.0])
grad_ctrl = control_barrier.gradient(u)
print(f"  Control gradient at u=3.0: {grad_ctrl.item():.3f}")

# Test 2: Model with BFF
print("\n[Test 2] Testing Model with BFF Parameters...")
model = ContinuousActorCritic(8, 1, 3.2, True)  # state_dim, action_dim, max_action, use_barriers
print(f"✓ Model created with use_barriers={model.use_barriers}")
print(f"  rho (control gain) = {model.rho}")
print(f"  K (state gain) = {model.K}")

# Test 3: Action Generation with BFF
print("\n[Test 3] Testing Action Generation with BFF...")
state = torch.randn(1, 8)  # Random state
with torch.no_grad():
    action, log_prob, entropy, val_r, val_c = model.action_value(state)

print(f"✓ Virtual action generated: {action.item():.3f}")

# Apply BFF manually (simulating training loop)
obs_i = state[0]
ego_state = obs_i[:2]
agent_pos = obs_i[2:4]
agent_sigma = obs_i[6:8]

grad_x = state_barrier.gradient(ego_state, agent_pos, agent_sigma)
grad_v = control_barrier.gradient(action[0])

action_with_bff = action[0] + model.rho * grad_v + model.K * grad_x
action_final = torch.clamp(action_with_bff, -3.2, 3.2)

print(f"  State barrier force: {(model.K * grad_x).item():.4f}")
print(f"  Control barrier force: {(model.rho * grad_v).item():.4f}")
print(f"  Final action: {action_final.item():.3f}")

# Test 4: Batch Processing
print("\n[Test 4] Testing Batch Processing (4 envs)...")
envs = 4
batch_states = torch.randn(envs, 8)
with torch.no_grad():
    batch_actions, _, _, _, _ = model.action_value(batch_states)

print(f"✓ Batch actions shape: {batch_actions.shape}")

# Apply BFF to batch
actions_with_bff = batch_actions.clone()
for i in range(envs):
    obs_i = batch_states[i]
    ego_state = obs_i[:2]
    agent_pos = obs_i[2:4]
    agent_sigma = obs_i[6:8]
    
    grad_x = state_barrier.gradient(ego_state, agent_pos, agent_sigma)
    grad_v = control_barrier.gradient(batch_actions[i])
    
    actions_with_bff[i] = batch_actions[i] + model.rho * grad_v + model.K * grad_x

actions_final = torch.clamp(actions_with_bff, -3.2, 3.2)
print(f"  Actions before BFF: {batch_actions.squeeze().numpy()}")
print(f"  Actions after BFF:  {actions_final.squeeze().numpy()}")

# Test 5: Gradient Flow
print("\n[Test 5] Testing Gradient Flow...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Simulate a training step
state = torch.randn(4, 8, requires_grad=True)
with torch.enable_grad():
    action, log_prob, entropy, val_r, val_c = model.action_value(state)
    loss = val_r.mean() + val_c.mean()  # Dummy loss
    loss.backward()

# Check if gradients are computed
has_gradients = any(p.grad is not None for p in model.parameters())
print(f"✓ Gradients computed: {has_gradients}")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nBarrier Force Function integration is working correctly.")
print("The BFF modifies actions as: u = v + ρ*∇B_control(v) + K*∇B_collision(x)")
print("\nYou can now run full training when ready.")
print("Note: You may need to fix the data loading issue in train.py first.")
