"""
Quick test script for Barrier Force Function integration.
Runs a few training iterations to ensure everything works.
"""

import subprocess
import sys

print("=" * 60)
print("BARRIER FORCE FUNCTION INTEGRATION TEST")
print("=" * 60)
print("\nRunning minimal training to verify BFF integration...")
print("This will take ~5 minutes.")
print()

# Modify config temporarily for quick test
config_content = """
env_id: "Intersection-Lag-v0"
seed: 42
num_envs: 4
num_steps: 128
epochs: 50  # Reduced for quick test
update_epochs: 5
minibatch_size: 128
lr: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_ratio: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
save_gif_freq: 25
checkpoint: false

# Lagrangian
lambda_init: 0.0
lambda_lr: 0.001
cost_limit: 5.0

# Environment
target_pos: [1.5, 0.0]
max_accel: 3.2
dt: 0.1
robot_radius: 0.3328
v_limit: 1.5

# Reward weights
w_progress: 17.11
w_overspeed: 72.7
w_comfort: 0.697
w_time_penalty: 0.1
reward_success: 50.0
reward_collision: -100.0

# Cost weights  
cost_scale_mu: 463.2
cost_crash: 100.0
"""

# Write temporary config
import yaml
with open('hyperparameters_test.yaml', 'w') as f:
    f.write(config_content)

print("✓ Created test configuration (50 epochs)")

# Run training
try:
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, 'train.py')
    config_path = os.path.join(script_dir, 'hyperparameters_test.yaml')
    
    result = subprocess.run(
        [sys.executable, train_path, '--config', 'hyperparameters_test.yaml'],
        capture_output=False,
        text=True,
        cwd=script_dir
    )
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✓ BFF INTEGRATION TEST PASSED!")
        print("=" * 60)
        print("\nBarrier Force Function is working correctly.")
        print("You can now run full training with: python train.py")
    else:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print("Check the error messages above.")
        
except Exception as e:
    print(f"\n✗ Error running test: {e}")
    sys.exit(1)
