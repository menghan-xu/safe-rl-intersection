"""
Test script to verify random starting position implementation.
Verifies that ego starts within 15cm radius of (0.5, -1.25).
"""

import numpy as np
import matplotlib.pyplot as plt
from env import IntersectionEnv

# Create dummy config and data
config = {
    'max_accel': 3.2,
    'robot_radius': 0.3328,
    'w_progress': 17.11,
    'w_overspeed': 72.7,
    'w_comfort': 0.697,
    'w_time_penalty': 0.1,
    'reward_success': 50.0,
    'reward_collision': -100.0,
    'cost_scale_mu': 463.2,
    'cost_crash': 100.0,
    'v_limit': 1.5,
    'dt': 0.1
}

# Create dummy agent trajectory data
dummy_trajectory = np.random.randn(100, 8)  # 100 steps, 8 features
dummy_trajectory[:, 0] = 0.5  # x position
dummy_trajectory[:, 1] = np.linspace(-1, 1, 100)  # y position

agent_data = [dummy_trajectory]

# Test random starting positions
print("="*60)
print("RANDOM STARTING POSITION TEST")
print("="*60)
print("\nTesting 1000 resets to verify starting position distribution...")

env = IntersectionEnv(target_pos=[1.5, 0.0], agent_data_list=agent_data, config=config)

starting_positions = []
for i in range(1000):
    obs = env.reset()
    y_start = env.state[0].item()
    starting_positions.append(y_start)

starting_positions = np.array(starting_positions)

# Calculate statistics
nominal_y = -1.25
distances = np.abs(starting_positions - nominal_y)
max_distance = np.max(distances)
mean_distance = np.mean(distances)
std_distance = np.std(distances)

print(f"\nNominal starting position: y = {nominal_y}")
print(f"Expected radius: 0.15m (15cm)")
print(f"\nResults from 1000 resets:")
print(f"  Min y: {np.min(starting_positions):.4f}")
print(f"  Max y: {np.max(starting_positions):.4f}")
print(f"  Mean y: {np.mean(starting_positions):.4f}")
print(f"  Std y: {np.std(starting_positions):.4f}")
print(f"\nDistance from nominal:")
print(f"  Max distance: {max_distance:.4f}m ({max_distance*100:.2f}cm)")
print(f"  Mean distance: {mean_distance:.4f}m ({mean_distance*100:.2f}cm)")
print(f"  Std distance: {std_distance:.4f}m ({std_distance*100:.2f}cm)")

# Verify all positions are within 15cm
within_radius = np.all(distances <= 0.15)
print(f"\n{'✓' if within_radius else '✗'} All positions within 15cm: {within_radius}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Histogram
ax1.hist(starting_positions, bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(nominal_y, color='red', linestyle='--', linewidth=2, label='Nominal (-1.25)')
ax1.axvline(nominal_y - 0.15, color='green', linestyle='--', linewidth=1, label='±15cm bounds')
ax1.axvline(nominal_y + 0.15, color='green', linestyle='--', linewidth=1)
ax1.set_xlabel('Starting Y Position (m)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Random Starting Positions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: 2D scatter (showing full circle)
# Generate x offsets for visualization (in reality, x is fixed at 0.5)
# For visualization, we'll show the theoretical circle
circle = plt.Circle((0.5, -1.25), 0.15, color='green', fill=False, linewidth=2, label='15cm radius')
ax2.add_patch(circle)
ax2.plot(0.5, -1.25, 'ro', markersize=10, label='Nominal position')
ax2.scatter([0.5]*len(starting_positions), starting_positions, alpha=0.3, s=5, label='Sampled positions')
ax2.set_xlim(0.2, 0.8)
ax2.set_ylim(-1.5, -1.0)
ax2.set_xlabel('X Position (m)')
ax2.set_ylabel('Y Position (m)')
ax2.set_title('Starting Position Distribution (Y-axis only)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('random_start_test.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: random_start_test.png")

print("\n" + "="*60)
if within_radius and mean_distance < 0.09:
    print("✓ TEST PASSED!")
    print("Random starting position is working correctly.")
else:
    print("✗ TEST FAILED!")
    print("Check the implementation.")
print("="*60)
