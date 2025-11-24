import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # Import for visualization

# Import your custom environment and model
from env import IntersectionEnv
from models import ContinuousActorCritic

# ==========================================
# CONFIGURATION
# ==========================================
# Set your best model path here directly
BEST_MODEL_PATH = "learned_policies/Intersection-Lag-v0_2025-11-23_22-57-50/model_1000.pt" 
DATA_PATH = "../../data/expert_agent_trajs.npy"
NUM_EPISODES = 100
# If True, it will pop up a window to show the animation every 20 episodes
SHOW_ANIMATION = True  

def get_deterministic_action(model, obs):
    """
    Evaluation Helper:
    We typically want the MEAN of the policy distribution (Deterministic),
    rather than sampling from the Normal distribution. This reduces variance in testing.
    """
    with torch.no_grad():
        # 1. Extract features
        x = model.trunk(obs)
        # 2. Directly calculate Mean (skip Normal sampling)
        mu = torch.tanh(model.mu_head(x)) * model.max_action
        return mu.cpu().numpy().item()

def view_animation(policy, env, device):
    """
    Runs one episode and opens a window to visualize it immediately.
    No files are saved.
    """
    print("Generating visual replay...")
    
    # 1. Collect Data (Same logic as evaluation loop)
    obs = env.reset()
    obs_tensor = obs.float().to(device).unsqueeze(0)
    
    ego_pos_hist = []
    agent_pos_hist = []
    
    done = False
    while not done:
        # Get Render Info
        render_info = env.render() 
        ego_pos_hist.append(render_info['ego_pos'])
        agent_pos_hist.append(render_info['agent_pos'])
        
        # Get Action (Deterministic for visual consistency)
        action = get_deterministic_action(policy, obs_tensor)
        
        # Step Environment
        next_obs, _, done, _ = env.step(action)
        
       
        obs_tensor = next_obs.float().to(device).unsqueeze(0)

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(6, 6)) 
    
    # Zoom in to the intersection area
    ax.set_xlim(-2, 2) 
    ax.set_ylim(-2, 2) 
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f"Live Replay (Length: {len(ego_pos_hist)})")

    # Draw Lanes (Static)
    ax.plot([-2, 2], [0, 0], 'k--', alpha=0.3, lw=1) # Horizontal
    ax.plot([0.5, 0.5], [-2, 2], 'k--', alpha=0.3, lw=1) # Vertical

    # Initialize Objects
    ego_dot, = ax.plot([], [], 'bo', label='Ego', markersize=12, zorder=5)
    agent_dot, = ax.plot([], [], 'ro', label='Agent', markersize=12, zorder=5)
    ego_line, = ax.plot([], [], 'b-', alpha=0.3, lw=2)
    agent_line, = ax.plot([], [], 'r-', alpha=0.3, lw=2)
    
    ax.legend(loc='lower right')

    def update(frame):
        # Update Positions
        e_pos = ego_pos_hist[frame]
        a_pos = agent_pos_hist[frame]
        
        ego_dot.set_data([e_pos[0]], [e_pos[1]])
        agent_dot.set_data([a_pos[0]], [a_pos[1]])
        
        # Update Trails
        e_path = np.array(ego_pos_hist[:frame+1])
        a_path = np.array(agent_pos_hist[:frame+1])
        ego_line.set_data(e_path[:, 0], e_path[:, 1])
        agent_line.set_data(a_path[:, 0], a_path[:, 1])
        
        return ego_dot, agent_dot, ego_line, agent_line

    # Create Animation
    anim = FuncAnimation(fig, update, frames=len(ego_pos_hist), interval=50, blit=True, repeat=False)
    
    # Show the window! (Script pauses here until you close the window)
    plt.show()

def evaluate():
    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    expert_data = np.load(DATA_PATH, allow_pickle=True)

    # 3. Initialize Environment
    env = IntersectionEnv(target_pos=[10.0, 0.0], agent_data_list=expert_data, dt=0.1)

    # 4. Initialize Model Architecture
    model = ContinuousActorCritic(state_dim=8, action_dim=1, max_action=1.0).to(device)
    
    # 5. Load Model Weights (Checkpoint)
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Model file {BEST_MODEL_PATH} not found.")
        return
    
    print(f"Loading model from: {BEST_MODEL_PATH}")
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() 

    # 6. Start Evaluation Loop
    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_rewards = []
    min_distances = []

    print(f"Starting evaluation over {NUM_EPISODES} episodes...")
    
    for i in tqdm(range(NUM_EPISODES)):
        obs = env.reset()
        obs = obs.float().to(device).unsqueeze(0)
        
        done = False
        episode_reward = 0
        min_dist_this_ep = float('inf')
        
        while not done:
            # Get deterministic action
            action = get_deterministic_action(model, obs)
            
            # Interaction
            obs_next, reward, done, info = env.step(action)
            
            # Record metrics
            episode_reward += reward
            
            # Calculate Euclidean Distance
            render_info = env.render()
            ego_pos = np.array(render_info['ego_pos'])
            agent_pos = np.array(render_info['agent_pos'])
            dist = np.linalg.norm(ego_pos - agent_pos)
            if dist < min_dist_this_ep:
                min_dist_this_ep = dist
            
            # Update Obs
            obs = obs_next.float().to(device).unsqueeze(0)

            # Analyze Result (based on final state)
            if done:
                if env.state[0] >= 10.0:
                    success_count += 1
                # Check Collision (Threshold 0.8m)
                elif min_dist_this_ep < 0.8: 
                    collision_count += 1
                else:
                    timeout_count += 1

        total_rewards.append(episode_reward)
        min_distances.append(min_dist_this_ep)

        # --- Visualization ---
        # Pop up a window to show the replay
        if SHOW_ANIMATION and (i % 20 == 0):
            # Pass the raw env (not Gym wrapper)
            view_animation(model, env, device)

    # 7. Print Statistics
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / NUM_EPISODES * 100
    collision_rate = collision_count / NUM_EPISODES * 100
    timeout_rate = timeout_count / NUM_EPISODES * 100
    avg_min_dist = np.mean(min_distances)

    print("\n" + "="*40)
    print("EVALUATION REPORT")
    print("="*40)
    print(f"Model: {BEST_MODEL_PATH}")
    print(f"Episodes: {NUM_EPISODES}")
    print("-" * 40)
    print(f"Success Rate   : {success_rate:.2f}%")
    print(f"Collision Rate : {collision_rate:.2f}% (Dist < 0.8m)")
    print(f"Timeout/Fail   : {timeout_rate:.2f}%")
    print("-" * 40)
    print(f"Avg Reward     : {avg_reward:.2f}")
    print(f"Avg Min Dist   : {avg_min_dist:.2f} m")
    print("="*40)

if __name__ == "__main__":
    evaluate()