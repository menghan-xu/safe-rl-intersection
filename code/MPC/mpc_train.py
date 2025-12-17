import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from tqdm import tqdm
import argparse

# Custom modules
from env import IntersectionEnv
from mpc_solver import MPCPilot

# --- Visualization Helper ---
def save_mpc_gif(ego_traj, agent_traj, ego_v_hist, agent_v_hist, 
                 episode_idx, category, result_status, min_dist_run, total_reward, cfg, filename):
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 11.0) # Expanded view
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(f"MPC | {category} | Ep {episode_idx}")

    # Lanes
    ax.plot([-5, 5], [0, 0], 'k--', alpha=0.3)
    ax.plot([0.5, 0.5], [-5, 15], 'k--', alpha=0.3)

    r_w = cfg.get('robot_width', 0.43)
    r_l = cfg.get('robot_length', 0.508)
    
    ego_box = Rectangle((0,0), r_w, r_l, color='green', alpha=0.5, label='Ego (MPC)')
    agent_box = Rectangle((0,0), r_w, r_l, color='red', alpha=0.5, label='Agent')
    ax.add_patch(ego_box)
    ax.add_patch(agent_box)
    ax.legend(loc='lower right')
    
    # Text box for metrics
    text_box = ax.text(-2.8, 10.5, "", fontsize=9, verticalalignment='top', 
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    frames = max(len(ego_traj), len(agent_traj))

    def update(frame):
        ei = min(frame, len(ego_traj)-1)
        ai = min(frame, len(agent_traj)-1)
        
        ex, ey = ego_traj[ei]
        ax_pos, ay_pos = agent_traj[ai]
        
        ego_box.set_xy((ex - r_w/2, ey - r_l/2))
        agent_box.set_xy((ax_pos - r_w/2, ay_pos - r_l/2))
        
        curr_dist = np.linalg.norm(np.array([ex, ey]) - np.array([ax_pos, ay_pos]))
        
        # Color change on collision
        if curr_dist < cfg['robot_radius'] * 2: # Simple collision check for viz
            ego_box.set_color('red')
            ego_box.set_alpha(0.8)
        else:
            ego_box.set_color('green')
            ego_box.set_alpha(0.5)

        info_str = (
            f"Result: {result_status}\n"
            f"Time  : {frame * cfg['dt']:.1f} s\n"
            f"Dist  : {curr_dist:.2f} m (Min: {min_dist_run:.2f})\n"
            f"Vel   : {ego_v_hist[ei]:.2f} m/s\n"
            f"Reward: {total_reward:.1f}" # We display the RL reward metric here
        )
        text_box.set_text(info_str)
        return ego_box, agent_box, text_box

    anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=True)
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        anim.save(filename, writer=PillowWriter(fps=12))
    except Exception as e:
        print(f"Error saving GIF: {e}")
    plt.close(fig)

def run_mpc_evaluation():
    # 1. Load Config
    print("Loading configuration...")
    with open("hyperparameters.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. Setup Output Directories
    base_dir = "mpc_results"
    success_dir = os.path.join(base_dir, "success")
    failure_dir = os.path.join(base_dir, "failure")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    # 3. Load Test Data
    # NOTE: Adjust this path to where your file actually is relative to this script
    test_data_path = "../../data/expert_agent_trajectories_noise_test_with_zigzag.npy"
    if not os.path.exists(test_data_path):
        test_data_path = "data/expert_agent_trajectories_noise_test_with_zigzag.npy"
    
    print(f"Loading TEST data from {test_data_path}...")
    data_dict = np.load(test_data_path, allow_pickle=True).item()
    trajectories = data_dict['trajectories']
    categories = data_dict['categories']
    
    num_episodes = len(trajectories)
    print(f"Total Test Episodes: {num_episodes}")

    # 4. Initialize Metrics Containers
    # We will store results as a list of dicts to easily split by category later
    all_results = []
    
    # Initialize MPC Solver
    # Note: MPCPilot init might need cfg. Ensure cfg has necessary keys.
    mpc = MPCPilot(cfg) 

    # 5. Evaluation Loop
    for i in tqdm(range(num_episodes), desc="Running MPC"):
        traj = trajectories[i]
        cat = categories[i]
        
        # Create a fresh env for this specific trajectory
        # We pass it as a list containing one trajectory
        env = IntersectionEnv(target_pos=cfg['target_pos'], agent_data_list=[traj], config=cfg)
        
        # obs, _ = env.reset() # Gym 0.26+ returns (obs, info)
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result
        else:
            obs = reset_result
        
        done = False
        steps = 0
        ep_reward = 0.0 # Accumulate RL reward metric
        min_dist = float('inf')
        
        # Trajectory history for GIF
        ego_hist = []
        agent_hist = []
        ego_v_hist = []
        agent_v_hist = [] # Placeholder
        
        while not done:
            # Prepare state for MPC
            # env.state is usually a tensor or array [y, v]
            if hasattr(env.state, 'item'):
                ego_y = env.state[0].item()
                ego_v = env.state[1].item()
            else:
                ego_y = env.state[0]
                ego_v = env.state[1]
                
            ego_state = [ego_y, ego_v]
            
            # Extract Future Trajectory (Oracle for MPC)
            # IntersectionEnv usually stores current agent path in self.current_agent_traj
            current_idx = min(env.steps, len(env.current_agent_traj)-1)
            future_traj = []
            for k in range(mpc.N + 1):
                idx = min(current_idx + k, len(env.current_agent_traj)-1)
                pt = env.current_agent_traj[idx]
                # Ensure it's numpy
                if hasattr(pt, 'numpy'): pt = pt.numpy()
                elif hasattr(pt, 'detach'): pt = pt.detach().cpu().numpy()
                future_traj.append(pt)

            # Solve MPC
            action = mpc.solve(ego_state, future_traj)
            
            # Step Environment
            # IMPORTANT: Capture 'reward' here to calculate Average Reward metric!
            # next_obs, r, term, trunc, info = env.step(action)
            next_obs, r, done, info = env.step(action)
            
            ep_reward += r
            steps += 1
            
            # Logging for GIF
            ego_fixed_x = 0.5
            ego_hist.append([ego_fixed_x, ego_y])
            agent_hist.append(env.current_agent_traj[current_idx][:2])
            ego_v_hist.append(ego_v)
            agent_v_hist.append(0) # Not tracked specifically
            
            # Distance tracking
            dist = np.linalg.norm(np.array([ego_fixed_x, ego_y]) - np.array(agent_hist[-1]))
            if dist < min_dist: min_dist = dist
            
            # done = term or trunc
            
        # --- Determine Outcome ---
        is_success = False
        is_collision = False
        
        # Check Success: Reached target Y
        if env.state[0] >= env.target_pos[0]:
            is_success = True
            
        # Check Collision: Based on distance or Info flag
        if info.get('collision', False) or info.get('crash', False):
            is_collision = True
        elif min_dist < cfg['collision_dist']:
            is_collision = True
            
        # Categorize Result
        status = "TIMEOUT"
        if is_success: status = "SUCCESS"
        if is_collision: status = "COLLISION" # Collision overrides timeout
        
        # --- Save GIF ---
        # Filename: cat_index_status.gif
        safe_cat = str(cat).replace(" ", "_")
        fname = f"{safe_cat}_{i:03d}_{status}.gif"
        
        if is_success:
            full_path = os.path.join(success_dir, fname)
        else:
            full_path = os.path.join(failure_dir, fname)
            
        save_mpc_gif(ego_hist, agent_hist, ego_v_hist, agent_v_hist, 
                     i, cat, status, min_dist, ep_reward, cfg, full_path)
        
        # Record Stats
        all_results.append({
            'category': str(cat),
            'success': is_success,
            'collision': is_collision,
            'reward': ep_reward,
            'time': steps * cfg['dt']
        })

    # 6. Compute and Print Statistics
    print("\n" + "="*60)
    print("MPC EVALUATION REPORT")
    print("="*60)
    
    # 6a. Overall
    total_succ = sum([r['success'] for r in all_results])
    total_coll = sum([r['collision'] for r in all_results])
    avg_rew = np.mean([r['reward'] for r in all_results])
    
    # Avg time (only for successes)
    succ_times = [r['time'] for r in all_results if r['success']]
    avg_time = np.mean(succ_times) if len(succ_times) > 0 else 0.0
    
    print(f"OVERALL ({num_episodes} episodes):")
    print(f"  Success Rate    : {total_succ/num_episodes*100:.1f}%")
    print(f"  Collision Rate  : {total_coll/num_episodes*100:.1f}%")
    print(f"  Avg Reward      : {avg_rew:.2f}")
    print(f"  Avg Time (Succ) : {avg_time:.2f} s")
    print("-" * 60)
    
    # 6b. Per Category
    unique_cats = sorted(list(set(categories)))
    
    # Map raw categories to your paper's names if needed (e.g. 'Turn Left' -> 'Left')
    # Assuming raw names are sufficient for now
    
    for cat in unique_cats:
        cat_res = [r for r in all_results if r['category'] == cat]
        n = len(cat_res)
        if n == 0: continue
        
        c_succ = sum([r['success'] for r in cat_res])
        c_coll = sum([r['collision'] for r in cat_res])
        c_rew = np.mean([r['reward'] for r in cat_res])
        
        c_times = [r['time'] for r in cat_res if r['success']]
        c_time = np.mean(c_times) if len(c_times) > 0 else 0.0
        
        print(f"CATEGORY: {cat} ({n} episodes)")
        print(f"  Success Rate    : {c_succ/n*100:.1f}%")
        print(f"  Collision Rate  : {c_coll/n*100:.1f}%")
        print(f"  Avg Reward      : {c_rew:.2f}")
        print(f"  Avg Time (Succ) : {c_time:.2f} s")
        print("-" * 30)

    print(f"\nGIFs saved to: {base_dir}")

if __name__ == "__main__":
    run_mpc_evaluation()