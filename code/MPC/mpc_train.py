import numpy as np
import yaml
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

# Custom modules
from env import IntersectionEnv
from mpc_solver import MPCPilot # Import the class we just wrote

# --- Visualization Helper (Same as before) ---
def save_mpc_gif(ego_traj, agent_traj, ego_v_hist, agent_v_hist, 
                 episode_idx, result_status, min_dist_run, cfg, filename):
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 15.0) 
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(f"MPC Controller - Ep {episode_idx}")

    ax.plot([-10, 10], [0, 0], 'k--', alpha=0.3)
    ax.plot([0.5, 0.5], [-10, 20], 'k--', alpha=0.3)

    r_w = cfg.get('robot_width', 0.43)
    r_l = cfg.get('robot_length', 0.508)
    
    ego_box = Rectangle((0,0), r_w, r_l, color='green', alpha=0.5, label='Ego (MPC)')
    agent_box = Rectangle((0,0), r_w, r_l, color='red', alpha=0.5, label='Agent')
    ax.add_patch(ego_box)
    ax.add_patch(agent_box)
    ax.legend(loc='lower right')
    
    text_box = ax.text(-4.5, 14.5, "", fontsize=9, bbox=dict(facecolor='white', alpha=0.9))

    frames = max(len(ego_traj), len(agent_traj))

    def update(frame):
        ei = min(frame, len(ego_traj)-1)
        ai = min(frame, len(agent_traj)-1)
        
        ex, ey = ego_traj[ei]
        ax_pos, ay_pos = agent_traj[ai]
        
        ego_box.set_xy((ex - r_w/2, ey - r_l/2))
        agent_box.set_xy((ax_pos - r_w/2, ay_pos - r_l/2))
        
        curr_dist = np.linalg.norm(np.array([ex, ey]) - np.array([ax_pos, ay_pos]))
        
        # Color red if collision
        if curr_dist < cfg['robot_radius'] * 2:
            ego_box.set_color('red')
        else:
            ego_box.set_color('green')

        info_str = (
            f"Time: {frame * cfg['dt']:.1f} s\n"
            f"Result: {result_status}\n"
            f"Min Dist: {min_dist_run:.3f} m\n"
            f"Ego Vel: {ego_v_hist[ei]:.2f}"
        )
        text_box.set_text(info_str)
        return ego_box, agent_box, text_box

    anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=True)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    anim.save(filename, writer=PillowWriter(fps=12))
    plt.close(fig)

def run_mpc():
    # 1. Load Config
    print("Loading configuration...")
    with open("hyperparameters.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Add visualization configs
    cfg['save_anim'] = True
    cfg['anim_folder'] = './animations_mpc'
    cfg['plot_freq'] = 20
    
    # 2. Load Data & Split (Use Test Set!)
    data_path = "../../data/expert_agent_trajs_noise.npy" 
    all_data = np.load(data_path, allow_pickle=True)
    
    # Assuming the same split seed as your training
    total_len = len(all_data)
    indices = np.arange(total_len)
    np.random.seed(42) # Use same seed!
    np.random.shuffle(indices)
    split_idx = int(total_len * 0.8)
    test_data = all_data[indices[split_idx:]] # Only evaluate on test set
    
    print(f"Running MPC on {len(test_data)} test episodes...")

    # 3. Setup Env & MPC
    env = IntersectionEnv(target_pos=cfg['target_pos'], agent_data_list=test_data, config=cfg)
    mpc = MPCPilot(cfg)
    
    success_count = 0
    collision_count = 0
    success_times = []
    
    # Limit number of episodes for baseline to save time
    num_eval_episodes = min(200, len(test_data)) 


    for i in tqdm(range(num_eval_episodes)):
        env.reset()
        done = False
        steps = 0
        
        ego_traj = []
        agent_traj = []
        ego_v_hist = []
        agent_v_hist = []

        is_timeout = False 
        
        final_info = {} 

        while not done:
            # 1. State Extraction
            ego_y = env.state[0].item()
            ego_v = env.state[1].item()
            ego_state = [ego_y, ego_v]
            
            # 2. Future Trajectory (Oracle)
            current_idx = min(env.steps, len(env.current_agent_traj)-1)
            horizon = mpc.N
            future_traj = []
            for k in range(horizon + 1):
                idx = min(current_idx + k, len(env.current_agent_traj)-1)
                future_traj.append(env.current_agent_traj[idx]) 

            # 3. MPC Solve
            action = mpc.solve(ego_state, future_traj)
            
            # 4. Step
            _, _, done, info = env.step(action)
            steps += 1
            final_info = info 
            
            # 5. Logging
            ego_fixed_x = 0.5
            current_ego_xy = np.array([ego_fixed_x, ego_y])
            agent_xy = env.current_agent_traj[current_idx][:2]
            
            ego_traj.append(current_ego_xy)
            agent_traj.append(agent_xy)
            ego_v_hist.append(ego_v)
            agent_v_hist.append(0)

            if steps >= 200: 
                is_timeout = True
                done = True 

        is_success = env.state[0] >= cfg['target_pos'][1]
        
        last_ego_xy = np.array([0.5, env.state[0].item()])
        last_agent_xy = env.current_agent_traj[min(env.steps, len(env.current_agent_traj)-1)][:2]
        final_dist = np.linalg.norm(last_ego_xy - last_agent_xy)
        
        if final_info.get('collision', False) or final_info.get('crash', False):
            is_collision = True
        elif final_info.get('cost', 0) >= 90: 
            is_collision = True
        elif not is_success and not is_timeout:
            is_collision = True

        elif final_dist < cfg['collision_dist']:
            is_collision = True
        else:
            is_collision = False

        if not is_collision and not is_success:
            is_timeout = True

        status = "FAIL"
        if is_collision:
            collision_count += 1
            status = "COLLISION"
        elif is_success:
            success_count += 1
            success_times.append(steps * cfg['dt'])
            status = "SUCCESS"
        elif is_timeout:
            status = "TIMEOUT"
            
        if cfg['save_anim'] and (i % cfg['plot_freq'] == 0):
            fname = f"{cfg['anim_folder']}/mpc_ep_{i}_{status.lower()}.gif"
            save_mpc_gif(ego_traj, agent_traj, ego_v_hist, agent_v_hist, 
                         i, status, final_dist, cfg, fname)


    print("\n" + "="*30)
    print("MPC BASELINE RESULTS")
    print("="*30)
    print(f"Success Rate    : {success_count/num_eval_episodes*100:.1f}%")
    print(f"Collision Rate  : {collision_count/num_eval_episodes*100:.1f}%")
    avg_t = np.mean(success_times) if len(success_times)>0 else 0
    print(f"Avg Success Time: {avg_t:.2f} s")
    print("="*30)

if __name__ == "__main__":
    run_mpc()