import numpy as np
import yaml
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import os

# Custom modules
from env import IntersectionEnv
from lqr_solver import LQRPilot

# ==========================================
# 1. Visualization Helper
# ==========================================
def save_control_gif(ego_traj, agent_traj, ego_v_hist, agent_v_hist, 
                     episode_idx, result_status, min_dist_run, cfg, filename):
    
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 15.0) 
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(f"LQR Controller - Ep {episode_idx}")

    ax.plot([-10, 10], [0, 0], 'k--', alpha=0.3)      # 横向路 (Agent)
    ax.plot([0.5, 0.5], [-10, 20], 'k--', alpha=0.3)  # 纵向路 (Ego)

    r_w = cfg.get('robot_width', 0.43)
    r_l = cfg.get('robot_length', 0.508)
    
    ego_box = Rectangle((0,0), r_w, r_l, color='blue', alpha=0.5, label='Ego (LQR)')
    agent_box = Rectangle((0,0), r_w, r_l, color='red', alpha=0.5, label='Agent')
    ax.add_patch(ego_box)
    ax.add_patch(agent_box)
    ax.legend(loc='lower right')

    text_box = ax.text(-4.5, 14.5, "", fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    frames = max(len(ego_traj), len(agent_traj))

    def update(frame):
        ei = min(frame, len(ego_traj)-1)
        ai = min(frame, len(agent_traj)-1)
        
        ex, ey = ego_traj[ei]
        ax_pos, ay_pos = agent_traj[ai]
        
        # Update Box Positions (Center -> Bottom-Left)
        ego_box.set_xy((ex - r_w/2, ey - r_l/2))
        agent_box.set_xy((ax_pos - r_w/2, ay_pos - r_l/2))

        curr_dist = np.linalg.norm(np.array([ex, ey]) - np.array([ax_pos, ay_pos]))
        
        # Color coding
        collision_dist = cfg.get('collision_dist', 0.66)
        if curr_dist < collision_dist:
            ego_box.set_color('red') 
        else:
            ego_box.set_color('blue')

        info_str = (
            f"Time: {frame * cfg['dt']:.1f} s\n"
            f"Result: {result_status}\n"
            f"Min Dist: {min_dist_run:.3f} m\n"
            f"Curr Dist: {curr_dist:.3f} m\n"
            f"Ego Y: {ey:.2f}\n"
            f"Ego Vel: {ego_v_hist[ei]:.2f}"
        )
        text_box.set_text(info_str)
        return ego_box, agent_box, text_box

    anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=True)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    anim.save(filename, writer=PillowWriter(fps=12))
    plt.close(fig)

# ==========================================
# 2. Main Logic
# ==========================================
def run_baseline():
    # 1. Load Config
    print("Loading configuration from hyperparameters.yaml...")
    with open("hyperparameters.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['save_anim'] = True
    cfg['anim_folder'] = './animations_lqr_yaml'
    cfg['plot_freq'] = 20
    
    # 2. Load Data
    data_path = "../../data/expert_agent_trajs_noise.npy" 
    expert_data = np.load(data_path, allow_pickle=True)

    # 3. Setup Env & LQR
    env = IntersectionEnv(target_pos=cfg['target_pos'], agent_data_list=expert_data, config=cfg)
    lqr = LQRPilot(dt=cfg['dt'], v_target=cfg['v_limit'])
    
    num_episodes = 200
    success_count = 0
    collision_count = 0
    success_times = []
    
    print(f"Running LQR Baseline for {num_episodes} episodes...")

    for i in tqdm(range(num_episodes)):
        env.reset() 
        done = False
        steps = 0
        
        ego_traj = []
        agent_traj = []
        ego_v_hist = []
        agent_v_hist = []
        episode_min_dist = float('inf')
        
        while not done:
            ego_s = env.state[0].item() if torch.is_tensor(env.state) else env.state[0]
            ego_v = env.state[1].item() if torch.is_tensor(env.state) else env.state[1]
            
            ego_fixed_x = 0.5 
            current_ego_xy = np.array([ego_fixed_x, ego_s]) 

            agent_idx = min(env.steps, len(env.current_agent_traj)-1)
            agent_pos_xy = env.current_agent_traj[agent_idx][:2]
            
            action = lqr.get_action(ego_s, ego_v)
            action = np.clip(action, -cfg['max_accel'], cfg['max_accel'])
            
            _, _, done, _ = env.step(action)
            steps += 1
            
            curr_dist = np.linalg.norm(current_ego_xy - agent_pos_xy)
            if curr_dist < episode_min_dist:
                episode_min_dist = curr_dist
            
            ego_traj.append(current_ego_xy)
            agent_traj.append(agent_pos_xy)
            ego_v_hist.append(ego_v)
            agent_v_hist.append(0.0)

        target_y = cfg['target_pos'][1] if isinstance(cfg['target_pos'], (list, tuple)) else cfg['target_pos']
        
        is_collision = episode_min_dist < cfg['collision_dist']
        
        is_reached = env.state[0] >= target_y
        
        is_success = (not is_collision) and is_reached
        
        status = "FAIL"
        if is_collision:
            collision_count += 1
            status = "COLLISION"
        elif is_success:
            success_count += 1
            success_times.append(steps * cfg['dt'])
            status = "SUCCESS"

        if cfg['save_anim'] and (i % cfg['plot_freq'] == 0):
            fname = f"{cfg['anim_folder']}/ep_{i}_{status.lower()}.gif"
            save_control_gif(ego_traj, agent_traj, ego_v_hist, agent_v_hist, 
                             i, status, episode_min_dist, cfg, fname)

    print("\n" + "="*30)
    print("LQR RESULTS (Fixed Y-Axis Control)")
    print("="*30)
    print(f"Success Rate    : {success_count/num_episodes*100:.1f}%")
    print(f"Collision Rate  : {collision_count/num_episodes*100:.1f}%")
    avg_time = np.mean(success_times) if len(success_times) > 0 else 0.0
    print(f"Avg Success Time: {avg_time:.2f} s")
    print("="*30)

if __name__ == "__main__":
    run_baseline()