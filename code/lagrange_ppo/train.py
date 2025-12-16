import os
import yaml 
import datetime
import pathlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

# Imports
from env import IntersectionEnv
from models import ContinuousActorCritic

# ==========================================
# 1. Gym Adapter
# ==========================================
class IntersectionGymAdapter(gym.Env):
    def __init__(self, agent_data_list, config):
        self.env = IntersectionEnv(
            target_pos=config['target_pos'], 
            agent_data_list=agent_data_list, 
            config=config, 
            dt=config['dt']
        ) 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        limit = config['max_accel']
        self.action_space = gym.spaces.Box(low=-limit, high=limit, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return obs.numpy(), {} 

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        obs, reward, done, info = self.env.step(action)
        return obs.numpy(), reward, done, False, info

# ==========================================
# 2. Utils
# ==========================================
def make_animation(policy, env, device, filename):
    """
    Generates a GIF with detailed Real-time Dashboard.
    Displays: Reward components, Safety Cost, Velocities, and Distance.
    """
    obs, _ = env.reset()
    obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
    
    # History lists
    ego_pos_hist = []
    agent_pos_hist = []
    ego_v_hist = []
    agent_v_hist = []
    
    # Metric history (Store the info dict for each step)
    metric_hist = []
    
    done = False
    while not done:
        render_info = env.env.render()
        ego_pos_hist.append(render_info['ego_pos'])
        agent_pos_hist.append(render_info['agent_pos'])
        ego_v_hist.append(render_info['ego_v'])
        agent_v_hist.append(render_info['agent_v'])
        
        with torch.no_grad():
            action, _, _, _, _ = policy.action_value(obs_tensor)
        action_scalar = action.cpu().numpy().item()
        
        # Step (Capture info!)
        next_obs, _, terminated, truncated, info = env.step(action_scalar)
        
        # Store metrics for this step
        metric_hist.append(info)
        
        done = terminated or truncated
        obs_tensor = torch.from_numpy(next_obs).float().to(device).unsqueeze(0)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(6, 7)) # Taller figure for text box
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 8.5) # Expanded view to see goal
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title("Intersection Lag-PPO Analysis")
    
    # Lanes
    ax.plot([-5, 5], [0, 0], 'k--', alpha=0.3)
    ax.plot([0.5, 0.5], [-5, 15], 'k--', alpha=0.3)

    # Objects
    ego_dot, = ax.plot([], [], 'bo', label='Ego', markersize=8)
    agent_dot, = ax.plot([], [], 'ro', label='Agent', markersize=8)
    
    ROBOT_W = 0.43
    ROBOT_L = 0.508
    ego_box = Rectangle((0,0), ROBOT_W, ROBOT_L, angle=0.0, color='blue', alpha=0.2)
    agent_box = Rectangle((0,0), ROBOT_W, ROBOT_L, angle=0.0, color='red', alpha=0.2)
    ax.add_patch(ego_box)
    ax.add_patch(agent_box)

    # Detailed Text Box (Positioned Top Left)
    text_box = ax.text(
        -2.8, 8.2, "", 
        fontsize=9, 
        verticalalignment='top', 
        fontfamily='monospace',
        bbox=dict(facecolor='white', alpha=0.95, edgecolor='gray', boxstyle='round')
    )
    
    ax.legend(loc='lower right')
    
    def update(frame):
        e_x, e_y = ego_pos_hist[frame]
        a_x, a_y = agent_pos_hist[frame]
        
        # 1. Update Visuals
        ego_dot.set_data([e_x], [e_y])
        agent_dot.set_data([a_x], [a_y])
        
        ego_box.set_xy((e_x - ROBOT_W/2, e_y - ROBOT_L/2))
        agent_box.set_xy((a_x - ROBOT_W/2, a_y - ROBOT_L/2))
        
        # 2. Get metrics
        # Handle case where metric_hist might be 1 step shorter or longer depending on loop
        idx = min(frame, len(metric_hist)-1)
        m = metric_hist[idx]
        
        # Calculated Distance
        dist_act = np.linalg.norm(np.array([e_x, e_y]) - np.array([a_x, a_y]))
        
        # Retrieved Metrics from Env (using .get with defaults)
        d_cons = m.get('d_cons', 0.0)
        tot_rew = m.get('total_reward', 0.0)
        cost = m.get('cost', 0.0) # Or final_cost
        
        r_prog = m.get('r_prog', 0.0)
        r_spd = m.get('r_speed', 0.0)
        r_cmf = m.get('r_comf', 0.0)
        bonus = m.get('bonus', 0.0)

        # 3. Format Dashboard
        info_str = (
            f"Step: {frame:03d}\n"
            f"=== SAFETY ===\n"
            f"d_actual : {dist_act:.3f} m\n"
            f"d_consv  : {d_cons:.3f} m\n"
            f"SafeCost : {cost:.2f}\n"
            f"=== REWARD ({tot_rew:.2f}) ===\n"
            f"Progress : {r_prog:+.2f}\n"
            f"SpeedPen : {r_spd:+.2f}\n"
            f"Comfort  : {r_cmf:+.2f}\n"
            f"Bonus    : {bonus:+.0f}\n"
            f"=== STATE ===\n"
            f"Ego Vel  : {ego_v_hist[frame]:.2f}"
        )
        text_box.set_text(info_str)
        
        # Visual Alert: Red border if distance < collision
        bbox = text_box.get_bbox_patch()
        if bbox:
            if dist_act < 0.67: 
                bbox.set_edgecolor('red')
                bbox.set_linewidth(2)
            else:
                bbox.set_edgecolor('gray')
                bbox.set_linewidth(1)
            
        return ego_dot, agent_dot, ego_box, agent_box, text_box

    anim = FuncAnimation(fig, update, frames=len(ego_pos_hist), interval=80, blit=True)
    anim.save(filename, writer=PillowWriter(fps=12))
    plt.close(fig)

def compute_gae(rewards, values, dones, gamma, lam):
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    return adv, adv + values[:-1]

# Return dict of losses for logging
def lagrangian_loss(agent, states, actions, adv, logp, ret_r, ret_c, clip, ent_c, vf_c):
    _, new_logp, entropy, val_r, val_c = agent.action_value(states, actions)
    ratio = torch.exp(new_logp - logp)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1-clip, 1+clip) * adv
    
    loss_pi = -torch.min(surr1, surr2).mean()
    loss_vr = (val_r - ret_r).pow(2).mean()
    loss_vc = (val_c - ret_c).pow(2).mean() # Safety Critic Loss
    loss_ent = -ent_c * entropy.mean()
    
    total_loss = loss_pi + vf_c * (loss_vr + loss_vc) + loss_ent
    
    return total_loss, {
        "loss_pi": loss_pi.item(),
        "loss_vr": loss_vr.item(),
        "loss_vc": loss_vc.item(),
        "loss_ent": loss_ent.item(),
        "loss_total": total_loss.item()
    }

def evaluate(model, env, device, num_ep=100, save_gifs=False, save_dir=None):
    model.eval()
    success_count = 0
    collision_count = 0
    total_r = []
    total_value_loss = [] 
    
    saved_failure_gifs = 0
    MAX_FAILURE_GIFS = 20
    success_times = []
    for ep_i in range(num_ep):
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        done = False
        ep_r = 0
        min_dist = float('inf')
        ep_steps = 0
        ep_values = []
        ep_rewards = []
        
        #  Cache frames for potential GIF generation
        frames_data = [] 
        
        while not done:
            # 1. Record Frame Data (Before step)
            if save_gifs:
                render_info = env.env.render()
                frames_data.append({
                    'ego_pos': render_info['ego_pos'],
                    'agent_pos': render_info['agent_pos'],
                    'ego_v': render_info['ego_v'],
                    'agent_v': render_info['agent_v']
                })

            # 2. Get Action
            with torch.no_grad():
                x = model.trunk(obs)
                action = torch.tanh(model.mu_head(x)) * model.max_action
                val_r = model.reward_value_head(x)
                ep_values.append(val_r.item())
                action_scalar = action.cpu().numpy().item()
            
            # 3. Step
            next_obs, r, term, trunc, info = env.step(action_scalar)
            
            #  Cache extra info for dashboard
            if save_gifs:
                frames_data[-1]['info'] = info # Append metrics to the last frame
            
            ep_r += r
            ep_rewards.append(r)
            ep_steps += 1
            # 4. Metrics
            render_info = env.env.render() # Get updated state
            e_pos = np.array(render_info['ego_pos'])
            a_pos = np.array(render_info['agent_pos'])
            dist = np.linalg.norm(e_pos - a_pos)
            if dist < min_dist: min_dist = dist
            
            done = term or trunc
            obs = torch.from_numpy(next_obs).float().to(device).unsqueeze(0)
            
            # 5. Outcome Check
            is_success = False
            is_collision = False
            if done:
                if env.env.state[0] >= env.env.target_pos[0]: 
                    success_count += 1
                    is_success = True
                    success_times.append(ep_steps * env.env.dt)
                if min_dist < env.env.collision_dist: 
                    collision_count += 1
                    is_collision = True

        # ---  Conditional GIF Saving ---
        # Save only if FAILED (Collision or Timeout/Out-of-bounds)
        if save_gifs and save_dir and saved_failure_gifs < MAX_FAILURE_GIFS:
            if not is_success: # Capture both collision and timeout
                filename = f"{save_dir}/fail_ep_{ep_i}_col_{is_collision}.gif"
                save_cached_animation(frames_data, filename)
                saved_failure_gifs += 1

        # Compute Value Loss
        returns = []
        G = 0
        for r in reversed(ep_rewards):
            G = r + 0.99 * G 
            returns.insert(0, G)
        if len(ep_values) > 0:
            v_loss = np.mean([(v - ret)**2 for v, ret in zip(ep_values, returns)])
            total_value_loss.append(v_loss)
        total_r.append(ep_r)
    if len(success_times) > 0:
        avg_time = np.mean(success_times)
    else:
        avg_time = 0.0 # No success, no time
    model.train()
    avg_v_loss = np.mean(total_value_loss) if len(total_value_loss) > 0 else 0.0
    return np.mean(total_r), success_count/num_ep, collision_count/num_ep, avg_v_loss, avg_time

#  Helper function to save GIF from cached data
def save_cached_animation(frames_data, filename):
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 11.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(f"Failure Case Replay")
    
    ax.plot([-5, 5], [0, 0], 'k--', alpha=0.3)
    ax.plot([0.5, 0.5], [-5, 15], 'k--', alpha=0.3)

    ego_dot, = ax.plot([], [], 'bo', label='Ego', markersize=8)
    agent_dot, = ax.plot([], [], 'ro', label='Agent', markersize=8)
    
    ROBOT_W, ROBOT_L = 0.43, 0.508
    ego_box = Rectangle((0,0), ROBOT_W, ROBOT_L, angle=0.0, color='blue', alpha=0.2)
    agent_box = Rectangle((0,0), ROBOT_W, ROBOT_L, angle=0.0, color='red', alpha=0.2)
    ax.add_patch(ego_box); ax.add_patch(agent_box)

    text_box = ax.text(-2.8, 10.5, "", fontsize=8.5, verticalalignment='top', 
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    def update(frame_idx):
        data = frames_data[frame_idx]
        e_x, e_y = data['ego_pos']
        a_x, a_y = data['agent_pos']
        
        ego_dot.set_data([e_x], [e_y])
        agent_dot.set_data([a_x], [a_y])
        ego_box.set_xy((e_x - ROBOT_W/2, e_y - ROBOT_L/2))
        agent_box.set_xy((a_x - ROBOT_W/2, a_y - ROBOT_L/2))
        
        dist = np.linalg.norm(np.array([e_x, e_y]) - np.array([a_x, a_y]))
        m = data.get('info', {})
        
        info_str = (
            f"Step: {frame_idx}\nDist: {dist:.3f} m\n"
            f"Ego V: {data['ego_v']:.2f}\nAgent V: {data['agent_v']:.2f}\n"
            f"R_Prog: {m.get('r_prog',0):.2f}\nCost: {m.get('cost',0):.1f}"
        )
        text_box.set_text(info_str)
        
        bbox = text_box.get_bbox_patch()
        if bbox:
            bbox.set_edgecolor('red' if dist < 0.67 else 'gray')
            bbox.set_linewidth(2 if dist < 0.67 else 1)
            
        return ego_dot, agent_dot, ego_box, agent_box, text_box

    anim = FuncAnimation(fig, update, frames=len(frames_data), interval=80, blit=True)
    try:
        anim.save(filename, writer=PillowWriter(fps=12))
    except:
        pass
    plt.close(fig)
# ==========================================
# 3. Main Runner
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hyperparameters.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Resolve config path relative to script location
    script_dir = pathlib.Path(__file__).parent
    config_path = script_dir / args.config

    # 1. Load Config
    # with open(args.config, 'r') as f:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. Setup Directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"learned_policies/{cfg['env_id']}_{timestamp}"
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f"{run_dir}/config.yaml", 'w') as f:
        yaml.dump(cfg, f)

    # 3. Load Data
    data_path = "../../data/expert_agent_trajs.npy"
    if not os.path.exists(data_path):
        data_path = "data/expert_agent_trajs.npy"
    
    print(f"Loading expert data from {data_path}...")
    expert_data = np.load(data_path, allow_pickle=True)
    if expert_data.ndim == 0:
        expert_data = expert_data.item()

    if isinstance(expert_data, dict):
        print(f"Detected dictionary data with keys: {list(expert_data.keys())}")
        if 'trajectories' in expert_data:
            expert_data = expert_data['trajectories']
        else:
            raise ValueError("Data dictionary is missing 'trajectories' key!")
            
    print(f"Data loaded successfully. Total trajectories: {len(expert_data)}")
    # 4. Setup Training
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f" Running on {device}. Saving to {run_dir}")

    # Envs
    def make_env(): return IntersectionGymAdapter(expert_data, cfg)
    env = gym.vector.SyncVectorEnv([make_env for _ in range(cfg['num_envs'])])
    eval_env = IntersectionGymAdapter(expert_data, cfg)

    # Model
    model = ContinuousActorCritic(8, 1, cfg['max_accel']).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    lambda_param = torch.tensor(cfg['lambda_init'], device=device)

    # Buffers
    steps = cfg['num_steps']
    envs = cfg['num_envs']
    obs = torch.zeros((steps, envs, 8)).to(device)
    acts = torch.zeros((steps, envs, 1)).to(device)
    logp = torch.zeros((steps, envs)).to(device)
    rews = torch.zeros((steps, envs)).to(device)
    costs = torch.zeros((steps, envs)).to(device)
    dones = torch.zeros((steps, envs)).to(device)
    val_r = torch.zeros((steps+1, envs)).to(device)
    val_c = torch.zeros((steps+1, envs)).to(device)

    next_o, _ = env.reset(seed=cfg['seed'])
    next_o = torch.from_numpy(next_o).float().to(device)
    
    # --- Logging Lists ---
    eval_rewards_history = []
    train_loss_history = [] # Total loss
    train_v_loss_history = [] # Value loss
    
    # Track best reward for model saving
    # best_reward = float('-inf')
    min_loss = float('inf')
    best_epoch = 0
    
    # --- Loop ---
    for itr in tqdm(range(1, cfg['epochs'] + 1)):
        # 1. Rollout
        for t in range(steps):
            obs[t] = next_o
            with torch.no_grad():
                a, lp, _, vr, vc = model.action_value(next_o)
            
            acts[t] = a; logp[t] = lp
            val_r[t] = vr; val_c[t] = vc
            
            next_o, r, term, trunc, info = env.step(a.cpu().numpy())
            rews[t] = torch.from_numpy(r).float().to(device)
            
            if isinstance(info, dict):
                c_batch = info.get('cost', np.zeros(envs))
            else:
                c_batch = np.array([i.get('cost', 0.0) for i in info])
            
            costs[t] = torch.from_numpy(c_batch).float().to(device)
            
            done = np.logical_or(term, trunc)
            dones[t] = torch.tensor(done, device=device, dtype=torch.float32)
            next_o = torch.from_numpy(next_o).float().to(device)

        # 2. GAE
        with torch.no_grad():
            _, _, _, n_vr, n_vc = model.action_value(next_o)
            val_r[steps] = n_vr.squeeze(-1)
            val_c[steps] = n_vc.squeeze(-1)
            
        adv_r, ret_r = compute_gae(rews, val_r, dones, cfg['gamma'], cfg['gae_lambda'])
        adv_c, ret_c = compute_gae(costs, val_c, dones, cfg['gamma'], cfg['gae_lambda'])
        
        # Combine
        adv_total = adv_r - lambda_param * adv_c
        adv_total = (adv_total - adv_total.mean()) / (adv_total.std() + 1e-8)
        
        # 3. Update
        bsz = steps * envs
        states_f = obs.reshape(bsz, -1)
        actions_f = acts.reshape(bsz, -1)
        logp_f = logp.reshape(bsz)
        ret_r_f = ret_r.reshape(bsz)
        ret_c_f = ret_c.reshape(bsz)
        adv_f = adv_total.reshape(bsz)
        
        # Track epoch loss
        epoch_losses = []
        epoch_vr_losses = []

        for _ in range(cfg['update_epochs']):
            perm = torch.randperm(bsz, device=device)
            for start in range(0, bsz, cfg['minibatch_size']):
                idx = perm[start : start + cfg['minibatch_size']]
                
                loss, loss_dict = lagrangian_loss(
                    model, states_f[idx], actions_f[idx], adv_f[idx], logp_f[idx],
                    ret_r_f[idx], ret_c_f[idx], 
                    cfg['clip_ratio'], cfg['ent_coef'], cfg['vf_coef']
                )
                
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
                opt.step()
                
                epoch_losses.append(loss_dict['loss_total'])
                epoch_vr_losses.append(loss_dict['loss_vr'])

        # Save average loss for plotting
        train_loss_history.append(np.mean(epoch_losses))
        train_v_loss_history.append(np.mean(epoch_vr_losses))
        
        # 4. Update Lambda
        ep_cost = costs.sum(dim=0).mean()
        lambda_param += cfg['lambda_lr'] * (ep_cost - cfg['cost_limit'])
        lambda_param = torch.clamp(lambda_param, min=0.0)

        # 5. Log & Eval
        if itr % 10 == 0:
            # Eval (returns 4 values now including loss)
            avg_r, succ, coll, eval_v_loss,_ = evaluate(model, eval_env, device, num_ep=20)
            tqdm.write(f"Iter {itr} | Rw: {avg_r:.1f} | Cost: {ep_cost:.1f} | Lam: {lambda_param.item():.2f} | Loss: {train_loss_history[-1]:.2f}")
            eval_rewards_history.append(avg_r)
            
            # --- Plot 1: Rewards ---
            plt.figure()
            iters_x = list(range(10, len(eval_rewards_history)*10 + 1, 10))
            plt.plot(iters_x, eval_rewards_history)
            plt.xlabel('Iterations'); plt.ylabel('Reward')
            plt.title('Training Progress')
            plt.savefig(f'{run_dir}/eval_reward_curve.png')
            plt.close()
            
            # --- Plot 2: Training Loss ---
            plt.figure()
            plt.plot(train_loss_history, label='Total Loss')
            plt.plot(train_v_loss_history, label='Value Loss', alpha=0.5)
            plt.xlabel('Iterations'); plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.yscale('log')  # Semilog plot on y-axis
            plt.legend()
            plt.savefig(f'{run_dir}/train_loss_curve.png')
            plt.close()
            
            # Track best reward and save model
            # if avg_r > best_reward:
            #     best_reward = avg_r
            #     best_epoch = itr
            #     torch.save(model.state_dict(), f"{run_dir}/best_model.pt")
            current_loss = train_loss_history[-1] 
            
            if current_loss < min_loss:
                min_loss = current_loss
                best_epoch = itr
                torch.save(model.state_dict(), f"{run_dir}/best_model.pt")
                # 可选：打印一下提示，让你知道模型更新了
                # tqdm.write(f"  [New Best Model] Loss improved to {min_loss:.4f}")
            
        if itr % cfg['save_gif_freq'] == 0:
            make_animation(model, eval_env, device, f"{run_dir}/iter_{itr}.gif")
        
        if cfg['checkpoint'] and itr % 100 == 0:
            torch.save(model.state_dict(), f"{run_dir}/model_{itr}.pt")
    

    # if best_epoch > 0:
    #     print(f"Loading best model from epoch {best_epoch} with reward {best_reward:.2f}")
    #     model.load_state_dict(torch.load(f"{run_dir}/best_model.pt"))
    #     torch.save(model.state_dict(), f"{run_dir}/final_model.pt")
    if best_epoch > 0:
        # (Change: Print loss instead of reward)
        print(f"Loading best model from epoch {best_epoch} with loss {min_loss:.4f}")
        model.load_state_dict(torch.load(f"{run_dir}/best_model.pt"))
        torch.save(model.state_dict(), f"{run_dir}/final_model.pt")
    else:
        torch.save(model.state_dict(), f"{run_dir}/final_model.pt")
    
    # 2. Load Test Data
    test_data_path = "../../data/expert_agent_trajectories_noise_test_with_zigzag.npy"
    if not os.path.exists(test_data_path):
        test_data_path = "data/expert_agent_trajectories_noise_test_with_zigzag.npy"

    print(f"\nLoading TEST data from {test_data_path}...")
    try:
        raw_test = np.load(test_data_path, allow_pickle=True)
        if raw_test.ndim == 0: raw_test = raw_test.item()
        
        test_trajs = raw_test['trajectories']
        test_cats = raw_test['categories']
        print(f"Test Data Loaded. Total samples: {len(test_trajs)}")
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return

    # 3. Define Evaluation Categories
    # Split trajectories by category
    cat_subsets = {'All': test_trajs, 'Straight': [], 'Left': [], 'Right': []}
    
    for traj, cat in zip(test_trajs, test_cats):
        # 简单归一化标签字符串
        c = str(cat).lower()
        if 'straight' in c: cat_subsets['Straight'].append(traj)
        elif 'left' in c: cat_subsets['Left'].append(traj)
        elif 'right' in c: cat_subsets['Right'].append(traj)

    print(f"Test Set Breakdown:")
    for k, v in cat_subsets.items():
        if k != 'All':
            print(f"  - {k}: {len(v)} trajectories")

    # 4. Run Evaluation Loop
    results = {}
    
    print("\nStarting Comprehensive Evaluation...")
    report_path = f"{run_dir}/report.txt"
    
    with open(report_path, "w") as f:
        f.write("="*40 + "\n")
        f.write("EXPERIMENT REPORT\n")
        f.write("="*40 + "\n")
        f.write(f"Date: {datetime.datetime.now()}\n\n")
        f.write("Hyperparameters:\n")
        for k, v in cfg.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n" + "="*40 + "\n")
        f.write("TEST RESULTS\n")
        f.write("="*40 + "\n")

    # Loop through [All, Straight, Left, Right]
    for cat_name, subset_trajs in cat_subsets.items():
        if len(subset_trajs) == 0:
            print(f"Skipping {cat_name} (No data)")
            continue
            
        print(f"Evaluating: {cat_name} ({len(subset_trajs)} samples)...")
        
        # Create a temporary environment for this subset
        # We assume the subset is large enough to be random sampled
        temp_env = IntersectionGymAdapter(subset_trajs, cfg)
        
        # Determine number of episodes to run
        # For 'All', we run 100. For specific categories, we run at least 50 or len(subset)
        n_eval = 100 if cat_name == 'All' else min(len(subset_trajs), 100)
        
        # Save GIFs only for the 'All' category to save space, or specific ones if needed
        do_save_gif = (cat_name == 'All')
        subdir = f"{run_dir}" if cat_name == 'All' else None
        
        avg_r, succ, coll, _, avg_time = evaluate(
            model, temp_env, device, 
            num_ep=n_eval, 
            save_gifs=do_save_gif, 
            save_dir=subdir
        )
        
        # Save results
        results[cat_name] = {
            'Reward': avg_r,
            'Success': succ,
            'Collision': coll,
            'Time': avg_time
        }
        
        # Append to report immediately
        with open(report_path, "a") as f:
            f.write(f"\nCategory: {cat_name}\n")
            f.write(f"  - Episodes evaluated: {n_eval}\n")
            f.write(f"  - Avg Reward      : {avg_r:.2f}\n")
            f.write(f"  - Success Rate    : {succ*100:.1f}%\n")
            f.write(f"  - Collision Rate  : {coll*100:.1f}%\n")
            f.write(f"  - Avg Success Time: {avg_time:.2f} s\n")
            f.write("-" * 30 + "\n")

    # 5. Final Summary Print
    print("\n" + "="*30)
    print("FINAL RESULTS SUMMARY")
    print("="*30)
    print(f"{'Category':<10} | {'Success':<8} | {'Collision':<10} | {'Time':<6}")
    print("-" * 40)
    for cat, res in results.items():
        print(f"{cat:<10} | {res['Success']*100:5.1f}%   | {res['Collision']*100:5.1f}%     | {res['Time']:.2f}s")
    print("="*30)
    print(f"Detailed report saved to {report_path}")
    print("Training Finished!")

if __name__ == "__main__":
    main()