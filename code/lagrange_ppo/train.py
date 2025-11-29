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
    """Generates a GIF with Bounding Boxes and Dashboard."""
    obs, _ = env.reset()
    obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
    
    ego_pos_hist, agent_pos_hist = [], []
    ego_v_hist, agent_v_hist = [], []
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
        
        next_obs, _, terminated, truncated, info = env.step(action_scalar)
        metric_hist.append(info)
        done = terminated or truncated
        obs_tensor = torch.from_numpy(next_obs).float().to(device).unsqueeze(0)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 8.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Intersection Lag-PPO")
    
    # Lanes
    ax.plot([-5, 5], [0, 0], 'k--', alpha=0.3)
    ax.plot([0.5, 0.5], [-5, 15], 'k--', alpha=0.3)

    # Objects
    ego_dot, = ax.plot([], [], 'bo', label='Ego', markersize=8)
    agent_dot, = ax.plot([], [], 'ro', label='Agent', markersize=8)
    
    ROBOT_W, ROBOT_L = 0.43, 0.508
    ego_box = Rectangle((0,0), ROBOT_W, ROBOT_L, angle=0.0, color='blue', alpha=0.3)
    agent_box = Rectangle((0,0), ROBOT_W, ROBOT_L, angle=0.0, color='red', alpha=0.3)
    ax.add_patch(ego_box); ax.add_patch(agent_box)

    text_box = ax.text(-2.3, 8.0, "", fontsize=10, verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.legend(loc='lower right')
    
    def update(frame):
        e_x, e_y = ego_pos_hist[frame]
        a_x, a_y = agent_pos_hist[frame]
        
        ego_dot.set_data([e_x], [e_y])
        agent_dot.set_data([a_x], [a_y])
        ego_box.set_xy((e_x - ROBOT_W/2, e_y - ROBOT_L/2))
        agent_box.set_xy((a_x - ROBOT_W/2, a_y - ROBOT_L/2))
        
        dist = np.linalg.norm(np.array([e_x, e_y]) - np.array([a_x, a_y]))
        
        # Metrics
        idx = min(frame, len(metric_hist)-1)
        m = metric_hist[idx]
        
        info = (
            f"Step: {frame}\nDist: {dist:.3f} m\n"
            f"SoftCost: {m.get('cost', 0):.1f}\n"
            f"TotalRew: {m.get('total_reward', 0):.1f}\n"
            f"----------------\n"
            f"Ego V : {ego_v_hist[frame]:.2f}\n"
            f"Agent V: {agent_v_hist[frame]:.2f}"
        )
        text_box.set_text(info)
        
        bbox = text_box.get_bbox_patch()
        if bbox:
            bbox.set_edgecolor('red' if dist < 0.67 else 'gray')
            bbox.set_linewidth(2 if dist < 0.67 else 1)
            
        return ego_dot, agent_dot, ego_box, agent_box, text_box

    anim = FuncAnimation(fig, update, frames=len(ego_pos_hist), interval=80, blit=True)
    anim.save(filename, writer=PillowWriter(fps=15))
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

# Evaluate with Value Loss calculation and GIF saving support
def evaluate(model, env, device, num_ep=100, save_gifs=False, save_dir=None):
    model.eval()
    success = 0
    collision = 0
    total_r = []
    total_value_loss = [] # Track critic accuracy
    
    for ep_i in range(num_ep):
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        done = False
        ep_r = 0
        min_dist = float('inf')
        
        # For Value Loss Calculation
        ep_values = []
        ep_rewards = []
        
        # GIF saving logic for Test Phase
        if save_gifs and save_dir and ep_i < 5: # Save first 5 episodes as sample
             gif_path = f"{save_dir}/test_ep_{ep_i}.gif"
             # We use make_animation which runs its own episode, so we call it separately
             # Note: This is slightly inefficient (runs extra episodes) but cleaner code-wise
             make_animation(model, env, device, gif_path)

        while not done:
            with torch.no_grad():
                x = model.trunk(obs)
                action = torch.tanh(model.mu_head(x)) * model.max_action
                
                # Get Value estimate for loss calc
                val_r = model.reward_value_head(x)
                ep_values.append(val_r.item())
                
                action_scalar = action.cpu().numpy().item()
            
            next_obs, r, term, trunc, _ = env.step(action_scalar)
            ep_r += r
            ep_rewards.append(r)
            
            render_info = env.env.render()
            e_pos = np.array(render_info['ego_pos'])
            a_pos = np.array(render_info['agent_pos'])
            dist = np.linalg.norm(e_pos - a_pos)
            if dist < min_dist: min_dist = dist
            
            done = term or trunc
            obs = torch.from_numpy(next_obs).float().to(device).unsqueeze(0)
            
            if done:
                if env.env.state[0] >= env.env.target_pos[0]: 
                    success += 1
                if min_dist < env.env.collision_dist: 
                    collision += 1

        # Compute Critic Loss (MSE between predicted Value and actual discounted Return)
        # Simple Monte Carlo return calculation
        returns = []
        G = 0
        for r in reversed(ep_rewards):
            G = r + 0.99 * G # assuming gamma=0.99 for eval
            returns.insert(0, G)
        
        # Calculate MSE
        v_loss = np.mean([(v - ret)**2 for v, ret in zip(ep_values, returns)])
        total_value_loss.append(v_loss)
        total_r.append(ep_r)
        
    model.train()
    return np.mean(total_r), success/num_ep, collision/num_ep, np.mean(total_value_loss)

# ==========================================
# 3. Main Runner
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hyperparameters.yaml", help="Path to config file")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
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
            avg_r, succ, coll, eval_v_loss = evaluate(model, eval_env, device, num_ep=20)
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
            plt.legend()
            plt.savefig(f'{run_dir}/train_loss_curve.png')
            plt.close()
            
        if itr % cfg['save_gif_freq'] == 0:
            make_animation(model, eval_env, device, f"{run_dir}/iter_{itr}.gif")
        
        if cfg['checkpoint'] and itr % 50 == 0:
            torch.save(model.state_dict(), f"{run_dir}/model_{itr}.pt")
    
    torch.save(model.state_dict(), f"{run_dir}/final_model.pt")
    
    # --- FINAL REPORT (TEST PHASE) ---
    print("Generating Final Report and Test GIFs...")
    # Pass run_dir to save test GIFs
    final_r, final_succ, final_coll, final_v_loss = evaluate(model, eval_env, device, num_ep=100, save_gifs=True, save_dir=run_dir)
    
    report_path = f"{run_dir}/report.txt"
    with open(report_path, "w") as f:
        f.write("="*40 + "\n")
        f.write("EXPERIMENT REPORT\n")
        f.write("="*40 + "\n")
        f.write(f"Date: {datetime.datetime.now()}\n\n")
        
        f.write("Hyperparameters:\n")
        for k, v in cfg.items():
            f.write(f"  {k}: {v}\n")
            
        f.write("\n" + "-"*40 + "\n")
        f.write("Final Results (100 Episodes):\n")
        f.write(f"  Avg Reward      : {final_r:.2f}\n")
        f.write(f"  Success Rate    : {final_succ*100:.1f}%\n")
        f.write(f"  Collision Rate  : {final_coll*100:.1f}%\n")
        f.write(f"  Eval Value Loss : {final_v_loss:.4f}\n")
        f.write(f"  Final Lambda    : {lambda_param.item():.4f}\n")
        f.write("="*40 + "\n")
        
    print(f"Report saved to {report_path}")
    print("Training Finished!")

if __name__ == "__main__":
    main()