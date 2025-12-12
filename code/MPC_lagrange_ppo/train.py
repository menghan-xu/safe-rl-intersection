# train.py (Residual Version)

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
from mpc_solver import MPCPilot 

# ==========================================
# 1. Gym Adapter with Residual MPC
# ==========================================
class ResidualIntersectionAdapter(gym.Env):
    def __init__(self, agent_data_list, config):
        self.cfg = config
        self.env = IntersectionEnv(
            target_pos=config['target_pos'], 
            agent_data_list=agent_data_list, 
            config=config, 
            dt=config['dt']
        ) 
        # State space remains the same [8]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        # Action space: Residual acceleration
        # We allow PPO to correct fully [-max, max] but usually it learns small corrections
        limit = config['max_accel']
        self.action_space = gym.spaces.Box(low=-limit, high=limit, shape=(1,), dtype=np.float32)
        
        # Initialize MPC
        self.mpc = MPCPilot(config)
        self.last_mpc_action = 0.0

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        self.last_mpc_action = 0.0
        return obs.numpy(), {} 

    def step(self, action_residual):
        # 1. Unpack Action
        if isinstance(action_residual, np.ndarray):
            action_residual = action_residual.item()
            
        # 2. Get Info for MPC
        # We need ego state [y, v] and Future Agent Trajectory
        # Access internal env state directly
        ego_y = self.env.state[0].item()
        ego_v = self.env.state[1].item()
        ego_state = [ego_y, ego_v]
        
        # Extract Future Trajectory (Oracle for now, from expert data)
        # Note: In real deployment, this would come from a predictor module
        current_idx = min(self.env.steps, len(self.env.current_agent_traj)-1)
        horizon = self.mpc.N
        future_traj = []
        for k in range(horizon + 1):
            idx = min(current_idx + k, len(self.env.current_agent_traj)-1)
            future_traj.append(self.env.current_agent_traj[idx]) # Convert tensor to numpy for CasADi
            
        # 3. Solve MPC
        # This might be slow! In a real "train.py", doing optimization inside environment step
        # drastically slows down training. But for proof-of-concept, it's fine.
        a_mpc = self.mpc.solve(ego_state, future_traj)
        self.last_mpc_action = a_mpc
        
        # 4. Combine: a_final = a_mpc + a_res * scale
        scale = self.cfg.get('residual_scale', 1.0)
        a_final = a_mpc + action_residual * scale
        
        # Clip to physics limits
        a_final = np.clip(a_final, -self.cfg['max_accel'], self.cfg['max_accel'])
        
        # 5. Step Environment
        obs, reward, done, info = self.env.step(a_final)
        
        # Inject MPC info into 'info' dict for logging/dashboard
        info['a_mpc'] = a_mpc
        info['a_res'] = action_residual
        info['a_final'] = a_final
        
        return obs.numpy(), reward, done, False, info

# ==========================================
# 2. Utils (Animation with Residual Info)
# ==========================================
def make_animation(policy, env, device, filename):
    """
    Generates a GIF with Dashboard showing Residual + MPC actions.
    """
    obs, _ = env.reset()
    obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
    
    ego_pos_hist = []
    agent_pos_hist = []
    metric_hist = [] # Will store a_mpc, a_res
    
    done = False
    while not done:
        render_info = env.env.render()
        ego_pos_hist.append(render_info['ego_pos'])
        agent_pos_hist.append(render_info['agent_pos'])
        
        with torch.no_grad():
            action, _, _, _, _ = policy.action_value(obs_tensor)
        action_res = action.cpu().numpy().item()
        
        # Step (Adapter handles MPC+Res combination)
        next_obs, _, terminated, truncated, info = env.step(action_res)
        metric_hist.append(info)
        
        done = terminated or truncated
        obs_tensor = torch.from_numpy(next_obs).float().to(device).unsqueeze(0)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 8)) # Taller for more info
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 8.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title("Residual MPC-PPO Analysis")
    
    ax.plot([-5, 5], [0, 0], 'k--', alpha=0.3)
    ax.plot([0.5, 0.5], [-5, 15], 'k--', alpha=0.3)

    ego_dot, = ax.plot([], [], 'bo', label='Ego', markersize=8)
    agent_dot, = ax.plot([], [], 'ro', label='Agent', markersize=8)
    
    ROBOT_W, ROBOT_L = 0.43, 0.508
    ego_box = Rectangle((0,0), ROBOT_W, ROBOT_L, color='blue', alpha=0.2)
    agent_box = Rectangle((0,0), ROBOT_W, ROBOT_L, color='red', alpha=0.2)
    ax.add_patch(ego_box); ax.add_patch(agent_box)

    text_box = ax.text(-2.8, 8.2, "", fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(facecolor='white', alpha=0.95, edgecolor='gray'))
    
    def update(frame):
        e_x, e_y = ego_pos_hist[frame]
        a_x, a_y = agent_pos_hist[frame]
        
        ego_dot.set_data([e_x], [e_y])
        agent_dot.set_data([a_x], [a_y])
        ego_box.set_xy((e_x - ROBOT_W/2, e_y - ROBOT_L/2))
        agent_box.set_xy((a_x - ROBOT_W/2, a_y - ROBOT_L/2))
        
        idx = min(frame, len(metric_hist)-1)
        m = metric_hist[idx]
        
        dist_act = m.get('d_act', 0.0)
        a_mpc = m.get('a_mpc', 0.0)
        a_res = m.get('a_res', 0.0)
        a_final = m.get('a_final', 0.0)
        cost = m.get('cost', 0.0)

        info_str = (
            f"Step: {frame}\n"
            f"Dist: {dist_act:.2f} m\n"
            f"=== CONTROL ===\n"
            f"MPC    : {a_mpc:+.2f}\n"
            f"Res(RL): {a_res:+.2f}\n"
            f"Final  : {a_final:+.2f}\n"
            f"=== SAFETY ===\n"
            f"SafeCost: {cost:.1f}"
        )
        text_box.set_text(info_str)
        
        bbox = text_box.get_bbox_patch()
        if bbox:
            bbox.set_edgecolor('red' if dist_act < 0.67 else 'gray')
            bbox.set_linewidth(2 if dist_act < 0.67 else 1)
            
        return ego_dot, agent_dot, ego_box, agent_box, text_box

    anim = FuncAnimation(fig, update, frames=len(ego_pos_hist), interval=80, blit=True)
    anim.save(filename, writer=PillowWriter(fps=12))
    plt.close(fig)


# ==========================================
# 3. Main Runner
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hyperparameters.yaml", help="Path to config file")
    args = parser.parse_args()
    
    script_dir = pathlib.Path(__file__).parent
    config_path = script_dir / args.config

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"learned_policies/{cfg['env_id']}_{timestamp}"
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f"{run_dir}/config.yaml", 'w') as f:
        yaml.dump(cfg, f)

    data_path = "../../data/expert_agent_trajs.npy"
    if not os.path.exists(data_path):
        data_path = "data/expert_agent_trajs.npy"
    
    print(f"Loading expert data from {data_path}...")
    expert_data = np.load(data_path, allow_pickle=True)

    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f" Running on {device}. Saving to {run_dir}")

    # --- USE RESIDUAL ADAPTER HERE ---
    def make_env(): return ResidualIntersectionAdapter(expert_data, cfg)
    
    # Note: VectorEnv might be too slow with MPC inside step!
    # If it's too slow, reduce num_envs to 1 or 2.
    print("Initializing Vector Envs (This might take a moment due to CasADi compilation)...")
    env = gym.vector.SyncVectorEnv([make_env for _ in range(cfg['num_envs'])])
    eval_env = ResidualIntersectionAdapter(expert_data, cfg)

    # ... (Rest of the training loop is IDENTICAL to previous train.py) ...
    # ... Copy paste the model setup, optimizer, loop, GAE, Update, Logging from previous code ...
    # ... I will re-paste the full loop below for completeness ...

    model = ContinuousActorCritic(8, 1, cfg['max_accel']).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    lambda_param = torch.tensor(cfg['lambda_init'], device=device)

    steps = cfg['num_steps']
    envs_num = cfg['num_envs']
    obs = torch.zeros((steps, envs_num, 8)).to(device)
    acts = torch.zeros((steps, envs_num, 1)).to(device)
    logp = torch.zeros((steps, envs_num)).to(device)
    rews = torch.zeros((steps, envs_num)).to(device)
    costs = torch.zeros((steps, envs_num)).to(device)
    dones = torch.zeros((steps, envs_num)).to(device)
    val_r = torch.zeros((steps+1, envs_num)).to(device)
    val_c = torch.zeros((steps+1, envs_num)).to(device)

    next_o, _ = env.reset(seed=cfg['seed'])
    next_o = torch.from_numpy(next_o).float().to(device)
    
    eval_rewards_history = []
    train_loss_history = []
    train_v_loss_history = []
    best_reward = float('-inf')
    best_epoch = 0
    
    for itr in tqdm(range(1, cfg['epochs'] + 1)):
        for t in range(steps):
            obs[t] = next_o
            with torch.no_grad():
                a, lp, _, vr, vc = model.action_value(next_o)
            
            acts[t] = a; logp[t] = lp
            val_r[t] = vr; val_c[t] = vc
            
            # Step happens here -> Calls MPC internally -> Returns combined state
            next_o, r, term, trunc, info = env.step(a.cpu().numpy())
            rews[t] = torch.from_numpy(r).float().to(device)
            
            if isinstance(info, dict):
                c_batch = info.get('cost', np.zeros(envs_num))
            else:
                c_batch = np.array([i.get('cost', 0.0) for i in info])
            
            costs[t] = torch.from_numpy(c_batch).float().to(device)
            
            done = np.logical_or(term, trunc)
            dones[t] = torch.tensor(done, device=device, dtype=torch.float32)
            next_o = torch.from_numpy(next_o).float().to(device)

        with torch.no_grad():
            _, _, _, n_vr, n_vc = model.action_value(next_o)
            val_r[steps] = n_vr.squeeze(-1)
            val_c[steps] = n_vc.squeeze(-1)
            
        adv_r, ret_r = compute_gae(rews, val_r, dones, cfg['gamma'], cfg['gae_lambda'])
        adv_c, ret_c = compute_gae(costs, val_c, dones, cfg['gamma'], cfg['gae_lambda'])
        
        adv_total = adv_r - lambda_param * adv_c
        adv_total = (adv_total - adv_total.mean()) / (adv_total.std() + 1e-8)
        
        bsz = steps * envs_num
        states_f = obs.reshape(bsz, -1)
        actions_f = acts.reshape(bsz, -1)
        logp_f = logp.reshape(bsz)
        ret_r_f = ret_r.reshape(bsz)
        ret_c_f = ret_c.reshape(bsz)
        adv_f = adv_total.reshape(bsz)
        
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

        train_loss_history.append(np.mean(epoch_losses))
        train_v_loss_history.append(np.mean(epoch_vr_losses))
        
        ep_cost = costs.sum(dim=0).mean()
        lambda_param += cfg['lambda_lr'] * (ep_cost - cfg['cost_limit'])
        lambda_param = torch.clamp(lambda_param, min=0.0)

        if itr % 10 == 0:
            avg_r, succ, coll, eval_v_loss, _ = evaluate(model, eval_env, device, num_ep=20)
            tqdm.write(f"Iter {itr} | Rw: {avg_r:.1f} | Cost: {ep_cost:.1f} | Lam: {lambda_param.item():.2f} | Succ: {succ:.2f} | Coll: {coll:.2f}")
            eval_rewards_history.append(avg_r)
            
            plt.figure()
            iters_x = list(range(10, len(eval_rewards_history)*10 + 1, 10))
            plt.plot(iters_x, eval_rewards_history)
            plt.title('Training Progress (Residual MPC)'); plt.savefig(f'{run_dir}/eval_reward_curve.png'); plt.close()
            
            if avg_r > best_reward:
                best_reward = avg_r
                best_epoch = itr
                torch.save(model.state_dict(), f"{run_dir}/best_model.pt")
            
        if itr % cfg['save_gif_freq'] == 0:
            make_animation(model, eval_env, device, f"{run_dir}/iter_{itr}.gif")
        
        if cfg['checkpoint'] and itr % 100 == 0:
            torch.save(model.state_dict(), f"{run_dir}/model_{itr}.pt")
    
    if best_epoch > 0:
        model.load_state_dict(torch.load(f"{run_dir}/best_model.pt"))
        torch.save(model.state_dict(), f"{run_dir}/final_model.pt")
    else:
        torch.save(model.state_dict(), f"{run_dir}/final_model.pt")
    
    print("Generating Final Report and Test GIFs...")
    final_r, final_succ, final_coll, final_v_loss, final_time = evaluate(model, eval_env, device, num_ep=100, save_gifs=True, save_dir=run_dir)
    
    report_path = f"{run_dir}/report.txt"
    with open(report_path, "w") as f:
        f.write("EXPERIMENT REPORT (Residual MPC-PPO)\n")
        f.write(f"Success Rate    : {final_succ*100:.1f}%\n")
        f.write(f"Collision Rate  : {final_coll*100:.1f}%\n")
    print(f"Report saved to {report_path}")

# Helper functions definition needed to run without errors:
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

def lagrangian_loss(agent, states, actions, adv, logp, ret_r, ret_c, clip, ent_c, vf_c):
    _, new_logp, entropy, val_r, val_c = agent.action_value(states, actions)
    ratio = torch.exp(new_logp - logp)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1-clip, 1+clip) * adv
    loss_pi = -torch.min(surr1, surr2).mean()
    loss_vr = (val_r - ret_r).pow(2).mean()
    loss_vc = (val_c - ret_c).pow(2).mean()
    loss_ent = -ent_c * entropy.mean()
    total_loss = loss_pi + vf_c * (loss_vr + loss_vc) + loss_ent
    return total_loss, {"loss_total": total_loss.item(), "loss_vr": loss_vr.item()}

def evaluate(model, env, device, num_ep=100, save_gifs=False, save_dir=None):
    model.eval()
    success_count = 0; collision_count = 0; total_r = []
    frames_data = [] # For GIF
    success_times = []
    
    for ep_i in range(num_ep):
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        done = False
        ep_r = 0
        min_dist = float('inf')
        ep_steps = 0
        
        # Only save frames for the first few failure cases to save time
        should_record = save_gifs and (ep_i < 20) 
        if should_record: frames_data = []
        
        while not done:
            if should_record:
                r_info = env.env.render() # Access internal env
                frames_data.append({**r_info}) # Copy dict

            with torch.no_grad():
                x = model.trunk(obs)
                action = torch.tanh(model.mu_head(x)) * model.max_action
                action_scalar = action.cpu().numpy().item()
            
            next_obs, r, term, trunc, info = env.step(action_scalar)
            
            if should_record:
                frames_data[-1].update({'info': info})

            ep_r += r
            ep_steps += 1
            
            # Check dist for metrics
            e_pos = np.array(env.env.render()['ego_pos'])
            a_pos = np.array(env.env.render()['agent_pos'])
            dist = np.linalg.norm(e_pos - a_pos)
            if dist < min_dist: min_dist = dist
            
            done = term or trunc
            obs = torch.from_numpy(next_obs).float().to(device).unsqueeze(0)
            
            if done:
                is_coll = min_dist < env.env.collision_dist
                is_succ = env.env.state[0] >= env.env.target_pos[0]
                
                if is_succ: 
                    success_count += 1
                    success_times.append(ep_steps * 0.1)
                if is_coll: collision_count += 1
                
                # Save GIF if failed
                if should_record and not is_succ and save_dir:
                    fname = f"{save_dir}/test_ep_{ep_i}_fail.gif"
                    save_cached_animation(frames_data, fname)

        total_r.append(ep_r)
    
    model.train()
    avg_time = np.mean(success_times) if len(success_times) > 0 else 0.0
    return np.mean(total_r), success_count/num_ep, collision_count/num_ep, 0.0, avg_time

def save_cached_animation(frames, filename):
    # Simplified saver for eval
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-3,3); ax.set_ylim(-3,10)
    ego_dot, = ax.plot([],[],'bo'); agent_dot, = ax.plot([],[],'ro')
    txt = ax.text(-2,9,"")
    def update(i):
        d = frames[i]
        ego_dot.set_data(d['ego_pos']); agent_dot.set_data(d['agent_pos'])
        a_mpc = d.get('info', {}).get('a_mpc', 0)
        a_res = d.get('info', {}).get('a_res', 0)
        txt.set_text(f"MPC:{a_mpc:.2f}\nRes:{a_res:.2f}")
    anim = FuncAnimation(fig, update, frames=len(frames), interval=80)
    anim.save(filename, writer='pillow', fps=12)
    plt.close(fig)

if __name__ == "__main__":
    main()