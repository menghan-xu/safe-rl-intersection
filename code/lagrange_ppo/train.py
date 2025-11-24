import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from argparse import ArgumentParser
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pathlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import datetime # Added for timestamp

# Imports
from env import IntersectionEnv  
from models import ContinuousActorCritic       

# ==========================================
# 1. Gym Adapter
# ==========================================
class IntersectionGymAdapter(gym.Env):
    def __init__(self, agent_data_list):
        self.env = IntersectionEnv(target_pos=[1.25, 0.0], agent_data_list=agent_data_list, dt=0.1) 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return obs.numpy(), {} 

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        # returns: obs, reward, done, info{'cost': ...}
        obs, reward, done, info = self.env.step(action)
        return obs.numpy(), reward, done, False, info

# ==========================================
# 2. Lag-PPO Utils
# ==========================================

def make_animation(policy, env, device, filename="trajectory.gif"):
    """
    Runs one episode and saves a GIF animation with Velocity/Accel dashboard.
    """
    print(f"Generating animation: {filename}...")
    
    # 1. Collect Data
    obs, _ = env.reset()
    obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
    
    # History containers
    ego_pos_hist = []
    agent_pos_hist = []
    
    ego_v_hist = []
    ego_a_hist = [] # Action is acceleration
    agent_v_hist = []
    
    done = False
    while not done:
        # Get Action
        with torch.no_grad():
            action, _, _, _, _ = policy.action_value(obs_tensor)
        action_scalar = action.cpu().numpy().item()
        
        # Step Environment
        next_obs, _, terminated, truncated, _ = env.step(action_scalar)
        
        # Get Render Info (After step to get updated state)
        render_info = env.env.render() 
        
        # Record Data
        ego_pos_hist.append(render_info['ego_pos'])
        agent_pos_hist.append(render_info['agent_pos'])
        
        ego_v_hist.append(render_info['ego_v'])
        ego_a_hist.append(action_scalar) # Current action is accel
        agent_v_hist.append(render_info['agent_v'])
        
        done = terminated or truncated
        obs_tensor = torch.from_numpy(next_obs).float().to(device).unsqueeze(0)

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(6, 6)) 
    
    # --- [UPDATED] Plot Limits ---
    # Zoom in to the 3x3m intersection area
    ax.set_xlim(-2, 2) 
    ax.set_ylim(-2, 2) 
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f"Intersection Lag-PPO\n(Episode Length: {len(ego_pos_hist)} steps)")

    # Draw Lanes (Static)
    # Drawn slightly larger than view to ensure they cover the edges
    ax.plot([-3, 3], [0, 0], 'k--', alpha=0.3, lw=1) # Horizontal lane center
    ax.plot([0.5, 0.5], [-3, 3], 'k--', alpha=0.3, lw=1) # Vertical lane center (Ego path)

    # Initialize Objects
    ego_dot, = ax.plot([], [], 'bo', label='Ego', markersize=12, zorder=5)
    agent_dot, = ax.plot([], [], 'ro', label='Agent', markersize=12, zorder=5)
    ego_line, = ax.plot([], [], 'b-', alpha=0.3, lw=2)
    agent_line, = ax.plot([], [], 'r-', alpha=0.3, lw=2)
    
    # Dashboard Text (Positioned relative to the new limits)
    # Top-left corner of the view is roughly (-1.8, 1.8)
    status_text = ax.text(
        -1.9, 1.9, "", 
        fontsize=9, 
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    )
    
    ax.legend(loc='lower right')

    def update(frame):
        # 1. Update Positions
        e_pos = ego_pos_hist[frame]
        a_pos = agent_pos_hist[frame]
        
        ego_dot.set_data([e_pos[0]], [e_pos[1]])
        agent_dot.set_data([a_pos[0]], [a_pos[1]])
        
        # 2. Update Trails
        e_path = np.array(ego_pos_hist[:frame+1])
        a_path = np.array(agent_pos_hist[:frame+1])
        ego_line.set_data(e_path[:, 0], e_path[:, 1])
        agent_line.set_data(a_path[:, 0], a_path[:, 1])
        
        # 3. Update Dashboard Text
        e_v = ego_v_hist[frame]
        e_a = ego_a_hist[frame]
        a_v = agent_v_hist[frame]
        
        info_str = (
            f"Step: {frame}\n"
            f"----------------\n"
            f"EGO (Blue)\n"
            f"Vel : {e_v:.2f} m/s\n"
            f"Acc : {e_a:.2f} m/s²\n"
            f"----------------\n"
            f"AGENT (Red)\n"
            f"Vel : {a_v:.2f} m/s"
        )
        status_text.set_text(info_str)
        
        return ego_dot, agent_dot, ego_line, agent_line, status_text

    # Create Animation
    anim = FuncAnimation(fig, update, frames=len(ego_pos_hist), interval=100, blit=True)
    
    # Save
    try:
        writer = PillowWriter(fps=10) # Slower FPS to read text easily
        anim.save(filename, writer=writer)
        print(f"Animation saved: {filename}")
    except Exception as e:
        print(f"Animation save failed: {e}")
    
    plt.close(fig)

def compute_gae_returns(rewards, values, dones, gamma, gae_lambda):
    """
    Computes GAE for either Reward or Cost.
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns
    

def lagrangian_ppo_loss(agent, states, actions, adv_total, logprobs, ret_r, ret_c, clip_ratio=0.2, ent_coef=0.01, vf_coef=0.5):
    """
    Loss function for Lag-PPO.
    Optimizes Policy using Combined Advantage (Reward - lambda * Cost).
    Optimizes TWO Critics (Reward Value and Cost Value).
    """
    _, new_logprob, entropy, val_r, val_c = agent.action_value(states, actions)

    # 1. Policy Loss (using Combined Advantage)
    ratio = torch.exp(new_logprob - logprobs)
    arg1 = ratio * adv_total
    arg2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_total
    L_Policy = -(torch.min(arg1, arg2)).mean()
    
    # 2. Value Loss 1: Task Reward (MSE)
    L_VR = (val_r - ret_r).pow(2).mean()

    # [cite_start]3. Value Loss 2: Safety Cost (MSE) [cite: 59]
    L_VC = (val_c - ret_c).pow(2).mean()
    
    # 4. Entropy Loss
    L_Entropy = -torch.mean(entropy)

    # Total Loss
    loss = L_Policy + vf_coef * (L_VR + L_VC) + ent_coef * L_Entropy
    return loss

# ==========================================
# 3. Main Loop
# ==========================================

def train(args, expert_data, log_dir): # <--- ADDED log_dir argument
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on {device}")

    # Env Setup
    def make_env_fn():
        return IntersectionGymAdapter(agent_data_list=expert_data)
    env = gym.vector.SyncVectorEnv([make_env_fn for _ in range(args.num_envs)])
    eval_env = IntersectionGymAdapter(agent_data_list=expert_data)
    
    # Model
    policy = ContinuousActorCritic(state_dim=8, action_dim=1, max_action=5.0).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    
    # --- Lagrange Multiplier Setup ---
    # Lambda is initialized and updated manually via Dual Gradient Ascent 
    lagrange_lambda = torch.tensor(args.lambda_init, device=device)
    
    # Buffers
    states = torch.zeros((args.num_steps, args.num_envs, 8)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, 1)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Two separate buffers for Reward and Cost
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    values_r = torch.zeros((args.num_steps + 1, args.num_envs)).to(device)
    values_c = torch.zeros((args.num_steps + 1, args.num_envs)).to(device)

    obs, _ = env.reset(seed=args.seed)
    obs = torch.from_numpy(obs).float().to(device)

    # Path creation moved to main block (no longer here)
    eval_rewards = []
    
    print("Start Training...")
    for iteration in tqdm(range(1, args.epochs + 1)):
        
        # --- 1. Rollout ---
        for t in range(args.num_steps):
            states[t] = obs
            with torch.no_grad():
                # Get both Value estimates
                action, logprob, _, val_r, val_c = policy.action_value(obs)
            
            actions[t] = action
            logprobs[t] = logprob
            values_r[t] = val_r
            values_c[t] = val_c

            next_obs, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
            
            # Store Reward and Cost separately
            rewards[t] = torch.from_numpy(reward).float().to(device)
            
            # Extract cost from info
            # SyncVectorEnv returns infos as a structure that can be indexed directly if using gymnasium > 0.29
            # Fallback logic for safety:
            if isinstance(infos, dict):
                if 'cost' in infos:
                    cost_batch = infos['cost']
                else:
                    cost_batch = np.zeros(args.num_envs)
            else:
                cost_batch = np.array([i.get('cost', 0.0) for i in infos])
                
            costs[t] = torch.from_numpy(cost_batch).float().to(device)

            done = np.logical_or(terminated, truncated)
            dones[t] = torch.tensor(done, device=device, dtype=torch.float32)
            obs = torch.from_numpy(next_obs).float().to(device)

        # --- 2. GAE Calculation ---
        with torch.no_grad():
            next_val_r, next_val_c = policy.value(obs)
            values_r[args.num_steps] = next_val_r
            values_c[args.num_steps] = next_val_c

        # Compute Advantages for Reward (A_R)
        adv_r, ret_r = compute_gae_returns(rewards, values_r, dones, args.gamma, args.gae_lambda)
        
        # Compute Advantages for Cost (A_C)
        adv_c, ret_c = compute_gae_returns(costs, values_c, dones, args.gamma, args.gae_lambda)

        # --- 3. Lagrangian Advantage Combination  ---
        # A_Lag = A_R - lambda * A_C
        adv_total = adv_r - lagrange_lambda * adv_c
        
        # Normalize combined advantage
        adv_flat = adv_total.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            
        # Flatten buffers for PPO
        bsz = args.num_steps * args.num_envs
        states_f = states.reshape(bsz, -1)
        actions_f = actions.reshape(bsz, -1)
        logp_f = logprobs.reshape(bsz)
        ret_r_f = ret_r.reshape(bsz)
        ret_c_f = ret_c.reshape(bsz) # Cost Returns needed for V_C training

        # --- 4. PPO Update ---
        for _ in range(args.update_epochs):
            perm = torch.randperm(bsz, device=device)
            for start in range(0, bsz, args.minibatch_size):
                idx = perm[start : start + args.minibatch_size]

                loss = lagrangian_ppo_loss(
                    policy, 
                    states_f[idx], actions_f[idx], 
                    adv_flat[idx], logp_f[idx], 
                    ret_r_f[idx], ret_c_f[idx], # Pass both returns
                    args.clip_ratio, args.ent_coef, args.vf_coef
                )
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
        
        # --- 5. Update Lambda (Dual Ascent)  ---
        # lambda <- max(0, lambda + alpha * (J_c - d))
        # We use the mean cost of the current batch as an approximation of J_c
        mean_cost = costs.mean().item()
        # Estimate episode cost roughly as step_cost / (1-gamma) or just sum if gamma=1
        # Here we use a simple metric: Average Step Cost vs Limit
        # If args.cost_limit is defined as "limit per episode", we should sum.
        # Assuming args.cost_limit is "Average Cost per Step Limit" for stability:
        # If your limit is 20.0 for an episode of 200 steps, step limit = 0.1
        
        # Let's assume cost_limit is the total budget per episode (e.g. 20)
        # We scale the batch mean cost to an episode scale approx
        est_episode_cost = costs.sum(dim=0).mean() # Sum over T, Mean over N
        
        lambda_loss = args.lambda_lr * (est_episode_cost - args.cost_limit)
        lagrange_lambda += lambda_loss
        lagrange_lambda = torch.clamp(lagrange_lambda, min=0.0)

        if iteration % 10 == 0:
            avg_rew = val(policy, eval_env, device)
            print(f"Iter {iteration} | Reward: {avg_rew:.2f} | Lambda: {lagrange_lambda.item():.4f} | Cost: {est_episode_cost:.2f}")
            eval_rewards.append(avg_rew)
            
            # Plot using log_dir
            plt.figure()
            iters = list(range(10, len(eval_rewards)*10 + 1, 10))
            plt.plot(iters, eval_rewards)
            plt.savefig(f'{log_dir}/eval_reward_curve.png')
            plt.close()
            
        # Save Animation using log_dir
        if iteration % 50 == 0:
                gif_path = f"{log_dir}/iter_{iteration}.gif"
                # Note: pass eval_env, not the vector env
                make_animation(policy, eval_env, device, filename=gif_path)
                
        # Save Checkpoint using log_dir
        if args.checkpoint and iteration % 50 == 0:
            torch.save(policy.state_dict(), f"{log_dir}/model_{iteration}.pt")
    
    # Save Final Model using log_dir
    torch.save(policy.state_dict(), f"{log_dir}/final_model.pt")
    print("Training Finished!")
    return policy


def val(model, env, device, num_ep=5):
    rew_sum = 0
    model.eval()
    for i in range(num_ep):
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        done = False
        while not done:
            with torch.no_grad():
                # Note: Returns 5 values now
                action, _, _, _, _ = model.action_value(obs)
            action_scalar = action.cpu().numpy().item()
            next_obs, reward, terminated, truncated, _ = env.step(action_scalar)
            rew_sum += reward
            done = terminated or truncated
            obs = torch.from_numpy(next_obs).float().to(device).unsqueeze(0)
    model.train()
    return rew_sum / num_ep

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Intersection-Lag-v0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", action="store_true", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    
    # Lag-PPO specific args
    parser.add_argument("--cost_limit", type=float, default=20.0, help="Safety budget d")
    parser.add_argument("--lambda_lr", type=float, default=0.05, help="Learning rate for lambda")
    parser.add_argument("--lambda_init", type=float, default=1.0, help="Initial lambda")
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)

    args = parser.parse_args()  
    
    # --- Create Timestamped Directory ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_id}_{timestamp}"
    log_dir = f"learned_policies/{run_name}/"
    
    # Create the directory
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(f" Saving results to: {log_dir}")
    
    data_path = "../../data/expert_agent_trajs.npy"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found! Please run process_data.py first.")
        exit()
        
    print(f"Loading expert data from {data_path}...")
    expert_data = np.load(data_path, allow_pickle=True)
    
    # Pass log_dir to train function
    train(args, expert_data, log_dir)