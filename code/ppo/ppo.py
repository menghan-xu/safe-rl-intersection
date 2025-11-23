import os
# Fix for some OpenMP errors on Mac/Windows
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

# --- Import your custom modules ---
# Ensure intersection_env.py and model.py are in the same directory
from env import IntersectionEnv  
from models import ContinuousActorCritic       

# ==========================================
# 1. Gym Adapter (Wrapper)
# ==========================================
class IntersectionGymAdapter(gym.Env):
    """
    Adapter class to wrap the custom IntersectionEnv into a standard Gymnasium API.
    This allows the PPO training loop to interact with your physics-based environment seamlessly.
    """
    def __init__(self, agent_data_list):
        # Initialize the underlying physics environment
        # target_pos is set to y=10 (cross intersection) and v=5 (target speed)
        self.env = IntersectionEnv(target_pos=[1.5, 0.0], agent_data_list=agent_data_list, dt=0.1)
        
        # Define Observation Space: 8 dimensions
        # [ego_y, ego_v, agent_x, agent_y, agent_vx, agent_vy, agent_sx, agent_sy]
        # We use -inf to inf as bounds since we primarily care about the shape here
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        # Define Action Space: 1 dimension (Acceleration)
        # Physical limits are roughly -5 to 5 m/s^2 (matching the max_accel in model.py)
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Gymnasium requires reset to return (observation, info)
        obs = self.env.reset()
        return obs.numpy(), {} 

    def step(self, action):
        # Convert input action (usually numpy array) to scalar for the internal env
        if isinstance(action, np.ndarray):
            action = action.item()
            
        obs, reward, done, _ = self.env.step(action)
        
        # Gymnasium requires: obs, reward, terminated, truncated, info
        # We map 'done' to 'terminated' and set 'truncated' to False for simplicity
        return obs.numpy(), reward, done, False, {}

# ==========================================
# 2. PPO Utility Functions
# ==========================================

def compute_gae_returns(rewards, values, dones, gamma, gae_lambda):
    """
    Computes Generalized Advantage Estimation (GAE) and discounted returns.
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        # Calculate TD error: delta = r + gamma * V(s') - V(s)
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        # Recursive GAE calculation
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    # Returns = Advantage + Value
    returns = advantages + values[:-1]
    return advantages, returns
    

def ppo_loss(agent, states, actions, advantages, logprobs, returns, clip_ratio=0.2, ent_coef=0.01, vf_coef=0.5):
    """
    Calculates the PPO composite loss function:
    Loss = Policy Loss (Clipped) + Value Loss - Entropy Bonus
    """
    _, new_logprob, entropy, value_preds = agent.action_value(states, actions)

    # Calculate probability ratio (pi_new / pi_old)
    ratio = torch.exp(new_logprob - logprobs)
    
    # Surrogate objectives
    arg1 = ratio * advantages
    arg2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    
    # Policy Loss (Maximize objective -> Minimize negative objective)
    LCLIP = -(torch.min(arg1, arg2)).mean()
    
    # Value Function Loss (MSE)
    LVF = (value_preds - returns).pow(2).mean()
    
    # Entropy Loss (Maximize entropy -> Minimize negative entropy)
    LS = torch.mean(entropy)

    # Total Loss
    loss = LCLIP + vf_coef * LVF - ent_coef * LS
    return loss

# ==========================================
# 3. Main Training Loop
# ==========================================

def train(args, expert_data):
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Device Selection ---
    # Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration!")
    else:
        device = torch.device("cpu")
        print("Using CPU. Training might be slow.")

    # --- Environment Setup ---
    # Factory function to create env instances
    def make_env_fn():
        return IntersectionGymAdapter(agent_data_list=expert_data)

    # Vectorized environment for parallel data collection
    env = gym.vector.SyncVectorEnv([make_env_fn for _ in range(args.num_envs)])
    
    # Single environment for evaluation
    eval_env = IntersectionGymAdapter(agent_data_list=expert_data)
    
    # --- Model & Optimizer ---
    # state_dim=8, action_dim=1, max_action=5.0 (based on data analysis)
    policy = ContinuousActorCritic(state_dim=8, action_dim=1, max_action=5.0).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    
    # Pre-allocate storage buffers on device
    states = torch.zeros((args.num_steps, args.num_envs, 8)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, 1)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # values buffer needs +1 size for bootstrapping the last step
    values = torch.zeros((args.num_steps + 1, args.num_envs)).to(device)

    # Initial observation
    obs, _ = env.reset(seed=args.seed)
    obs = torch.from_numpy(obs).float().to(device)

    # Directory for saving models
    pathlib.Path(f"learned_policies/{args.env_id}/").mkdir(parents=True, exist_ok=True)
    eval_rewards = []
    
    print("Start Training...")
    
    # --- Main Loop ---
    for iteration in tqdm(range(1, args.epochs + 1)):
        
        # 1. Data Collection (Rollout Phase)
        for t in range(args.num_steps):
            states[t] = obs
            
            # Get action from policy (no_grad for efficiency during rollout)
            with torch.no_grad():
                action, logprob, _, value = policy.action_value(obs)
            
            actions[t] = action
            logprobs[t] = logprob
            values[t] = value

            # Step the environment
            next_obs, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
            
            # Handle done flags
            done = np.logical_or(terminated, truncated)
            dones[t] = torch.tensor(done, device=device, dtype=torch.float32)
            rewards[t] = torch.from_numpy(reward).float().to(device)

            obs = torch.from_numpy(next_obs).float().to(device)

        # 2. Advantage Estimation (GAE)
        with torch.no_grad():
            # Calculate value of the *next* state (for the last step bootstrapping)
            values[args.num_steps] = policy.value(obs)

        advantages, returns = compute_gae_returns(rewards, values, dones, args.gamma, args.gae_lambda)

        # Normalize advantages (Standard practice to stabilize training)
        adv_flat = advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            
        # 3. PPO Update Phase
        # Flatten batch dimensions
        bsz = args.num_steps * args.num_envs
        states_flat   = states.reshape(bsz, -1)
        actions_flat  = actions.reshape(bsz, -1)
        logp_flat     = logprobs.reshape(bsz)
        returns_flat  = returns.reshape(bsz)

        # Multi-epoch update on collected batch
        for _ in range(args.update_epochs):
            perm = torch.randperm(bsz, device=device)
            
            # Mini-batch updates
            for start in range(0, bsz, args.minibatch_size):
                idx = perm[start : start + args.minibatch_size]

                loss = ppo_loss(
                    policy, 
                    states_flat[idx], 
                    actions_flat[idx], 
                    adv_flat[idx], 
                    logp_flat[idx], 
                    returns_flat[idx],
                    args.clip_ratio, args.ent_coef, args.vf_coef
                )
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
            
        # 4. Evaluation & Checkpointing
        if iteration % 10 == 0:
            avg_rew = val(policy, eval_env, device)
            tqdm.write(f"Iter {iteration} | Eval Reward: {avg_rew:.2f}")
            eval_rewards.append(avg_rew)
            
            # Plot learning curve
            plt.figure()
            iters = list(range(10, len(eval_rewards)*10 + 1, 10))
            plt.plot(iters, eval_rewards)
            plt.xlabel('Iteration')
            plt.ylabel('Average Eval Reward')
            plt.title(f'Training Curve')
            plt.savefig(f'learned_policies/{args.env_id}/eval_reward_curve.png')
            plt.close()

        # Periodic Model Save
        if args.checkpoint and iteration % 50 == 0:
            torch.save(policy.state_dict(), f"learned_policies/{args.env_id}/model_{iteration}.pt")
    
    # Save Final Model
    torch.save(policy.state_dict(), f"learned_policies/{args.env_id}/final_model.pt")
    print("Training Finished!")
    return policy


def val(model, env, device, num_ep=5):
    """
    Evaluation function: Runs 'num_ep' episodes and returns average reward.
    Uses deterministic policy (or samples) in eval mode.
    """
    rew_sum = 0
    model.eval() # Set model to evaluation mode
    
    for i in range(num_ep):
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0) # Add batch dim
        done = False
        
        while not done:
            with torch.no_grad():
                # Get action
                action, _, _, _ = model.action_value(obs)
            
            # Convert tensor to scalar for env
            action_scalar = action.cpu().numpy().item()
            
            next_obs, reward, terminated, truncated, _ = env.step(action_scalar)
            
            rew_sum += reward
            done = terminated or truncated
            obs = torch.from_numpy(next_obs).float().to(device).unsqueeze(0)

    model.train() # Set model back to training mode
    return rew_sum / num_ep

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Intersection-v0")
    parser.add_argument("--epochs", type=int, default=1000) # Total training iterations
    parser.add_argument("--num_envs", type=int, default=4)  # Number of parallel environments
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4) # Learning rate
    parser.add_argument("--num_steps", type=int, default=200) # Steps per rollout (matches typical data length)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=10) # PPO updates per iteration
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", action="store_true", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    args = parser.parse_args()  
    
    # --- 1. Load Data ---
    data_path = "data/expert_agent_trajs.npy" # Should match the output from process_data.py
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found! Please run process_data.py first.")
        exit()
        
    print(f"Loading expert data from {data_path}...")
    expert_data = np.load(data_path, allow_pickle=True)
    print(f"Loaded {len(expert_data)} trajectories.")

    # --- 2. Start Training ---
    train(args, expert_data)