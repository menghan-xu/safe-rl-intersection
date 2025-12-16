import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# ==========================================
# 1. Physics Constants (Fixed)
# ==========================================
V_LIMIT = 1.5         # Based on your previous analysis (Expert max ~1.35)
DT = 0.1               # Simulation time step (10Hz)
R_EGO = 0.3328         # Robot radius
R_AGENT = 0.3328       # Robot radius

# Distance Thresholds (Squared)
# d_collision = (Re + Ra)^2
D_COLLISION_SQ = (R_EGO + R_AGENT)**2  

# ==========================================
# 2. Target Values (Balancing Objectives)
# ==========================================
# These targets determine "How much score" we want the agent to get.

# Target A: How much reward should a typical step give?
# 0.2 is a good baseline (accumulates to ~6.0 over 30 steps)
TARGET_STEP_REWARD = 0.2 

# Target B: How much should comfort penalty be relative to reward?
# We usually want penalty to be 10% - 20% of the reward, so it doesn't paralyze the agent.
TARGET_COMFORT_RATIO = 0.2 

# Target C: How high should the Soft Cost be right before a crash?
# We want it to be significant (e.g., 20.0) so Lambda reacts, 
# but not as huge as the Hard Crash Cost (100.0).
TARGET_MAX_SOFT_COST = 20.0

def get_resampled_stats(ego_path, agent_path):
    """
    Reads and resamples data to 10Hz (dt=0.1) to get realistic training physics.
    """
    try:
        # Load CSVs
        df_e = pd.read_csv(ego_path)
        if 'time' not in df_e.columns:
            df_e = pd.read_csv(ego_path, header=None)
            df_e.columns = ['time', 'x', 'y', 'yaw', 'vx', 'vy', 'sx', 'sy', 'svx', 'svy'][:len(df_e.columns)]
            
        df_a = pd.read_csv(agent_path)
        if 'time' not in df_a.columns:
            df_a = pd.read_csv(agent_path, header=None)
            df_a.columns = ['time', 'x', 'y', 'yaw', 'vx', 'vy', 'sx', 'sy', 'svx', 'svy'][:len(df_a.columns)]

        # Sort
        df_e = df_e.sort_values('time').reset_index(drop=True)
        df_a = df_a.sort_values('time').reset_index(drop=True)

        # --- 1. Ego Velocity & Acceleration (Resampled) ---
        t = df_e['time'].values
        vx = df_e['vx'].values
        
        # Remove duplicates
        _, unique_indices = np.unique(t, return_index=True)
        t = t[unique_indices]
        vx = vx[unique_indices]
        
        # Resample to DT=0.1
        duration = t[-1] - t[0]
        if duration < DT: return None
        
        new_t = np.arange(t[0], t[-1], DT)
        f_vx = interp1d(t, vx, kind='linear')
        new_vx = f_vx(new_t)
        
        # Calculate Acceleration (a = dv/dt)
        dv = np.diff(new_vx)
        acc = dv / DT
        
        # --- 2. Agent Uncertainty (Raw) ---
        # We assume sigma doesn't change drastically with sampling frequency
        try:
            sx = df_a['sx'].values
            sy = df_a['sy'].values
            # Combined Sigma = max(sx, sy) according to your notes
            sigma_comb = np.maximum(sx, sy)
            avg_sigma = np.mean(sigma_comb)
        except:
            avg_sigma = 0.0

        return {
            'vx': new_vx,
            'acc_sq': acc**2,
            'sigma': avg_sigma
        }
    except:
        return None

def main():
    root_dir = "../../intersection_data_1106"
    print(f"Analyzing data in '{root_dir}' to calculate weights...\n")
    
    all_vx = []
    all_acc_sq = []
    all_sigmas = []
    
    # --- Data Collection ---
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            ego_csv = os.path.join(entry.path, "ego.csv")
            agent_csv = os.path.join(entry.path, "agent.csv")
            
            if os.path.exists(ego_csv) and os.path.exists(agent_csv):
                stats = get_resampled_stats(ego_csv, agent_csv)
                if stats:
                    all_vx.extend(stats['vx'])
                    all_acc_sq.extend(stats['acc_sq'])
                    all_sigmas.append(stats['sigma'])

    # --- Statistics ---
    avg_v = np.mean(np.abs(all_vx))
    avg_acc_sq = np.mean(all_acc_sq)
    avg_sigma = np.mean(all_sigmas)
    
    print("="*60)
    print("DATA STATISTICS (Simulation Scale)")
    print("="*60)
    print(f"Avg Velocity (|v|)  : {avg_v:.4f} m/s")
    print(f"Avg Accel^2 (a^2)   : {avg_acc_sq:.4f} (m/s^2)^2")
    print(f"Avg Agent Sigma     : {avg_sigma:.6f} m")
    print("-" * 60)

    # --- Calculation ---
    
    # 1. Alpha (Progress Weight)
    # Formula: alpha * avg_v * dt = TARGET_STEP_REWARD
    alpha = TARGET_STEP_REWARD / (avg_v * DT + 1e-6)
    
    # 2. Eta (Comfort Weight)
    # Formula: eta * avg_acc_sq = TARGET_STEP_REWARD * TARGET_COMFORT_RATIO
    eta = (TARGET_STEP_REWARD * TARGET_COMFORT_RATIO) / (avg_acc_sq + 1e-6)
    
    # 3. Beta (Overspeed Weight)
    # Formula: Penalize heavily if speed > limit + 0.2m/s
    # We want penalty at (limit+0.2) to cancel out the progress reward at that speed.
    overspeed_margin = 0.2
    v_high = V_LIMIT + overspeed_margin
    reward_at_high = alpha * v_high * DT
    beta = reward_at_high / (overspeed_margin**2)
    
    # 4. Mu (Cost Scale)
    # D_conservative = (Re + Ra + 2*sigma)^2
    d_cons_sqrt = R_EGO + R_AGENT + avg_sigma
    d_cons_sq = d_cons_sqrt**2
    
    # The gap where Soft Cost is active: [D_collision_sq, D_conservative_sq]
    gap = d_cons_sq - D_COLLISION_SQ
    
    # Formula: mu * gap = TARGET_MAX_SOFT_COST
    if gap < 1e-4:
        mu = 1000.0 # Cap for safety if gap is tiny
    else:
        mu = TARGET_MAX_SOFT_COST / gap

    # 5. Terminal Rewards (Relative to Alpha)
    # Estimate total progress reward for a full run (approx 10m)
    # Total Progress Score = alpha * Distance
    total_progress_score = alpha * 2.5 # Assuming 10m run
    
    # Success reward should be comparable or larger than total progress
    r_terminal = total_progress_score  # Set equal or slightly higher
    r_collision = - (total_progress_score * 2) # Collision should verify painful

    print("\n" + "="*60)
    print("🏆 CALCULATED HYPERPARAMETERS")
    print("="*60)
    print(f"1. w_progress (alpha) : {alpha:.2f}")
    print(f"2. w_comfort (eta)    : {eta:.3f}")
    print(f"3. w_overspeed (beta) : {beta:.1f}")
    print("-" * 60)
    print(f"4. cost_scale (mu)    : {mu:.1f}")
    print(f"   (Based on Gap = {gap:.4f} m^2)")
    print("-" * 60)
    print(f"5. reward_success     : +{r_terminal:.0f} (Approx)")
    print(f"6. reward_collision   : {r_collision:.0f} (Approx)")
    print("="*60)

if __name__ == "__main__":
    main()