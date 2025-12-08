import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# --- Configuration Parameters ---
DATA_ROOT_DIR = 'intersection_data_1106'  # Root directory of your data
OUTPUT_FILE = 'data/expert_ego_trajectories.npy' # Output file path
DT = 0.1  # Sampling time interval (seconds)
GOAL_Y = 1.5 # Target Y position (optional, for reference)

# State structure: [y_ego, v_ego, x_agent, y_agent, vx_agent, vy_agent, sx_agent, sy_agent]
# Action structure: [acc_ego]

def process_single_scenario(folder_path):
    """
    Process a single scenario folder, returning the (State_Sequence, Action_Sequence) pair.
    """
    agent_file = os.path.join(folder_path, 'agent.csv')
    ego_file = os.path.join(folder_path, 'ego.csv')

    # 1. Check if files exist
    if not (os.path.exists(agent_file) and os.path.exists(ego_file)):
        print(f"Skipping {folder_path}: Files not found.")
        return None, None

    # 2. Read CSV files
    try:
        df_agent = pd.read_csv(agent_file)
        df_ego = pd.read_csv(ego_file)
    except Exception as e:
        print(f"Error reading {folder_path}: {e}")
        return None, None

    # 3. Define Standard Timeline (0.1s interval)
    # Get the time intersection to ensure both ego and agent are present in the scene
    t_start = max(df_agent['time'].iloc[0], df_ego['time'].iloc[0])
    t_end = min(df_agent['time'].iloc[-1], df_ego['time'].iloc[-1])

    # If the overlap duration is too short, skip this scenario
    if t_end - t_start < DT:
        print(f"Skipping {folder_path}: Duration too short.")
        return None, None

    # Create the standard time grid
    t_grid = np.arange(t_start, t_end, DT)

    # 4. Data Interpolation Function
    def interpolate_data(df, target_times):
        # Create interpolation function: input time -> output all columns
        # axis=0 means interpolate along rows
        # fill_value="extrapolate" allows for minor extrapolation at boundaries
        f = interp1d(df['time'], df.values, axis=0, kind='linear', fill_value="extrapolate")
        interpolated_data = f(target_times)
        # Convert back to DataFrame to access by column names
        return pd.DataFrame(interpolated_data, columns=df.columns)

    # Perform interpolation
    agent_interp = interpolate_data(df_agent, t_grid)
    ego_interp = interpolate_data(df_ego, t_grid)

    # 5. Extract Features to Construct State
    # Target definition: s_t = [y_ego, v_ego, x_agent, y_agent, vx_agent, vy_agent, sx_agent, sy_agent]
    
    # Ego Features
    y_ego = ego_interp['y'].values
    # Ego velocity: using 'vy' as the ego controls longitudinal acceleration
    v_ego = ego_interp['vy'].values 

    # Agent Features
    x_agent = agent_interp['x'].values
    y_agent = agent_interp['y'].values
    vx_agent = agent_interp['vx'].values
    vy_agent = agent_interp['vy'].values
    # 'sx', 'sy' correspond to the uncertainty (sigma) mentioned in your formula
    sx_agent = agent_interp['sx'].values
    sy_agent = agent_interp['sy'].values

    # Stack features to form State (N, 8)
    # np.stack along axis 1 makes them columns
    states = np.stack([
        y_ego, v_ego, 
        x_agent, y_agent, vx_agent, vy_agent, sx_agent, sy_agent
    ], axis=1)

    # 6. Calculate Action (Acceleration)
    # a_t = (v_{t+1} - v_t) / dt
    # Using np.diff calculates the difference between consecutive elements
    acc_ego = np.diff(v_ego) / DT
    
    # Clip acceleration to strictly adhere to physical limits (e.g., Jackal robot limits)
    # This also helps filter out noise from interpolation
    acc_ego = np.clip(acc_ego, -3.2, 3.2) # Assuming max_accel is 3.2

    # Align State and Action
    # Since np.diff reduces length by 1, we drop the last state to ensure 1-to-1 mapping
    states = states[:-1]
    actions = acc_ego.reshape(-1, 1) # Reshape to (N, 1)

    return states, actions

def main():
    all_states = []
    all_actions = []
    
    # Scan all subdirectories in the data root
    subfolders = [f.path for f in os.scandir(DATA_ROOT_DIR) if f.is_dir()]
    
    print(f"Found {len(subfolders)} scenarios. Processing...")

    for folder in subfolders:
        s, a = process_single_scenario(folder)
        if s is not None:
            all_states.append(s)
            all_actions.append(a)

    # Concatenate all data
    if len(all_states) > 0:
        expert_states = np.concatenate(all_states, axis=0)
        expert_actions = np.concatenate(all_actions, axis=0)
        
        print(f"\nProcessing complete.")
        print(f"Total data points: {expert_states.shape[0]}")
        print(f"State shape: {expert_states.shape}")
        print(f"Action shape: {expert_actions.shape}")

        # Save the data
        # Saving as a dictionary for easier loading later
        save_path = 'data/expert_ego_trajectories.npy'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        np.save(save_path, {'states': expert_states, 'actions': expert_actions})
        print(f"Saved to {save_path}")
        
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()