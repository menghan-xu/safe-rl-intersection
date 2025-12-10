import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# --- Configuration Parameters ---
DATA_ROOT_DIR = 'intersection_data_1106'  # Root directory of your data
OUTPUT_FILE = 'data/expert_ego_trajectories.npy' # Output file path
DT = 0.1  # Sampling time interval (seconds)
GOAL_Y = 1.5 # Target Y position (optional, for reference)

# Additional data folders to process
NOISY_DATA_FOLDERS = ['noisy_keep_straight', 'noisy_leftturn', 'noisy_rightturn']

# State structure: [y_ego, v_ego, x_agent, y_agent, vx_agent, vy_agent, sx_agent, sy_agent]
# Action structure: [acc_ego]

def process_single_scenario(folder_path):
    """
    Process a single scenario folder, returning the (State_Sequence, Action_Sequence) pair.
    Supports both old structure (subdirectories with agent.csv/ego.csv) and new structure (CSV files directly in folder).
    """
    # Try old structure first (subdirectories with agent.csv and ego.csv)
    agent_file = os.path.join(folder_path, 'agent.csv')
    ego_file = os.path.join(folder_path, 'ego.csv')

    # If old structure doesn't exist, try to find files by pattern matching
    if not (os.path.exists(agent_file) and os.path.exists(ego_file)):
        # Look for numbered agent and ego files (e.g., noisy_agent_1.csv, noisy_ego_1.csv)
        folder_name = os.path.basename(folder_path)
        parent_dir = os.path.dirname(folder_path)
        
        # Get all CSV files in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        # Try to find matching agent and ego files
        agent_files = [f for f in csv_files if 'agent' in f.lower()]
        ego_files = [f for f in csv_files if 'ego' in f.lower()]
        
        if len(agent_files) > 0 and len(ego_files) > 0:
            # Extract numbers from filenames and match them
            def extract_number(filename):
                match = re.search(r'(\d+)', filename)
                return int(match.group(1)) if match else None
            
            agent_dict = {extract_number(f): f for f in agent_files if extract_number(f) is not None}
            ego_dict = {extract_number(f): f for f in ego_files if extract_number(f) is not None}
            
            # Find matching pairs
            common_numbers = set(agent_dict.keys()) & set(ego_dict.keys())
            if common_numbers:
                # Use the first matching pair
                num = sorted(common_numbers)[0]
                agent_file = os.path.join(folder_path, agent_dict[num])
                ego_file = os.path.join(folder_path, ego_dict[num])
            else:
                print(f"Skipping {folder_path}: No matching agent/ego file pairs found.")
                return None, None
        else:
            print(f"Skipping {folder_path}: Files not found.")
            return None, None

    # 1. Check if files exist
    if not (os.path.exists(agent_file) and os.path.exists(ego_file)):
        print(f"Skipping {folder_path}: Files not found.")
        return None, None

    # Use the shared processing function
    return process_single_file_pair(agent_file, ego_file, folder_path)

def process_noisy_folder(folder_path):
    """
    Process a noisy data folder (e.g., noisy_keep_straight) where CSV files are directly in the folder.
    Matches agent and ego files by their number pattern.
    """
    all_states = []
    all_actions = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist. Skipping.")
        return all_states, all_actions
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Separate agent and ego files
    agent_files = sorted([f for f in csv_files if 'agent' in f.lower()])
    ego_files = sorted([f for f in csv_files if 'ego' in f.lower()])
    
    if len(ego_files) == 0:
        print(f"Warning: No ego files found in {folder_path}. Checking parent directory...")
        # Check if ego files are in the parent directory with matching numbers
        parent_dir = os.path.dirname(os.path.abspath(folder_path))
        parent_csv_files = []
        if os.path.exists(parent_dir):
            parent_csv_files = [f for f in os.listdir(parent_dir) if f.endswith('.csv') and 'ego' in f.lower()]
        
        if len(parent_csv_files) > 0:
            print(f"Found {len(parent_csv_files)} ego files in parent directory. Attempting to match...")
            ego_files = sorted(parent_csv_files)
        else:
            print(f"Warning: No ego files found in {folder_path} or parent directory. Skipping this folder.")
            return all_states, all_actions
    
    # Extract numbers from filenames and match them
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else None
    
    # Create dictionaries mapping numbers to filenames
    agent_dict = {}
    for f in agent_files:
        num = extract_number(f)
        if num is not None:
            agent_dict[num] = f
    
    ego_dict = {}
    for f in ego_files:
        num = extract_number(f)
        if num is not None:
            ego_dict[num] = f
    
    # Find matching pairs
    common_numbers = sorted(set(agent_dict.keys()) & set(ego_dict.keys()))
    
    print(f"Processing {folder_path}: Found {len(common_numbers)} matching agent/ego pairs.")
    
    for num in common_numbers:
        agent_file = os.path.join(folder_path, agent_dict[num])
        # Check if ego file is in folder_path or parent directory
        if num in ego_dict:
            ego_file = os.path.join(folder_path, ego_dict[num])
        else:
            # Try parent directory
            parent_dir = os.path.dirname(os.path.abspath(folder_path))
            parent_ego_file = os.path.join(parent_dir, ego_dict[num])
            if os.path.exists(parent_ego_file):
                ego_file = parent_ego_file
            else:
                print(f"Skipping pair {num}: Ego file not found.")
                continue
        
        # Process this pair using the existing logic
        s, a = process_single_file_pair(agent_file, ego_file, f"{folder_path}/pair_{num}")
        if s is not None:
            all_states.append(s)
            all_actions.append(a)
    
    return all_states, all_actions

def process_single_file_pair(agent_file, ego_file, scenario_name):
    """
    Process a single agent/ego file pair, returning the (State_Sequence, Action_Sequence) pair.
    """
    # 1. Check if files exist
    if not (os.path.exists(agent_file) and os.path.exists(ego_file)):
        print(f"Skipping {scenario_name}: Files not found.")
        return None, None

    # 2. Read CSV files
    try:
        df_agent = pd.read_csv(agent_file)
        df_ego = pd.read_csv(ego_file)
    except Exception as e:
        print(f"Error reading {scenario_name}: {e}")
        return None, None

    # 3. Define Standard Timeline (0.1s interval)
    # Get the time intersection to ensure both ego and agent are present in the scene
    t_start = max(df_agent['time'].iloc[0], df_ego['time'].iloc[0])
    t_end = min(df_agent['time'].iloc[-1], df_ego['time'].iloc[-1])

    # If the overlap duration is too short, skip this scenario
    if t_end - t_start < DT:
        print(f"Skipping {scenario_name}: Duration too short.")
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
    
    # Process original data structure (subdirectories with agent.csv/ego.csv)
    if os.path.exists(DATA_ROOT_DIR):
        subfolders = [f.path for f in os.scandir(DATA_ROOT_DIR) if f.is_dir()]
        print(f"Processing {DATA_ROOT_DIR}: Found {len(subfolders)} scenarios.")
        
        for folder in subfolders:
            s, a = process_single_scenario(folder)
            if s is not None:
                all_states.append(s)
                all_actions.append(a)
    else:
        print(f"Warning: {DATA_ROOT_DIR} does not exist. Skipping.")
    
    # Process noisy data folders
    for noisy_folder in NOISY_DATA_FOLDERS:
        if os.path.exists(noisy_folder):
            states, actions = process_noisy_folder(noisy_folder)
            all_states.extend(states)
            all_actions.extend(actions)
        else:
            print(f"Warning: {noisy_folder} does not exist. Skipping.")

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