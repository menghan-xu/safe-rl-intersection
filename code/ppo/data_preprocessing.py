import os
import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def process_single_track(csv_path, target_dt=0.1, min_len=500):
    """
    Reads CSV, interpolates to dt=0.1s.
    Logic: 
    1. If length < 500: Pad to 500 (continue velocity until x >= 1.5 then stop).
    2. If length >= 500: Keep original length (NO TRUNCATION).
    
    Handles both CSV files with headers and without headers.
    """
    col_names = ['time', 'x', 'y', 'yaw', 'vx', 'vy', 'sx', 'sy', 'svx', 'svy']
    
    # Read CSV - pandas will auto-detect if there's a header
    df = pd.read_csv(csv_path)
    
    # Check if 'time' column exists (files with headers will have it)
    # If not, assume no header and re-read with column names
    if 'time' not in df.columns:
        df = pd.read_csv(csv_path, names=col_names, header=None)
    else:
        # File has header, ensure we only use the columns we need
        # Select only the columns that exist and are in our expected list
        available_cols = [col for col in col_names if col in df.columns]
        df = df[available_cols]
        # Rename to match expected order (in case columns are in different order)
        df.columns = col_names[:len(df.columns)]

    # 1. Basic Cleaning
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    
    target_cols = ['x', 'y', 'vx', 'vy', 'sx', 'sy']
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=target_cols)

    # 2. Interpolation
    times = df['time'].values
    times = times - times[0]
    max_time = times[-1]
    new_times = np.arange(0, max_time, target_dt)
    
    data_interpolated = []
    for col in target_cols:
        f = interp1d(times, df[col].values, kind='linear', fill_value="extrapolate")
        data_interpolated.append(f(new_times))
        
    processed_data = np.stack(data_interpolated, axis=1)
    
    # ---------------------------------------------------------
    # 3. Padding Logic Only (No Truncation)
    # ---------------------------------------------------------
    current_steps = processed_data.shape[0]

    if current_steps < min_len:
        steps_to_pad = min_len - current_steps
        last_row = processed_data[-1]
        
        # Unpack last state
        curr_x, curr_y = last_row[0], last_row[1]
        curr_vx, curr_vy = last_row[2], last_row[3]
        curr_sx, curr_sy = last_row[4], last_row[5]
        
        new_rows = []
        
        for _ in range(steps_to_pad):
            if curr_x >= 1.5:
                step_vx, step_vy = 0.0, 0.0
            else:
                step_vx, step_vy = curr_vx, curr_vy
                curr_x += step_vx * target_dt
                curr_y += step_vy * target_dt
                

                if curr_x >= 1.5:
                    curr_x = 1.5
            
            new_row = [curr_x, curr_y, step_vx, step_vy, curr_sx, curr_sy]
            new_rows.append(new_row)
            
        padding_array = np.array(new_rows)
        processed_data = np.vstack([processed_data, padding_array])
    


    return processed_data

def load_all_data(root_dir):
    """
    Load agent trajectories from subdirectories (original structure).
    Each subdirectory should contain an 'agent.csv' file.
    """
    all_agent_trajs = []
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    
    print(f"Found {len(subfolders)} subfolders in {root_dir}")

    for folder in subfolders:
        agent_file = os.path.join(folder, 'agent.csv') 

        if os.path.exists(agent_file):
            try:
                agent_traj = process_single_track(agent_file, target_dt=0.1, min_len=500)
                all_agent_trajs.append(agent_traj)
                
                print(f"Loaded {os.path.basename(folder)}: Final shape {agent_traj.shape}")
            except Exception as e:
                print(f"Error loading {folder}: {e}")
        else:
            print(f"Warning: 'agent.csv' not found in {folder}")
                
    return all_agent_trajs

def load_noisy_folder(folder_path):
    """
    Load agent trajectories from a noisy data folder where CSV files are directly in the folder.
    Files should be named like 'noisy_agent_1.csv', 'noisy_agent_2.csv', etc.
    """
    all_agent_trajs = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist. Skipping.")
        return all_agent_trajs
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Filter for agent files
    agent_files = sorted([f for f in csv_files if 'agent' in f.lower()])
    
    print(f"Processing {folder_path}: Found {len(agent_files)} agent files.")
    
    for agent_file in agent_files:
        agent_path = os.path.join(folder_path, agent_file)
        try:
            agent_traj = process_single_track(agent_path, target_dt=0.1, min_len=500)
            all_agent_trajs.append(agent_traj)
            print(f"Loaded {agent_file}: Final shape {agent_traj.shape}")
        except Exception as e:
            print(f"Error loading {agent_path}: {e}")
    
    return all_agent_trajs

if __name__ == "__main__":
    # Configuration
    # Get the project root directory (two levels up from code/ppo/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    root_path = os.path.join(project_root, "intersection_data_1106")
    noisy_folders = [
        os.path.join(project_root, "noisy_keep_straight"),
        os.path.join(project_root, "noisy_leftturn"),
        os.path.join(project_root, "noisy_rightturn")
    ]
    
    all_training_data = []
    
    # Process original data structure (subdirectories with agent.csv)
    if os.path.exists(root_path):
        training_data = load_all_data(root_path)
        all_training_data.extend(training_data)
    else:
        print(f"Warning: {root_path} does not exist. Skipping.")
    
    # Process noisy data folders (CSV files directly in folders)
    for noisy_folder in noisy_folders:
        if os.path.exists(noisy_folder):
            noisy_data = load_noisy_folder(noisy_folder)
            all_training_data.extend(noisy_data)
        else:
            print(f"Warning: {noisy_folder} does not exist. Skipping.")

    output_filename = "data/expert_agent_trajs.npy"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    np.save(output_filename, np.array(all_training_data, dtype=object))
    
    print(f"\nProcessing Complete! Saved {len(all_training_data)} trajectories to {output_filename}")
    if len(all_training_data) > 0:
        lengths = [t.shape[0] for t in all_training_data]
        print(f"Trajectory lengths range from {min(lengths)} to {max(lengths)}")
    else:
        print("Warning: No trajectories were processed.")




# Found 15 subfolders in ../../intersection_data_1106
# Loaded right_turn_t3: Converted to 156 steps.
# Loaded left_turn_aggressive_t1: Converted to 208 steps.
# Loaded keep_straight_polite_t1: Converted to 270 steps.
# Loaded right_turn_t2: Converted to 156 steps.
# Loaded keep_straight_aggressive_t1: Converted to 204 steps.
# Loaded left_turn_polite_t2: Converted to 235 steps.
# Loaded left_turn_polite_t3: Converted to 215 steps.
# Loaded keep_straight_polite_t2: Converted to 279 steps.
# Loaded left_turn_aggressive_t2: Converted to 212 steps.
# Loaded keep_straight_polite_t3: Converted to 263 steps.
# Loaded left_turn_aggressive_t3: Converted to 153 steps.
# Loaded right_turn_t1: Converted to 1633 steps.
# Loaded keep_straight_aggressive_t3: Converted to 206 steps.
# Loaded keep_straight_aggressive_t2: Converted to 215 steps.
# Loaded left_turn_polite_t1: Converted to 243 steps.

# Processing Complete! Saved 15 trajectories to data/expert_agent_trajs.npy