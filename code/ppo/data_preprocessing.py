import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def process_single_track(csv_path, target_dt=0.1):
    """
    Reads the CSV file and performs downsampling/interpolation to a fixed dt=0.1s.
    Retains columns: x, y, vx, vy, sx, sy.
    """
    col_names = ['time', 'x', 'y', 'yaw', 'vx', 'vy', 'sx', 'sy', 'svx', 'svy']
    df = pd.read_csv(csv_path, names=col_names, header=None)

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    
    target_cols = ['x', 'y', 'vx', 'vy', 'sx', 'sy']
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=target_cols)

    times = df['time'].values
    times = times - times[0]
    
    max_time = times[-1]
    new_times = np.arange(0, max_time, target_dt)
    
    # Interpolation, converts ~1000Hz data to 10Hz
    target_cols = ['x', 'y', 'vx', 'vy', 'sx', 'sy']
    data_interpolated = []
    
    for col in target_cols:
        f = interp1d(times, df[col].values, kind='linear', fill_value="extrapolate")
        data_interpolated.append(f(new_times))
        
    processed_data = np.stack(data_interpolated, axis=1)
    return processed_data

def load_all_data(root_dir):
    """
    Iterates through all subfolders in the root directory and processes 'agent.csv'.
    """
    all_agent_trajs = []
    

    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    
    print(f"Found {len(subfolders)} subfolders in {root_dir}")

    for folder in subfolders:
        agent_file = os.path.join(folder, 'agent.csv') 

        if os.path.exists(agent_file):
            try:
                agent_traj = process_single_track(agent_file, target_dt=0.1)
                all_agent_trajs.append(agent_traj)
                
                print(f"Loaded {os.path.basename(folder)}: Converted to {len(agent_traj)} steps.")
            except Exception as e:
                print(f"Error loading {folder}: {e}")
        else:
            print(f"Warning: 'agent.csv' not found in {folder}")
                
    return all_agent_trajs

if __name__ == "__main__":
    root_path = "../../intersection_data_1106" 
    
    training_data = load_all_data(root_path)
    
    output_filename = "data/expert_agent_trajs.npy"
    np.save(output_filename, np.array(training_data, dtype=object))
    
    print(f"\nProcessing Complete! Saved {len(training_data)} trajectories to {output_filename}")






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