import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

# --- Configuration Parameters ---
DATA_ROOT_DIR = 'intersection_data_1106'  # Root directory of your data
OUTPUT_FILE = 'data/expert_agent_trajectories.npy' # Output file path
DT = 0.1  # Sampling time interval (seconds)
GOAL_Y = 1.5 # Target Y position (optional, for reference)

# Additional data folders to process
NOISY_DATA_FOLDERS = ['noisy_keep_straight', 'noisy_leftturn', 'noisy_rightturn']
# Map folder names to categories
FOLDER_TO_CATEGORY = {
    'noisy_keep_straight': 'keep straight',
    'noisy_leftturn': 'left turn',
    'noisy_rightturn': 'right turn'
}

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

def process_noisy_folder_agent_trajectories(folder_path):
    """
    Process a noisy data folder to extract agent trajectories with matching ego files.
    Returns list of (states, actions) tuples and their metadata.
    """
    all_states = []
    all_actions = []
    trajectory_metadata = []  # Store metadata for tracking
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist. Skipping.")
        return all_states, all_actions, trajectory_metadata
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Separate agent and ego files
    agent_files = sorted([f for f in csv_files if 'agent' in f.lower()])
    ego_files = sorted([f for f in csv_files if 'ego' in f.lower()])
    
    # Check parent directory for ego files if not found in folder
    if len(ego_files) == 0:
        parent_dir = os.path.dirname(os.path.abspath(folder_path))
        if os.path.exists(parent_dir):
            parent_csv_files = [f for f in os.listdir(parent_dir) if f.endswith('.csv') and 'ego' in f.lower()]
            if len(parent_csv_files) > 0:
                ego_files = sorted(parent_csv_files)
                print(f"Found {len(ego_files)} ego files in parent directory for {folder_path}")
    
    if len(agent_files) == 0:
        print(f"Warning: No agent files found in {folder_path}. Skipping.")
        return all_states, all_actions, trajectory_metadata
    
    print(f"Processing {folder_path}: Found {len(agent_files)} agent files, {len(ego_files)} ego files.")
    
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
    
    if len(common_numbers) == 0:
        print(f"Warning: No matching agent/ego pairs found in {folder_path}")
        return all_states, all_actions, trajectory_metadata
    
    print(f"Found {len(common_numbers)} matching agent/ego pairs.")
    
    for num in common_numbers:
        agent_file = os.path.join(folder_path, agent_dict[num])
        
        # Check if ego file is in folder_path or parent directory
        if num in ego_dict and os.path.exists(os.path.join(folder_path, ego_dict[num])):
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
        if s is not None and a is not None:
            all_states.append(s)
            all_actions.append(a)
            trajectory_metadata.append((folder_path, agent_dict[num], num))
    
    return all_states, all_actions, trajectory_metadata

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

def validate_trajectory_alignment(df_agent, df_ego, scenario_name):
    """
    Validate that agent and ego trajectories are properly aligned.
    Returns True if aligned, False otherwise.
    """
    # Check time overlap
    agent_t_start = df_agent['time'].iloc[0]
    agent_t_end = df_agent['time'].iloc[-1]
    ego_t_start = df_ego['time'].iloc[0]
    ego_t_end = df_ego['time'].iloc[-1]
    
    overlap_start = max(agent_t_start, ego_t_start)
    overlap_end = min(agent_t_end, ego_t_end)
    overlap_duration = overlap_end - overlap_start
    
    if overlap_duration < 0.1:  # Less than 100ms overlap
        print(f"Warning: {scenario_name} has minimal time overlap ({overlap_duration:.3f}s)")
        return False
    
    # Check if time ranges are reasonable
    agent_duration = agent_t_end - agent_t_start
    ego_duration = ego_t_end - ego_t_start
    
    # If one trajectory is much longer than the other, it might indicate a mismatch
    duration_ratio = max(agent_duration, ego_duration) / min(agent_duration, ego_duration)
    if duration_ratio > 2.0:
        print(f"Warning: {scenario_name} has significant duration mismatch (ratio: {duration_ratio:.2f})")
        print(f"  Agent: {agent_duration:.2f}s, Ego: {ego_duration:.2f}s")
    
    return True

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
    
    # 2.5. Validate trajectory alignment
    if not validate_trajectory_alignment(df_agent, df_ego, scenario_name):
        # Don't skip, but warn - the interpolation should handle it
        pass

    # 3. Define Standard Timeline (0.1s interval)
    # Get the time intersection to ensure both ego and agent are present in the scene
    t_start = max(df_agent['time'].iloc[0], df_ego['time'].iloc[0])
    t_end = min(df_agent['time'].iloc[-1], df_ego['time'].iloc[-1])

    # If the overlap duration is too short, skip this scenario
    if t_end - t_start < DT:
        print(f"Skipping {scenario_name}: Duration too short.")
        return None, None

    # Create the standard time grid
    # Use np.linspace to ensure we include points up to (but not beyond) t_end
    # Calculate number of points to include t_end if possible
    num_points = int(np.ceil((t_end - t_start) / DT)) + 1
    t_grid = np.linspace(t_start, t_end, num_points)
    # Ensure we don't exceed t_end due to floating point precision
    t_grid = t_grid[t_grid <= t_end]

    # 4. Data Interpolation Function
    def interpolate_data(df, target_times):
        # Create interpolation function: input time -> output all columns
        # axis=0 means interpolate along rows
        # Use bounds_error=False and fill_value='extrapolate' but clip to valid range
        # to avoid large extrapolation errors
        df_times = df['time'].values
        valid_mask = (target_times >= df_times[0]) & (target_times <= df_times[-1])
        
        if not np.all(valid_mask):
            # Warn if we need to extrapolate significantly
            extrapolated = np.sum(~valid_mask)
            if extrapolated > len(target_times) * 0.1:  # More than 10% extrapolation
                print(f"Warning: {scenario_name} requires extrapolation for {extrapolated}/{len(target_times)} points")
        
        f = interp1d(df_times, df.values, axis=0, kind='linear', 
                     bounds_error=False, fill_value="extrapolate")
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
    # Note: acc_ego[i] represents acceleration from t_grid[i] to t_grid[i+1]
    acc_ego = np.diff(v_ego) / DT
    
    # Clip acceleration to strictly adhere to physical limits (e.g., Jackal robot limits)
    # This also helps filter out noise from interpolation
    acc_ego = np.clip(acc_ego, -3.2, 3.2) # Assuming max_accel is 3.2

    # Align State and Action
    # Since np.diff reduces length by 1, we drop the last state to ensure 1-to-1 mapping
    # states[i] at time t_grid[i] corresponds to action[i] (acceleration from t_grid[i] to t_grid[i+1])
    states = states[:-1]
    actions = acc_ego.reshape(-1, 1) # Reshape to (N, 1)
    
    # Verify alignment: states and actions should have the same length
    if len(states) != len(actions):
        print(f"Warning: {scenario_name} state-action length mismatch: {len(states)} vs {len(actions)}")
        return None, None

    return states, actions

def process_ego_data_only():
    """
    Process all ego data from intersection_data_1106 folder.
    Returns list of ego trajectories (states and actions).
    """
    all_ego_states = []
    all_ego_actions = []
    
    if not os.path.exists(DATA_ROOT_DIR):
        print(f"Warning: {DATA_ROOT_DIR} does not exist. Skipping ego data processing.")
        return all_ego_states, all_ego_actions
    
    subfolders = [f.path for f in os.scandir(DATA_ROOT_DIR) if f.is_dir()]
    print(f"Processing ego data from {DATA_ROOT_DIR}: Found {len(subfolders)} scenarios.")
    
    for folder in subfolders:
        # Process each scenario to extract ego data
        s, a = process_single_scenario(folder)
        if s is not None:
            all_ego_states.append(s)
            all_ego_actions.append(a)
    
    return all_ego_states, all_ego_actions

def main():
    # ===== PART 1: Process agent trajectories from noisy folders and split 80/20 =====
    print("=" * 60)
    print("PART 1: Processing agent trajectories from noisy folders")
    print("=" * 60)
    
    all_noisy_states = []
    all_noisy_actions = []
    all_noisy_categories = []
    all_noisy_metadata = []
    
    for noisy_folder in NOISY_DATA_FOLDERS:
        if not os.path.exists(noisy_folder):
            print(f"Warning: {noisy_folder} does not exist. Skipping.")
            continue
        
        # Get category for this folder
        category = FOLDER_TO_CATEGORY.get(noisy_folder, 'unknown')
        
        # Process agent trajectories from this folder (with matching ego files)
        states, actions, metadata = process_noisy_folder_agent_trajectories(noisy_folder)
        
        # Add category for each trajectory
        for s, a, meta in zip(states, actions, metadata):
            all_noisy_states.append(s)
            all_noisy_actions.append(a)
            all_noisy_categories.append(category)
            all_noisy_metadata.append(meta)
    
    if len(all_noisy_states) == 0:
        print("Warning: No agent trajectories found in noisy folders.")
    else:
        print(f"\nTotal agent trajectories from noisy folders: {len(all_noisy_states)}")
        
        # Random 80/20 train/test split
        indices = np.arange(len(all_noisy_states))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Prepare training data (concatenate all training trajectories)
        train_states_list = [all_noisy_states[i] for i in train_indices]
        train_actions_list = [all_noisy_actions[i] for i in train_indices]
        train_states = np.concatenate(train_states_list, axis=0)
        train_actions = np.concatenate(train_actions_list, axis=0)
        
        # Prepare test data (keep as list of trajectories with categories)
        test_states_list = [all_noisy_states[i] for i in test_indices]
        test_actions_list = [all_noisy_actions[i] for i in test_indices]
        test_categories = [all_noisy_categories[i] for i in test_indices]
        test_metadata = [all_noisy_metadata[i] for i in test_indices]
        
        print(f"Train trajectories: {len(train_indices)} (total data points: {train_states.shape[0]})")
        print(f"Test trajectories: {len(test_indices)}")
        print(f"Train state shape: {train_states.shape}, action shape: {train_actions.shape}")
        
        # Save training agent data
        os.makedirs('data', exist_ok=True)
        train_data = {
            'states': train_states,
            'actions': train_actions
        }
        np.save('data/expert_agent_trajectories.npy', train_data)
        print(f"Saved training agent data to data/expert_agent_trajectories.npy")
        
        # Save test data with categories
        # Note: We need to handle the "zigzag" category - check if any trajectories match this
        # For now, we'll use the categories from folder names
        test_data = {
            'states': test_states_list,  # List of trajectory arrays
            'actions': test_actions_list,  # List of trajectory arrays
            'categories': test_categories,
            'metadata': test_metadata
        }
        np.save('data/test_agent_trajectories.npy', test_data)
        print(f"Saved test agent data with categories to data/test_agent_trajectories.npy")
        print(f"  Categories: {set(test_categories)}")
    
    # ===== PART 2: Process ego data from intersection_data_1106 =====
    print("\n" + "=" * 60)
    print("PART 2: Processing ego data from intersection_data_1106")
    print("=" * 60)
    
    ego_states, ego_actions = process_ego_data_only()
    
    if len(ego_states) > 0:
        # Concatenate all ego data
        expert_ego_states = np.concatenate(ego_states, axis=0)
        expert_ego_actions = np.concatenate(ego_actions, axis=0)
        
        print(f"\nEgo data processing complete.")
        print(f"Total ego data points: {expert_ego_states.shape[0]}")
        print(f"Ego state shape: {expert_ego_states.shape}")
        print(f"Ego action shape: {expert_ego_actions.shape}")
        
        # Save ego data
        os.makedirs('data', exist_ok=True)
        ego_data = {
            'states': expert_ego_states,
            'actions': expert_ego_actions
        }
        np.save('data/expert_ego_trajectories.npy', ego_data)
        print(f"Saved ego data to data/expert_ego_trajectories.npy")
    else:
        print("Warning: No ego data found.")
    
    print("\n" + "=" * 60)
    print("All processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()