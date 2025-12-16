import os
import pandas as pd
import numpy as np

def analyze_csv(csv_path):
    """
    Reads a CSV file and calculates motion statistics.
    
    Args:
        csv_path (str): Path to the csv file.
        
    Returns:
        max_acc_x (float): Maximum longitudinal acceleration (m/s^2).
        max_vx (float): Maximum forward speed (m/s).
        max_vy (float): Maximum lateral speed (m/s).
    """
    # Read the CSV file
    # Assuming no header based on your previous snippets, but if there is a header, remove 'header=None'
    # If your csv has headers like 'time,x,y...', pandas handles it automatically.
    # Based on your previous code, I'll assume it has headers or you handle columns by index.
    # Let's assume standard pandas read with auto-detection for safety.
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return 0.0, 0.0, 0.0

    # Ensure data is sorted by time
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)
    
    # Extract necessary columns (using numpy for speed)
    # We use 'vx' for acceleration because it represents forward motion
    try:
        t = df['time'].values
        vx = df['vx'].values  # Longitudinal velocity
        vy = df['vy'].values  # Lateral velocity
    except KeyError:
        # Fallback if column names don't match exactly
        return 0.0, 0.0, 0.0

    # Calculate time differences
    dt = np.diff(t)
    # Avoid division by zero (replace 0 with a very small number)
    dt[dt == 0] = 1e-6
    
    # Calculate Acceleration (a = dv/dt) using longitudinal velocity (vx)
    dv_x = np.diff(vx)
    acc_x = dv_x / dt
    
    # Get Statistics
    # We take absolute value for acceleration to capture braking (deceleration) too
    max_acc_x = np.max(np.abs(acc_x)) 
    max_vx = np.max(vx)
    # Take absolute value for lateral velocity to see max drift magnitude
    max_vy = np.max(np.abs(vy)) 
    
    return max_acc_x, max_vx, max_vy

# --- Main Execution ---

# Define your root directory containing the data folders
root = "/Users/xumenghan/Documents/Cornell courses/CS 6784 ml for feedback systems/project/intersection_data_1106" 

# Initialize containers for global statistics
ego_global_max_acc = -np.inf
ego_global_max_vx = -np.inf
ego_file_max_acc = None

agent_global_max_vx = -np.inf
agent_global_max_vy = -np.inf
agent_file_max_vx = None

# Print Header
print(f"{'Folder Name':<30} | {'Ego Max v':<10} | {'Agent Max v':<10} | {'Status'}")
print("-" * 70)

# Iterate through all subdirectories
for entry in os.scandir(root):
    if entry.is_dir():
        ego_csv = os.path.join(entry.path, "ego.csv")
        agent_csv = os.path.join(entry.path, "agent.csv")

        ego_vx = 0.0
        agent_vx = 0.0
        
        # 1. Analyze Ego Data
        if os.path.exists(ego_csv):
            try:
                e_acc, e_vx, e_vy = analyze_csv(ego_csv)
                ego_vx = e_vx
                
                # Update Global Ego Stats
                if e_vx > ego_global_max_vx: 
                    ego_global_max_vx = e_vx
                if e_acc > ego_global_max_acc: 
                    ego_global_max_acc = e_acc
                    ego_file_max_acc = entry.name
            except Exception as e:
                print(f"Error analyzing Ego in {entry.name}: {e}")

        # 2. Analyze Agent Data
        if os.path.exists(agent_csv):
            try:
                a_acc, a_vx, a_vy = analyze_csv(agent_csv)
                agent_vx = a_vx
                
                # Update Global Agent Stats
                if a_vx > agent_global_max_vx: 
                    agent_global_max_vx = a_vx
                    agent_file_max_vx = entry.name
                if a_vy > agent_global_max_vy: 
                    agent_global_max_vy = a_vy
            except Exception as e:
                print(f"Error analyzing Agent in {entry.name}: {e}")

        # 3. Compare and Print
        # Determine if Ego is significantly faster than Agent
        if ego_vx > 0 and agent_vx > 0:
            status = "Ego Faster" if ego_vx > agent_vx else "Agent Faster"
        else:
            status = "No Data"
            
        print(f"{entry.name:<30} | {ego_vx:.3f}      | {agent_vx:.3f}      | {status}")

# --- Final Summary ---
print("\n" + "="*40)
print("FINAL STATISTICS SUMMARY")
print("="*40)

print(f"EGO  Global Max Vx  : {ego_global_max_vx:.4f} m/s")
print(f"EGO  Global Max Acc : {ego_global_max_acc:.4f} m/s^2 (Found in: {ego_file_max_acc})")
print("-" * 40)
print(f"AGENT Global Max Vx : {agent_global_max_vx:.4f} m/s (Found in: {agent_file_max_vx})")
print(f"AGENT Global Max Vy : {agent_global_max_vy:.4f} m/s (Max Lateral Slide)")

print("="*40)
# Logic Check
if ego_global_max_vx > agent_global_max_vx * 1.5:
    print("    ANALYSIS: Ego is significantly faster than the Agent.")
    print("    It is likely that the Ego is 'rushing' through the intersection")
    print("    before the slower Agent becomes a threat.")
else:
    print("    ANALYSIS: Speeds are comparable. The interaction data seems valid.")



# Folder Name                    | Ego Max v  | Agent Max v | Status
# ----------------------------------------------------------------------
# right_turn_t3                  | 0.368      | 0.419      | Agent Faster
# left_turn_aggressive_t1        | 1.350      | 0.435      | Ego Faster
# keep_straight_polite_t1        | 0.345      | 0.345      | Ego Faster
# right_turn_t2                  | 0.352      | 0.601      | Agent Faster
# keep_straight_aggressive_t1    | 1.302      | 0.424      | Ego Faster
# left_turn_polite_t2            | 0.361      | 0.432      | Agent Faster
# left_turn_polite_t3            | 0.360      | 0.435      | Agent Faster
# keep_straight_polite_t2        | 0.346      | 0.556      | Agent Faster
# left_turn_aggressive_t2        | 1.103      | 0.410      | Ego Faster
# keep_straight_polite_t3        | 0.348      | 0.376      | Agent Faster
# left_turn_aggressive_t3        | 1.105      | 0.391      | Ego Faster
# right_turn_t1                  | 0.367      | 0.415      | Agent Faster
# keep_straight_aggressive_t3    | 1.279      | 0.381      | Ego Faster
# keep_straight_aggressive_t2    | 1.127      | 0.300      | Ego Faster
# left_turn_polite_t1            | 0.352      | 0.424      | Agent Faster

# ========================================
# FINAL STATISTICS SUMMARY
# ========================================
# EGO  Global Max Vx  : 1.3499 m/s
# EGO  Global Max Acc : 7.4451 m/s^2 (Found in: left_turn_aggressive_t2)
# ----------------------------------------
# AGENT Global Max Vx : 0.6013 m/s (Found in: right_turn_t2)
# AGENT Global Max Vy : 0.0496 m/s (Max Lateral Slide)
# ========================================
#     ANALYSIS: Ego is significantly faster than the Agent.
#     It is likely that the Ego is 'rushing' through the intersection
#     before the slower Agent becomes a threat.