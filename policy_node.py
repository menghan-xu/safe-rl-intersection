#!/usr/bin/env python3
"""
Policy Node
-----------
Runs a trained RL policy for intersection navigation.

Pipeline:
1. Subscribe to ego and agent odometry
2. Estimate agent velocity + uncertainty
3. Construct 8D policy state
4. Run policy → lateral acceleration a_y
5. Integrate a_y → v_y
6. Publish Twist command (v, omega=0)


FOR SWITCHING MODELS AT RUNTIME:
# Use vanilla_lppo (default)
ros2 run intersection_rl lppo_node

# Use lppo_bc
ros2 run intersection_rl lppo_node --ros-args -p model_name:=lppo_bc

# Use lppo_random_startingPt
ros2 run intersection_rl lppo_node --ros-args -p model_name:=lppo_random_startingPt

# Use lppo_bc_random_startingPt
ros2 run intersection_rl lppo_node --ros-args -p model_name:=lppo_bc_random_startingPt

# Use lppo_bff_random_startingPt
ros2 run intersection_rl lppo_node --ros-args -p model_name:=lppo_bff_random_startingPt
"""
import pickle
import io
            
import rclpy
from rclpy.node import Node
import torch
import numpy as np
from collections import deque
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import os
from ament_index_python.packages import get_package_share_directory

from intersection_rl.models import ContinuousActorCritic


class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')

        # ============================================================
        # Parameters
        # ============================================================
        self.declare_parameter('model_name', 'vanilla_lppo')
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('agent_history_len', 5)

        self.dt = self.get_parameter('dt').value
        self.agent_history_len = self.get_parameter('agent_history_len').value
        model_name = self.get_parameter('model_name').value

        # ============================================================
        # Load trained policy
        # ============================================================
        pkg_dir = get_package_share_directory('intersection_rl')
        model_path = os.path.join(pkg_dir, model_name)

        self.get_logger().info(f'Loading policy from {model_path}')

        self.model = ContinuousActorCritic(
            state_dim=8,
            action_dim=1,
            max_action=3.2,
        )
        
        # Load model weights
        if os.path.isfile(model_path):  # Single .pth file
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        else:  # Directory format - use torch._load with proper persistent_load

            # Create custom unpickler that can handle persistent IDs
            class TorchUnpickler(pickle.Unpickler):
                def __init__(self, file, storage_dir):
                    super().__init__(file)
                    self.storage_dir = storage_dir
                
                def persistent_load(self, pid):
                    # pid format: ('storage', StorageType, key, location, size)
                    if isinstance(pid, tuple) and len(pid) >= 3:
                        storage_type = pid[1]  # The actual storage class (e.g., FloatStorage)
                        key = pid[2]
                        size = pid[4] if len(pid) > 4 else 0
                    else:
                        raise ValueError(f"Unexpected pid format: {pid}")
                    
                    # Load tensor data from data/ directory
                    data_file = os.path.join(self.storage_dir, 'data', str(key))
                    # Create typed storage from file (size is already in bytes)
                    return storage_type.from_file(data_file, shared=False, size=size)
            
            with open(os.path.join(model_path, 'data.pkl'), 'rb') as f:
                unpickler = TorchUnpickler(f, model_path)
                state_dict = unpickler.load()
        
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # ============================================================
        # Internal state
        # ============================================================
        self.ego_state = None           # [y_ego, v_ego]
        self.agent_pose = None          # [x_agent, y_agent]
        self.agent_velocity = np.zeros(2)
        self.agent_uncertainty = np.ones(2) * 0.1

        self.agent_history = deque(maxlen=self.agent_history_len)

        # Integrated control state
        self.vy_cmd = 0.0

        # ============================================================
        # ROS2 subscriptions and publishers
        # ============================================================
        # ego odometry SUBSCRIBER
        self.create_subscription(
            Odometry,
            '/j100_0088/odometry/filtered',
            self.ego_odom_cb,
            10,
        )
        # agent odometry SUBCRIBER
        self.create_subscription(
            Odometry,
            '/j100_0484/odometry/filtered',
            self.agent_odom_cb,
            10,
        )
        # cmd_vel PUBLISHER
        self.cmd_pub = self.create_publisher(
            Twist,
            '/j100_0088/platform/cmd_vel_unstamped',
            10,
        )

        self.timer = self.create_timer(self.dt, self.policy_cb)

        self.get_logger().info('Policy node running.')

    # ================================================================
    # Ego odometry callback
    # ================================================================
    def ego_odom_cb(self, msg: Odometry):
        y = msg.pose.pose.position.y
        vy = msg.twist.twist.linear.y
        self.ego_state = np.array([y, vy])

    # ================================================================
    # Agent odometry callback (mean + covariance from estimator)
    # ================================================================
    def agent_odom_cb(self, msg: Odometry):
        # --- Position ---
        self.agent_pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])

        # --- Velocity ---
        self.agent_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        ])

        # --- Uncertainty (std from pose covariance) ---
        # pose.covariance is a 6x6 row-major matrix
        var_x = msg.pose.covariance[0]   # x variance
        var_y = msg.pose.covariance[7]   # y variance

        self.agent_uncertainty = np.sqrt([var_x, var_y])
        # self.agent_uncertainty = np.clip( # clipped covariance
        #     np.sqrt([var_x, var_y]),
        #     0.01,
        #     1.0,
        # )

    # ================================================================
    # Policy execution loop
    # ================================================================
    def policy_cb(self):
        if self.ego_state is None or self.agent_pose is None:
            return

        state = np.concatenate([
            self.ego_state,          # y_ego, v_ego
            self.agent_pose,         # x_agent, y_agent
            self.agent_velocity,     # vx_agent, vy_agent
            self.agent_uncertainty,  # sigma_x, sigma_y
        ])

        state_t = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            x = self.model.trunk(state_t)
            ay = torch.tanh(self.model.mu_head(x)) * self.model.max_action
            ay = ay.item()
            self.get_logger().info(f"{ay}")

        # Integrate acceleration to velocity -- one step Euler
        # self.vy_cmd += ay * self.dt
        self.vy_cmd = self.vy_cmd + ay * 0.1

        # Publish Twist
        msg = Twist()
        msg.linear.x = self.vy_cmd # linear.x is the vel in the direction of heading!!
        msg.linear.y = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

        self.get_logger().info(
            f'ay={ay:.3f}  vy_cmd={self.vy_cmd:.3f}',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
