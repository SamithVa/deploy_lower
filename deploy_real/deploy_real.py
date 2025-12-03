"""
deploy_real.py - Real Robot Deployment Script for Unitree Humanoid Robots

This script deploys a trained locomotion policy on Unitree humanoid robots (G1, H1, H1_2).
It handles:
    - DDS communication with the robot's low-level controller
    - Reading sensor data (IMU, joint encoders, remote controller)
    - Running the neural network policy to generate actions
    - Sending motor commands with position/velocity/torque control

Supported robots:
    - G1: Uses HG message type
    - H1: Uses GO message type  
    - H1_2: Uses HG message type
"""

from typing import Union
import numpy as np
import time
import torch
import math

# Unitree SDK2 imports for DDS communication
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

# Message type imports - different robot models use different message formats
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC  # Cyclic Redundancy Check for command validation

# Local helper modules
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config
from scipy.spatial.transform import Rotation as R
from collections import deque


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def quaternion_to_euler_array(quat):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).
    
    Uses the ZYX (yaw-pitch-roll) convention commonly used in robotics.
    
    Args:
        quat: Quaternion in [x, y, z, w] format
        
    Returns:
        np.ndarray: Euler angles [roll, pitch, yaw] in radians
    """
    # Extract quaternion components
    x, y, z, w = quat
    
    # Roll (x-axis rotation) - rotation around the forward axis
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation) - rotation around the lateral axis
    # Clipping prevents numerical issues near gimbal lock
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation) - rotation around the vertical axis
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return np.array([roll_x, pitch_y, yaw_z])


# ==============================================================================
# CONTROLLER CLASS
# ==============================================================================

class Controller:
    """
    Main controller class for deploying RL policies on Unitree humanoid robots.
    
    This class manages:
        - Communication with the robot via DDS (Data Distribution Service)
        - State estimation from IMU and joint encoders
        - Running the trained policy neural network
        - Generating and sending motor commands
        - State machine for safe operation (zero torque -> default pose -> walking)
    
    Attributes:
        config: Configuration object with robot-specific parameters
        remote_controller: Handler for wireless controller inputs
        policy: JIT-compiled PyTorch neural network for locomotion
        low_cmd: Command message to send to robot
        low_state: Latest state message from robot
        qj: Current joint positions (radians)
        dqj: Current joint velocities (rad/s)
        action: Latest policy output (normalized actions)
        hist_obs: History buffer for observation stacking
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the controller with robot configuration.
        
        Args:
            config: Config object containing robot parameters, gains, and paths
        """
        self.config = config
        self.remote_controller = RemoteController()

        # ==================== Policy Network Setup ====================
        # Load the pre-trained locomotion policy (TorchScript format)
        self.policy = torch.jit.load(config.policy_path)
        
        # ==================== State Variables ====================
        # Joint state arrays - sized for the number of controlled joints
        self.qj = np.zeros(config.num_actions, dtype=np.float32)    # Joint positions (rad)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)   # Joint velocities (rad/s)
        self.action = np.zeros((config.num_actions), dtype=np.float32)  # Policy outputs
        self.target_dof_pos = np.zeros((config.num_actions), dtype=np.double)  # Target positions

        # Low-level control counter (used for gait phase calculation)
        self.count_lowlevel = 0

        # ==================== Observation History ====================
        # Frame stacking: store multiple timesteps of observations for temporal context
        self.hist_obs = deque()
        for _ in range(config.frame_stack):
            self.hist_obs.append(np.zeros([1, config.num_obs], dtype=np.double))

        # ==================== Default Joint Angles ====================
        # Standing pose joint angles loaded from config
        self.default_angle = np.zeros((config.num_actions), dtype=np.double)
        for i in range(config.num_actions):
            self.default_angle[i] = config.default_angles[i]

        # ==================== DDS Communication Setup ====================
        if config.msg_type == "hg":
            # HG message type used by G1 and H1_2 robots
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR  # Position-Rate control mode
            self.mode_machine_ = 0  # Robot state machine mode

            # Create publisher for sending commands
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            # Create subscriber for receiving robot state (callback-based)
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # GO message type used by H1 robot
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            # Create publisher for sending commands
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            # Create subscriber for receiving robot state (callback-based)
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # ==================== Initialize Command Message ====================
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    # ==========================================================================
    # STATE HANDLERS (DDS Callbacks)
    # ==========================================================================

    def LowStateHgHandler(self, msg: LowStateHG):
        """
        Callback handler for HG-type robots (G1, H1_2).
        
        Called automatically by the DDS subscriber whenever a new LowState 
        message arrives from the robot. Updates internal state variables.
        
        Args:
            msg: LowStateHG message containing:
                - motor_state: Joint positions (q), velocities (dq), torques
                - imu_state: Orientation quaternion, angular velocity
                - wireless_ctrl: Remote controller button/joystick states
                - mode_machine: Robot's internal state machine mode
        """
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine  # Track mode (HG-specific)
        self.remote_controller.set(self.low_state.wireless_ctrl)

    def LowStateGoHandler(self, msg: LowStateGo):
        """
        Callback handler for GO-type robots (H1).
        
        Called automatically by the DDS subscriber whenever a new LowState 
        message arrives from the robot. Updates internal state variables.
        
        Args:
            msg: LowStateGo message containing:
                - motor_state: Joint positions (q), velocities (dq), torques
                - imu_state: Orientation quaternion, angular velocity
                - wireless_ctrl: Remote controller button/joystick states
        """
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_ctrl)

    # ==========================================================================
    # COMMAND PUBLISHING
    # ==========================================================================

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        """
        Send a low-level motor command to the robot.
        
        Computes CRC checksum for data integrity and publishes via DDS.
        The robot will reject commands with invalid CRC.
        
        Args:
            cmd: Command message containing motor targets for each joint:
                - motor_cmd[i].q: Target position (radians)
                - motor_cmd[i].qd: Target velocity (rad/s)
                - motor_cmd[i].kp: Position gain (Nm/rad)
                - motor_cmd[i].kd: Damping gain (Nm·s/rad)
                - motor_cmd[i].tau: Feedforward torque (Nm)
        """
        cmd.crc = CRC().Crc(cmd)  # Compute CRC for command validation
        self.lowcmd_publisher_.Write(cmd)

    # ==========================================================================
    # STATE MACHINE METHODS
    # ==========================================================================

    def wait_for_low_state(self):
        """Wait for valid state data from the robot before proceeding."""
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        """
        Enter zero torque (passive) state.
        
        In this state, all motors have zero torque command, allowing the robot
        to be manually positioned or to hang in a harness. Waits for the START
        button on the remote controller to proceed to the next state.
        
        This is a safety state - the robot will not actively control its joints.
        """
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)  # Set all motor commands to zero
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2.0
        max_step=0.05
        # 组合索引 & 目标
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        # flip_mask2policy = [1, 1, -1, 1, 1, -1, 1, -1, -1, -1]
        # self.config.default_angles *=  np.array(flip_mask2policy, dtype=np.float32)
        target  = np.concatenate([self.config.default_angles, self.config.arm_waist_target]).astype(np.float32)

        # 从当前 q 开始
        q_now = np.array([self.low_state.motor_state[idx].q for idx in dof_idx], dtype=np.float32)
        steps = max(1, int(total_time / self.config.control_dt))

        # 准备模式过渡增益：比正式站立再软一点
        kps = (self.config.kps + self.config.arm_waist_kps)
        kds = (self.config.kds + self.config.arm_waist_kds)
        kp_move = np.array(kps, dtype=np.float32) * 0.6
        kd_move = np.array(kds, dtype=np.float32) * 1.5

        for i in range(steps):
            # S-curve 时间系数（0→1）
            t = i / steps
            s = 3*t**2 - 2*t**3

            q_des_raw = q_now + (target - q_now) * s

            # 每周期限幅，避免猛跳
            if i == 0:
                last = q_now.copy()
            delta = np.clip(q_des_raw - last, -max_step, max_step)
            q_des = last + delta
            last  = q_des

            # 下发指令（注意索引对应）
            for j, idx in enumerate(dof_idx):
                # 添加关节限位保护
                if j < len(self.config.leg_joint2motor_idx):  # 腿部关节
                    q_clipped = np.clip(q_des[j],
                                    self.config.joint_limits['leg']['q_min'][j],
                                    self.config.joint_limits['leg']['q_max'][j])
                else:  # 臂部/腰部关节
                    arm_idx = j - len(self.config.leg_joint2motor_idx)
                    q_clipped = np.clip(q_des[j],
                                    self.config.joint_limits['arm_waist']['q_min'][arm_idx],
                                    self.config.joint_limits['arm_waist']['q_max'][arm_idx])
                
                self.low_cmd.motor_cmd[idx].q  = float(q_clipped)
                self.low_cmd.motor_cmd[idx].qd = 0.0
                self.low_cmd.motor_cmd[idx].kp = float(kp_move[j])
                self.low_cmd.motor_cmd[idx].kd = float(kd_move[j])
                self.low_cmd.motor_cmd[idx].tau= 0.0

            self.send_cmd(self.low_cmd)
            # 用“定时器式 sleep”保证周期稳定
            self._next_t = getattr(self, "_next_t", time.perf_counter())
            self._next_t += self.config.control_dt
            delay = self._next_t - time.perf_counter()
            time.sleep(delay if delay > 0 else 0)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                # 添加腿部关节限位
                q_clipped = np.clip(self.config.default_angles[i],
                                self.config.joint_limits['leg']['q_min'][i],
                                self.config.joint_limits['leg']['q_max'][i])
                self.low_cmd.motor_cmd[motor_idx].q = q_clipped 
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                # 添加臂部/腰部关节限位
                q_clipped = np.clip(self.config.arm_waist_target[i],
                                self.config.joint_limits['arm_waist']['q_min'][i],
                                self.config.joint_limits['arm_waist']['q_max'][i])
                self.low_cmd.motor_cmd[motor_idx].q = q_clipped
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        self.count_lowlevel += 1
        # 获取所有关节的位置和速度
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq
            # if self.qj[i] < self.config.joint_limits['leg']['q_min'][i] or self.qj[i] > self.config.joint_limits['leg']['q_max'][i]:
            #     print(f"Motor {motor_idx}超出限位！当前角度：{self.qj[i]:.2f} rad")
            #     # 触发保护措施
            #     controller.move_to_default_pos()
            #     controller.default_pos_state()

        self.obs = np.zeros([1, config.num_obs], dtype=np.float32)
        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        eu_ang = quaternion_to_euler_array(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi
        # quat = R.from_euler('xyz', rpy).as_quat()  # 新增：将欧拉角转换为四元数[x,y,z,w]
        # quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # 调整顺序为[w,x,y,z]
        omega = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        flip_mask2policy = [1, 1, -1, 1, 1, -1, 1, -1, -1, -1]
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy() * np.array(flip_mask2policy, dtype=np.float32)
        # create observation
        self.obs[0, 0] = math.sin(2 * math.pi * self.count_lowlevel * self.config.control_dt  / 0.64)
        self.obs[0, 1] = math.cos(2 * math.pi * self.count_lowlevel * self.config.control_dt  / 0.64)
        self.obs[0, 2] = self.remote_controller.lx
        self.obs[0, 3] = self.remote_controller.ly
        self.obs[0, 4] = self.remote_controller.rx
        self.obs[0, 5:15] = (qj_obs -self.default_angle) * np.array(flip_mask2policy, dtype=np.float32)
        self.obs[0, 15:25] = dqj_obs * 0.05
        self.obs[0, 25:35] = self.action
        self.obs[0, 35:38] = omega
        self.obs[0, 38:41] = eu_ang
        
        self.obs = np.clip(self.obs, -18, 18)

        self.hist_obs.append(self.obs)
        self.hist_obs.popleft()

        self.policy_input = np.zeros([1, self.config.num_obs * self.config.frame_stack], dtype=np.float32)
        for i in range(self.config.frame_stack):
            self.policy_input[0, i * self.config.num_obs : (i + 1) * self.config.num_obs] = self.hist_obs[i][0, :]
        self.action[:] = self.policy(torch.tensor(self.policy_input))[0].detach().numpy()
        self.action = np.clip(self.action, -18, 18)

        self.target_q = self.action * self.config.action_scale
        self.target_q = self.target_q * np.array(flip_mask2policy, dtype=np.float32) + self.default_angle
        # print(self.target_q)
        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            clipped_q = np.clip(self.target_q[i], 
                      self.config.joint_limits['leg']['q_min'][i],
                      self.config.joint_limits['leg']['q_max'][i])
            self.low_cmd.motor_cmd[motor_idx].q = clipped_q
            # self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = 0
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"/home/jetson/sdk2_python_DM/example/deploy_lower/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(99, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            print("Press the select key to exit")
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    controller.move_to_default_pos()
    controller.default_pos_state()
    # print("Exit")
