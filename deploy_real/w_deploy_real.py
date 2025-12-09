from typing import Union
import numpy as np
import time
import torch
import math
import os
from datetime import datetime
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from w_config import Config
from scipy.spatial.transform import Rotation as R
from collections import deque

"""
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])
"""

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize data logger for sim2sim debugging
        self.log_data = {
            'timestamp': [],
            'observations': [],
            'policy_input': [],
            'actions': [],
            'target_q': [],
            'qj': [],
            'dqj': [],
            'imu_quat': [],
            'imu_euler': [],
            'imu_gyro': [],
            'commands': [],  # lx, ly, rx
        }
        self.logging_enabled = True
        self.log_dir = "/home/jetson/sdk2_python_DM/example/deploy_lower/deploy_real/logs"
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros((config.num_actions), dtype=np.float32)
        self.target_dof_pos = np.zeros((config.num_actions), dtype=np.double)

        self.count_lowlevel = 0

        self.hist_obs = deque()
        for _ in range(config.frame_stack):
            self.hist_obs.append(np.zeros([1, config.num_obs], dtype=np.double))

        self.default_angle =np.zeros((config.num_actions),dtype=np.double)

        # read and set default angles from config
        for i in range(config.num_actions):
            self.default_angle[i] = config.default_angles[i]

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_() 
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # Wait for the subscriber to receive data
        # self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_ctrl)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_ctrl)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def save_log(self, suffix=""):
        """Save logged data to npz file for sim2sim debugging"""
        if not self.log_data['timestamp']:
            print("No data to save.")
            return
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deploy_log_{timestamp_str}{suffix}.npz"
        filepath = os.path.join(self.log_dir, filename)
        
        # Convert lists to numpy arrays
        save_data = {
            'timestamp': np.array(self.log_data['timestamp']),
            'observations': np.array(self.log_data['observations']),
            'policy_input': np.array(self.log_data['policy_input']),
            'actions': np.array(self.log_data['actions']),
            'target_q': np.array(self.log_data['target_q']),
            'qj': np.array(self.log_data['qj']),
            'dqj': np.array(self.log_data['dqj']),
            'imu_quat': np.array(self.log_data['imu_quat']),
            'imu_euler': np.array(self.log_data['imu_euler']),
            'imu_gyro': np.array(self.log_data['imu_gyro']),
            'commands': np.array(self.log_data['commands']),
            'default_angles': self.default_angle,
            'control_dt': self.config.control_dt,
            'action_scale': self.config.action_scale,
            'num_obs': self.config.num_obs,
            'frame_stack': self.config.frame_stack,
        }
        
        np.savez(filepath, **save_data)
        print(f"Log saved to: {filepath}")
        print(f"Total frames logged: {len(self.log_data['timestamp'])}")
        return filepath

    def clear_log(self):
        """Clear logged data"""
        for key in self.log_data:
            self.log_data[key] = []
        print("Log data cleared.")

    def zero_torque_state(self):
        """Enter the zero torque state, wait for the start signal"""
        
        print("Enter zero torque state.")
        print("Waiting for the Start signal...")
        while self.remote_controller.button[KeyMap.start] != 1: # Wait for the Start button
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 4.0
        max_step=0.05
        # 组合索引 & 目标
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        # flip_mask2policy = [1, 1, -1, 1, 1, -1, 1, -1, -1, -1]
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

        self.obs = np.zeros([1, config.num_obs], dtype=np.float32)
        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        r = R.from_quat(quat) 
        eu_ang = r.as_euler('xyz', degrees=False)
        print("imu_state old:", eu_ang)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        # quat = R.from_euler('xyz', rpy).as_quat()  # 新增：将欧拉角转换为四元数[x,y,z,w]
        # quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # 调整顺序为[w,x,y,z]
        omega = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        flip_mask2policy = np.array([1, 1, -1,  1,  1, 
                           -1, 1, -1, -1, -1], dtype=np.float32)
         # NOTE : Right leg joints need to be flipped
        
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy() * flip_mask2policy
        # create observation
        phase = 2 * math.pi * self.count_lowlevel * self.config.control_dt / 0.64
        self.obs[0, 0] = math.sin(phase)
        self.obs[0, 1] = math.cos(phase)
        self.obs[0, 2] = self.remote_controller.lx * self.config.lin_vel_scale
        self.obs[0, 3] = self.remote_controller.ly * self.config.lin_vel_scale
        self.obs[0, 4] = self.remote_controller.rx * self.config.ang_vel_scale
        # joint positions and velocities
        self.obs[0, 5:15] = (qj_obs - self.default_angle) * flip_mask2policy * self.config.dof_pos_scale
        self.obs[0, 15:25] = dqj_obs * self.config.dof_vel_scale * self.config.dof_vel_scale
        self.obs[0, 25:35] = self.action
        self.obs[0, 35:38] = omega
        self.obs[0, 38:41] = eu_ang
        
        self.obs = np.clip(self.obs, -self.config.clip_observations, self.config.clip_observations)

        self.hist_obs.append(self.obs)
        self.hist_obs.popleft()

        self.policy_input = np.zeros([1, self.config.num_obs * self.config.frame_stack], dtype=np.float32)
        for i in range(self.config.frame_stack):
            self.policy_input[0, i * self.config.num_obs : (i + 1) * self.config.num_obs] = self.hist_obs[i][0, :]

        self.action[:] = self.policy(torch.tensor(self.policy_input))[0].detach().numpy()
        self.action = np.clip(self.action, -self.config.clip_actions, self.config.clip_actions)

        self.target_q = self.action * self.config.action_scale
        self.target_q = self.target_q * flip_mask2policy + self.default_angle
        
        # lower body
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

        # upper body
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = 0
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)
        
        # Log data for sim2sim debugging
        if self.logging_enabled:
            self.log_data['timestamp'].append(time.time())
            self.log_data['observations'].append(self.obs.copy())
            self.log_data['policy_input'].append(self.policy_input.copy())
            self.log_data['actions'].append(self.action.copy())
            self.log_data['target_q'].append(self.target_q.copy())
            self.log_data['qj'].append(self.qj.copy())
            self.log_data['dqj'].append(self.dqj.copy())
            self.log_data['imu_quat'].append(np.array(self.low_state.imu_state.quaternion))
            self.log_data['imu_euler'].append(eu_ang.copy())
            self.log_data['imu_gyro'].append(omega.copy())
            self.log_data['commands'].append(np.array([self.remote_controller.lx, 
                                                        self.remote_controller.ly, 
                                                        self.remote_controller.rx]))

        time.sleep(self.config.control_dt)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface", default="wlP1p1s0")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="c")
    parser.add_argument("--log", action="store_true", help="enable data logging")
    args = parser.parse_args()

    # Load config
    config_path = f"/home/jetson/sdk2_python_DM/example/deploy_lower/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication 
    print("Initialize DDS communication")
    ChannelFactoryInitialize(99, args.net)

    print("Initialize controller")
    controller = Controller(config)
    
    # Enter the zero torque state, press the start key to continue executing
    print("Enter the zero torque state, press the start key to continue executing")
    controller.zero_torque_state()

    # Move to the default position
    print("Move to the default position")
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    print("Starting main control loop. Press Ctrl+C to exit and save log.")
    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("Select key pressed, exiting...")
                break
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, saving log...")
            break
    
    # Save the logged data
    if args.log:
        controller.save_log()
    
    # Enter the damping state
    controller.move_to_default_pos()
    controller.default_pos_state()
    print("Exit")
