import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"]

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)
            self.frame_stack = 15
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            # 添加关节限位参数
            self.joint_limits = {
                'leg': {
                    'q_min': np.array([-0.9, -0.1, -0.8, -0.1, -0.9, -0.9, -0.785, -0.8, -1.8, -0.25], dtype=np.float32),  # 髋滚/髋摆/膝摆
                    'q_max': np.array([ 0.9, 0.785, 0.8, 1.8, 0.25, 0.9, 0.1, 0.8, 0.1, 0.9], dtype=np.float32)
                },
                'arm_waist': {
                    'q_min': np.array([-0.8, -1.5, -1.4, -1.5, -0.3, -1.1, -0.1, -0.8, -1.5], dtype=np.float32), # 腰部/肩/肘
                    'q_max': np.array([0.8, 1.1, 0.1, 0.8, 1.5, 1.5, 1.4, 1.5, 0.3], dtype=np.float32)
                }
            }

