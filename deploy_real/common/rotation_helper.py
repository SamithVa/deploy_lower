import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix() #使用R.from_euler("z", waist_yaw)创建腰部Yaw关节的Z轴旋转矩阵RzWaist
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]])#将IMU四元数imu_quat转换为旋转矩阵R_torso（注意四元数顺序调整为[x,y,z,w]）
    R_pelvis = np.dot(R_torso, RzWaist.T)#通过矩阵乘法将IMU坐标系（躯干姿态）转换到骨盆坐标系
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])#消除腰部Yaw关节角速度对IMU角速度测量的影响，IMU通常安装在躯干上，测量的是躯干的角速度。但当腰部Yaw关节旋转时，躯干相对于骨盆的运动会引入额外的角速度。如果不补偿，IMU的数据会包含腰部关节的运动，导致姿态估计错误。
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w
