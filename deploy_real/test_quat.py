from scipy.spatial.transform import Rotation as R
import numpy as np

def quaternion_to_euler_array(quat):
    """

    Output : rollx, pitch_y, yaw_z in radians
    """
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

if __name__=="__main__":
    # test the function
    quat = np.array([0.0, 0.0, 0.7071, 0.7071], dtype=np.float32)  # Example quaternion
    euler_angles = quaternion_to_euler_array(quat)
    print("Euler Angles (radians):", euler_angles)
    print("Euler Angles (degrees):", np.degrees(euler_angles))
    # Verify with scipy
    r = R.from_quat(quat)
    euler_scipy = r.as_euler('xyz', degrees=False)
    print("Scipy Euler Angles (radians):", euler_scipy)
    print("Scipy Euler Angles (degrees):", np.degrees(euler_scipy))

    print("Test passed:", np.allclose(euler_angles, euler_scipy))
