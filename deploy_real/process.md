
- set motor to zero positions

```bash
source /home/jetson/unitree_ws/src/unitree_ros2/setup.sh
export CYCLONEDDS_HOME="/home/jetson/cyclonedds/install" 
ros2 topic pub /lowcmd unitree_hg/msg/LowCmd '{set_zero: false}' --once
ros2 topic pub /lowcmd unitree_hg/msg/LowCmd '{set_zero: true}' --once
```

- building necessary packages

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select unitree_hg
colcon build --packages-select imu_package
colcon build --packages-select wheeltec_joy
colcon build --packages-select motor_package
```

- initiating for deployment

```bash
source /home/jetson/ros2_driver/install/setup.bash
ros2 launch wheeltec_joy wheeltec_wireless_joy.launch.py    # wireless joystick
ros2 launch imu_package imu_driver_launch.py         # imu driver
ros2 launch motor_package motor_driver_launch.py     # motor driver
```
- deploying policy 

```bash
cd /home/jetson/sdk2_python_DM/example/deploy_lower/deploy_real
python w_deploy_real.py wlP1p1s0 w_g1.yaml
```

- Testing motors position (might need to set zero again, optional)

```bash 
cd /home/jetson/sdk2_python_DM/example/g1/low_level
python g1_low_level_example.py
```