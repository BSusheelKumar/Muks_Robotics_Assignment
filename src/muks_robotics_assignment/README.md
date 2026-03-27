# Muks Robotics Assignment (ROS 2)

This package contains my full solution for:

1. Path smoothing from sparse waypoints
2. Trajectory generation (time-parameterized path)
3. Trajectory tracking controller
4. Obstacle-aware local replanning using `/scan` (extra task) (I have some little bugs that needs to be fixed)

The implementation is done as a ROS 2 Python package with separate task nodes and shared algorithm modules.

## 2.1 Setup and Execution Instructions

### Environment

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10+

### Build Steps

```bash
# 1) Go to your ROS 2 workspace
cd ~/ros2_ws/src

# 2) Put this folder here and keep the package name as below
#    ~/ros2_ws/src/muks_robotics_assignment

# 3) Build package
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select muks_robotics_assignment

# 4) Source install setup
source install/setup.bash
```

### Run Nodes

```bash
# Task 1: Path smoothing only
ros2 run muks_robotics_assignment task1_path_smoothing_node

# Interactive path smoothing from RViz clicked points
ros2 run muks_robotics_assignment interactive_path_smoothing_node

# Task 2: Trajectory generation
ros2 run muks_robotics_assignment task2_trajectory_generation_node

# Task 3: Trajectory tracking
ros2 run muks_robotics_assignment task3_trajectory_tracking_node


# All-in-one integrated node
ros2 run muks_robotics_assignment path_smoothing_node
```

### Launch Task 1 + Task 2 + Task 3 Together

```bash
ros2 launch muks_robotics_assignment tasks_1_2_3.launch.py
```

Optional launch args:

```bash
ros2 launch muks_robotics_assignment tasks_1_2_3.launch.py frame_id:=odom use_sim_time:=false
```

### Interactive RViz Demo Flow

1. Start node:
   `ros2 run muks_robotics_assignment interactive_path_smoothing_node`
2. Open RViz, Fixed Frame = `odom`.
3. Use **Publish Point** tool and click multiple points.
4. Add `Path` displays for:
   - `/interactive_raw_path`
   - `/interactive_smoothed_path`
5. Add `MarkerArray` display for:
   - `/interactive_path_markers`
6. Clear path anytime:
   `ros2 service call /interactive_path/clear std_srvs/srv/Empty {}`

### RViz2 Quick Setup

1. Set Fixed Frame to `odom`
2. Add `Path` displays:
   - `/raw_path`
   - `/smoothed_path`
   - `/trajectory_path`
   - `/robot_tracked_path`
   - `/avoidance_path` (Task 4 / integrated)
3. Add `RobotModel` (optional)
4. Add `LaserScan` display on `/scan` (Task 4)

### Useful Parameters (examples)

```bash
# Example: enable external odometry and disable local TF publishing
ros2 run muks_robotics_assignment task3_trajectory_tracking_node --ros-args \
  -p use_external_odometry:=true \
  -p publish_tf:=false \
  -p odom_topic:=/odom

# Example: slower motion (keep trajectory speed feasible)
ros2 run muks_robotics_assignment task3_trajectory_tracking_node --ros-args \
  -p max_linear_speed:=0.5 \
  -p constant_speed:=0.35
```

## Testing and Reports

Run all automated tests and generate reports:

```bash
cd src/muks_robotics_assignment
./scripts/run_tests.sh
```

Artifacts generated:

- `test_reports/pytest_output.txt`
- `test_reports/pytest_junit.xml`
- `test_reports/TEST_REPORT.md`

Detailed quality report:

- `REPORT.md`

## 2.2 Design Choices, Algorithms, and Architecture

### Design Approach

The main intention was to keep it modular and evaluation-friendly:

- Task-wise nodes for clean demonstration
- Shared pure-Python algorithm modules for reusability/testing
- ROS parameters instead of hardcoded constants
- Clear topic-level observability in RViz

### Algorithms Used

#### Task 1: Path Smoothing

- **Centripetal Catmull-Rom spline** interpolation
- Reason for choosing: gives smooth and continuous path while reducing overshoot around sharp turns compared to uniform parameterization
- Output: dense geometric path from sparse waypoints

#### Task 2: Trajectory Generation

- Path is resampled at nearly uniform arc-length intervals
- Time stamps are assigned using:
  - constant speed profile, or
  - trapezoidal/triangular profile (based on `max_speed`, `max_accel`)
- Output: `(x, y, t)` trajectory samples

#### Task 3: Tracking Controller

- Unicycle/differential-drive tracking law with feedforward + feedback
- Errors are computed in robot frame (`e_x`, `e_y`, `e_theta`)
- Control outputs:
  - linear velocity `v_cmd`
  - angular velocity `w_cmd`
- Added feasibility handling so reference timing does not exceed tracking speed limits
- Added TF and joint-state publication for consistent RobotModel visualization in RViz

#### Extra : Obstacle Avoidance ( Have some little Bugs that need to be fixed)

- LaserScan points are projected into world frame
- A local lookahead window checks whether active trajectory is blocked
- Candidate detours are generated via lateral offsets around the nearest obstacle
- Candidates are smoothed and collision-checked
- Best candidate is selected and converted into a new timed trajectory

### Architecture

Package layout:

- `muks_robotics_assignment/task1_path_smoothing_node.py`
- `muks_robotics_assignment/task2_trajectory_generation_node.py`
- `muks_robotics_assignment/task3_trajectory_tracking_node.py`
- `muks_robotics_assignment/task4_obstacle_avoidance_node.py`
- `muks_robotics_assignment/path_smoothing_node.py` (integrated)
- `muks_robotics_assignment/path_smoothing.py`
- `muks_robotics_assignment/trajectory_generation.py`
- `muks_robotics_assignment/trajectory_tracking.py`
- `muks_robotics_assignment/obstacle_avoidance.py`
- `muks_robotics_assignment/geometry_utils.py`
- `muks_robotics_assignment/ros_message_utils.py`
- `muks_robotics_assignment/common_types.py`
- `muks_robotics_assignment/waypoint_utils.py`

Also see `ARCHITECTURE.md` for a compact architecture summary.

## 2.3 How This Extends to a Real Robot

For a real robot deployment, the same architecture can be used with a few practical upgrades:

### 1) Use Real State Estimation as Source of Truth

- Set `use_external_odometry:=true`
- Subscribe to robot odometry/filtered localization topic
- Normally set `publish_tf:=false` if odom TF is already published by robot stack

### 2) Control Interface and Safety Layer

- Keep output on `/cmd_vel` but add:
  - command timeout watchdog
  - acceleration/jerk limits
  - emergency stop integration
- Use conservative speed caps initially and tune gradually

### 3) Better Obstacle Avoidance for Field Conditions

- Current method is local and reactive; good for assignment/demo
- For production use:
  - integrate costmaps and robot footprint model
  - include dynamic obstacle prediction
  - add recovery behaviors if local planner repeatedly fails

### 4) Robustness and Calibration

- Calibrate wheel radius and wheelbase (`wheel_separation`)
- Use IMU + wheel odometry fusion (e.g., EKF)
- Validate TF tree and frame conventions before controller tuning

### 5) Deployment Workflow

- Start in simulation
- Dry run on robot at low speed
- Log trajectories/errors
- Tune gains and speed profiles incrementally

## Troubleshooting Notes

- If `RobotModel` and tracked path do not match:
  - verify fixed frame (`odom`)
  - verify TF publisher ownership (avoid duplicate `odom -> base_*`)
- If wheel links are missing in RViz:
  - check wheel joint names in `wheel_joint_names` parameter
  - check `/joint_states` publishing
