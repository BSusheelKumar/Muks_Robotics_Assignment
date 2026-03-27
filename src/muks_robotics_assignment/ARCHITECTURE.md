# Architecture and Design Choices

The goal of this package is to keep the code modular, easy to understand, and testable. 

## Design Choices

- **Separation of Concerns:** The ROS 2 specifics (like nodes, publishers, and subscribers) are kept in their own files, completely separate from the core math and algorithms. This makes it much easier to write unit tests for the logic without needing a running ROS environment.
- **Node per Task:** Each task (smoothing, generation, tracking, avoidance) has its own dedicated node. This helps a lot when debugging because you can run one piece at a time and visualize its specific output in RViz.
- **Configurable Parameters:** Hardcoding things like speeds, wheel sizes, or controller gains is usually a bad idea. We exposed these as ROS parameters so they can be easily tuned during runtime or via launch files.

## Algorithms Used

- **Path Smoothing:** We used Catmull-Rom splines instead of standard Bézier curves. Catmull-Rom guarantees that the smoothed path actually passes directly through the original waypoints, which is important if those waypoints represent tight doorways or critical checkpoints.
- **Trajectory Generation:** The code samples the spline at uniform distances and then assigns timestamps. This allows us to apply different velocity profiles (like moving at a constant speed or accelerating smoothly).
- **Trajectory Tracking:** We use a kinematic controller based on a unicycle model. It constantly measures the distance and angle error between the robot's current pose and the target reference point, outputting linear (`v`) and angular (`w`) velocities to correct the error.
- **Obstacle Avoidance:** We take the 2D laser scan data and project it onto our map. If any scan points fall directly on our upcoming trajectory, we consider it blocked and automatically generate a local detour to get around the obstacle.
