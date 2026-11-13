# Gripper for Shipping

Gripper for Shipping explores how a robot can receive cargo from a crane. Current development focuses on [visual servo tracking](examples/visual_servo/visual_servo_tracking.py). A fixed-base Franka Panda uses its wrist camera to follow an AprilTag on a swinging load. The experiment measures how the camera and control update rate affects tracking when the trolley reverses or brakes.

## Installation and Quick Start

Run these commands from the repository root.

```sh
conda create -n robotics python=3.12
conda activate robotics
pip install -e .
python examples/visual_servo/visual_servo_tracking.py
```

Click the PyBullet window and hold **A** or **D** to move the trolley. Release to brake and watch the wrist camera track the swinging tag. Press **Esc** to exit.

### Tracking examples

The camera and arm controller update at 15 Hz by default. Use `--fps` to change both rates.

```sh
# Track interactively at 30 Hz
python examples/visual_servo/visual_servo_tracking.py --fps 30

# Record a manual trolley run at 15 Hz
python examples/visual_servo/visual_servo_tracking.py --fps 15 --record

# Run without windows for five simulated seconds
python examples/visual_servo/visual_servo_tracking.py --headless --seconds 5
```

When recording, use **A/D** to drive along X and **Esc** to save. The 15 Hz run writes `tracking_15Hz.csv` and `final_camera_15Hz.png` to `examples/visual_servo/results/`, replacing any previous recording at that rate. See the [experiment report](examples/visual_servo/visual_servo_tracking.md) for the comparison across 7.5, 15, 30, and 60 Hz.

## Packages and tools

The simulation uses Python 3.12. Dependencies are listed in [pyproject.toml](pyproject.toml).

| Package or tool | Role in the project |
| --- | --- |
| PyBullet | Loads URDF models, simulates the crane and arm, and renders the wrist camera view. |
| OpenCV | Detects the AprilTag and estimates its pose from the camera image. |
| NumPy and SciPy | Compute joint velocities and transform poses between camera and robot coordinates. |
| Matplotlib | Plots crane and cart velocities and image tracking error during a run. |
| pytest | Runs the simulation and example tests in `tests/`. |

## Project structure

```text
main.py          Basic PyBullet demo of a cube falling onto a plane
src/             Simulation scene, crane controls, camera tracking, and arm control
examples/        Standalone experiments in grasping, visual servoing, and other topics
assets/          Robot models, meshes, and AprilTag resources
tests/           Tests for the simulation and examples
typings/         Python type stubs for third-party packages
pyproject.toml   Python dependencies and development tool settings
```
