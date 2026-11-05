# Gripper for Shipping

Gripper for Shipping explores how a robot can receive cargo from a crane. The project is currently at the PyBullet learning stage, with a cargo-handling simulation in `src/` and smaller experiments in `examples/`.

In the main simulation, the operator uses the keyboard to lift and move a load with a gantry crane. A mobile Franka Panda follows the load. Its wrist camera tracks an AprilTag on the load and guides the arm into position for a magnetic handoff. See the [simulation guide](src/README.md) for run commands and controls.

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
main.py          Entry point for the crane and Panda simulation
src/             Simulation scene, crane controls, camera tracking, and arm control
examples/        Standalone experiments in grasping, visual servoing, and other topics
assets/          Robot models, meshes, and AprilTag resources
tests/           Tests for the simulation and examples
typings/         Python type stubs for third-party packages
pyproject.toml   Python dependencies and development tool settings
```
