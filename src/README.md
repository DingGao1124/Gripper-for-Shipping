# Gripper for Shipping

PyBullet simulation of a manually operated gantry crane and a mobile Franka Panda. A wrist camera detects an AprilTag on the swinging load. The cart follows its overall motion, the arm performs image-based visual servoing (IBVS), and a magnetic tool takes over inside the green handoff region.

## Run

Run the commands below from the repository root.

```sh
conda run --no-capture-output -n robotics python main.py
```

Use Python 3.12 with the dependencies in `pyproject.toml`. OpenCV must provide `cv2.aruco.ArucoDetector`; the current project dependency includes it. Panda meshes come from `pybullet_data`.

For a short run without windows:

```sh
conda run --no-capture-output -n robotics python main.py --headless --seconds 3
```

## Controls and workflow

Click the PyBullet window before using the keyboard.

Hold a motion key for continuous travel; release it to brake. Press H/F/R/C once for their actions. Input heartbeats use wall-clock time: if the viewer stops responding for 0.35 s, the crane brakes rather than retaining an old held key.

| Key | Action |
| --- | --- |
| W / S, up / down | Gantry Y |
| A / D, left / right | Trolley X |
| Q / E | Raise / lower cable |
| F | Attach the hook tip near the load's top face |
| R | Release hook |
| H | Show / hide the native panel (starts hidden; no sliders) |
| C | Save wrist-camera RGB and calibration to `tmp/captures` |
| Escape in camera window | Exit |

1. Position the hook over the load's top face, lower the cable to about **1.07 m**, then press **F**. The tip may be up to **3.5 cm above** the face or **1 cm below** it; exact center alignment is unnecessary. The hook has no collision shape, so stop lowering when its visible tip reaches the surface. A rejected attachment prints the height gap and distance outside the top edge.
2. Raise the load; cable length about **0.55 m** puts its center near **0.63 m** above ground.
3. Move toward **X = 0.25 m, Y = 0.25 m**. The cart and arm follow automatically.
4. Stop inside the green region and wait for `SETTLE`, then `APPROACH`.
5. When distance, alignment and relative speed permit, the tool locks, the hook releases and the state becomes `HOLD`.

The crane starts idle and uses keyboard controls only. The native panel starts hidden and H toggles it; sliders and native buffer previews stay disabled. It never autonomously transports the load. The camera window shows real detections and tracking status; Matplotlib shows measured rail velocities, cart speed and image error.

## Implementation

Parameters are centralized in `src/config.py`. `scene`, `crane`, `robot`, `vision`, `servo`, and `task` handle their respective functions; `app` owns the clock and `display` owns the windows. `main.py` is the entrypoint.

The camera uses OpenCV axes (+X right, +Y down, +Z forward). The interaction matrix maps camera twist to normalized corner motion. Its combination with the camera Jacobian yields Panda joint velocities, including cart-motion compensation and visual target-motion feedforward. PnP and robot proprioception provide the world-frame motion estimate used for cart following and settling.

The camera is mounted at the middle of the hand plate's +X side, beside the magnetic tool. A small pad joins the housing directly to the plate. `camera_mount` and `camera_rpy` in `src/config.py` define its optical origin and tilt in the hand frame. Rendering, the camera Jacobian and the desired docking features all use this mounting transform; approach still follows the load's attachment-face normal.

The cart stays 1.00 m on the load's +Y side, while the camera's nominal observation depth is 0.35 m. This separates the cart from the load and extends the arm beyond the chassis. `base_offset`, `observation_distance` and `initial_joints` configure the spacing and initial IK posture. Only the bolted root is exempt from cart collision; moving links retain collisions and the controller pauses on insufficient arm/cart clearance.

The cart is kinematic: its pose is integrated with speed and acceleration limits. Wheel visuals do not model traction. The suspended load uses two passive swing joints and an inextensible taut-cable approximation. Magnetic locking is a proximity-gated fixed constraint, not an electromagnetic or friction model. Object ground truth is used only for the final physical attachment gate, never to replace marker detection.

Lost detections stop approach and brake the cart. New crane motion cancels approach. Clearance and joint-tracking checks pause automatic motion. Close the camera or simulation window to exit.

The existing standalone crane is available through `python examples/crane.py`; the original xArm demonstration is `python examples/xarm_pick_place.py`.

## Refresh and performance

Physics stays at **240 Hz**, robot control at **60 Hz**, and 640 x 480 camera capture plus AprilTag detection at **30 Hz** in simulation time. No physics, sensor or control steps are dropped. Absolute frame deadlines allow a light frame to catch up after camera capture.

Physics, camera capture and control run in the original DIRECT world. A dedicated viewer process mirrors body poses and joint positions into the PyBullet GUI and forwards held-key snapshots with timestamps. OpenCV and Matplotlib run in another process so their native event loops cannot consume the viewer's keyboard events. The viewer never steps physics or supplies measurements to the controller. The camera uses TinyRenderer on the actual physics world with shadows disabled. NumPy-disabled Bullet builds use a byte-buffer conversion for identical RGB pixels.

Camera display updates only for new frames. Both display processes use bounded latest-frame queues; slow windows may drop display snapshots, while every sensor/control update still runs. Plot samples remain at 60 Hz, plots refresh at 10 Hz, and fixed relative time axes plus blitting avoid repeated full redraws. Status text changes only when needed; cable visuals use batched joint updates instead of blocking debug-line replacements. Close the camera window or press Escape to stop; close the plot independently to keep running.

The rates above use simulation time. If rendering exceeds the available CPU budget, wall-clock simulation runs slower instead of skipping physics or visual measurements.
