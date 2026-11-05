# Simple single-camera visual servoing

This folder is a self-contained PyBullet example using the thesis robot:
the Summit mobile platform, seven-joint WAM arm, and Barrett hand. One virtual
RGB camera is rigidly mounted on the wrist. The controller moves the base and
arm to keep the **moving blue block's image centroid near the camera frame centre**.
By default the block repeats a smooth figure-eight indefinitely and the robot
keeps tracking until you quit.

Only Python, NumPy, OpenCV, and PyBullet are needed. All robot meshes and scene
URDFs are local to this folder. You can copy the entire folder elsewhere and
run it without the thesis directory, ROS, MATLAB, HQP, or any optimisation package.

## Run

Use Python 3.10 or newer. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python simple_visual_servo.py
```

Two windows open: the PyBullet scene and the wrist camera image. The scene shows
the entire mobile manipulator and moving block against a fixed floor grid;
an orange trail shows base translation and a green trail shows wrist movement.
The trails fade after eight seconds. In the wrist image, the green
cross marks the desired image centre; the yellow circle marks the detected
blue centroid. Press **Q** or **Esc** in the camera window, close a window,
or press **Ctrl+C** in the terminal to stop. A desktop display and the regular
`opencv-python` package are required for the windows.

For a bounded run or the original stationary-block demonstration:

```bash
python simple_visual_servo.py --duration 15
python simple_visual_servo.py --static-target
```

For a run without display windows, with optional error logging and a final image:

```bash
python simple_visual_servo.py --headless --duration 10 --log results/error.csv --save-frame results/final_camera.png
```

The headless mode runs as fast as possible; `--duration` is **simulated** time.
Without a duration (or with `--duration 0`), both modes run indefinitely until
quit. `--moving-target` remains available as an explicit flag, but moving is
already the default.
Paths for assets are resolved relative to the script, so launching the script
from a different working directory also works.

## Feedback loop

Everything is in `simple_visual_servo.py`:

1. Render a 640 × 480 RGB image from the wrist camera.
2. Threshold blue pixels in HSV colour space and find the largest region's centroid.
3. Compute the pixel error relative to `(320, 240)` and normalise it by the focal length.
4. Multiply the standard point-feature interaction matrix `L` by the robot's
   camera Jacobian `J_camera`, giving `J_image = L @ J_camera`.
5. Apply proportional feedback through a damped pseudoinverse. With diagonal
   mobility scaling `S`, use `A = J_image @ S`, then
   `q_dot = -gain * S @ A.T @ solve(A @ A.T + damping² * I, error)`.
   Add a small posture correction projected into the remaining redundant motion
   to resist gradual drift away from the starting configuration.
6. Limit the speeds and integrate the base and arm positions at 30 Hz; repeat.

The ten velocity commands are world-frame base `vx`, `vy`, yaw rate, and WAM
joint rates `J1` through `J7`. The geometric Jacobian includes the camera's
offset from the wrist. PyBullet's `calculateJacobian` supplies the arm columns;
the script adds the three ideal planar base columns and rotates the result into
camera coordinates. Joint names and limits come from the URDF.
The fixed mobility scales give base translation a larger share of the image
correction, so the platform travels visibly while the arm follows the block's
vertical motion. This is still a single feedback calculation using NumPy.
The posture correction uses the robot's own base and joint positions; target
feedback still comes solely from the RGB image.
The fingers stay open. The camera orientation stays attached to the wrist;
the controller does not point the rendered camera directly at the target.

The depth in `L` is a constant nominal estimate of 2 m. Feedback uses only
the RGB image: it does not read the target's world pose, simulator segmentation,
or depth image. The focal length is computed from PyBullet's vertical FOV.
The robot stops within a 2-pixel tolerance and stops immediately when the blue
block is not detected. A lost block must re-enter view before tracking resumes.

This example centres the target; it does not control target distance or full
camera pose. The block's scene animation is separate from the controller:
it follows a bounded, repeating 24-second path with excursions of ±0.35 m
in x, ±0.70 m in y, and ±0.22 m in z. Its motion never stops at the end of a
cycle. Proportional feedback leaves some tracking lag while the block moves.

## Settings to edit

The constants near the top of the Python file set the target position, camera
offset, motion amplitudes and period, gain, damping, mobility scaling, nominal
depth, posture gain, speed limits, and centring tolerance.
The starting pose matches the thesis scripts: base height 0.13 m, `J4 = 1.572`
rad, other arm joints zero. The default block position is `(2.6, 0.5, 1.0)` m.
CSV output includes the base x/y/yaw and all seven arm joint positions as well
as the image errors, so both motions can be measured from a run.

The base and arm use **ideal kinematic velocity integration**, with zero gravity
and a fixed-base URDF whose planar pose is updated explicitly. This keeps the
example focused on visual feedback. It does not simulate wheel traction, motor
torques, or contact response as a robot controller. The script clamps arm joint
positions to their limits; it has no collision avoidance or target search.

## Included files

```text
simple_visual_servo.py              Complete simulation and feedback controller
requirements.txt                    Three Python dependencies
assets/waterloo_steel_full.urdf      Full mobile manipulator, with relative mesh paths
assets/blue_block.urdf               Blue 15 cm target cube
assets/ground.urdf                   Grey floor
assets/meshes/                       All 41 referenced robot mesh files
backups/                            Original URDF and working-file backup copies
```

The robot URDF and meshes were copied from `waterloo_steel_full.urdf` and
`waterloo_steel_description/meshes/` in the thesis folder. Only the mesh filename
references in the working robot URDF were changed, from absolute paths to local
relative paths. No existing thesis files were modified. See `backups/README.md`
for the backup details.

## Validation

Tested headlessly with Python 3.10.12, PyBullet 3.2.7, NumPy 2.2.6, and
OpenCV 4.12.0.88. The original stationary-default version is preserved in
`backups/simple_visual_servo.stationary-default.py.bak`.

The current default completed a full 24-second cycle using rendered 640 × 480
RGB images with zero lost detections across 721 frames. The base travelled
through ranges of 0.19 m in x and 0.65 m in y; several arm joints moved through
approximately 5–6 degrees. After the first four seconds the tracking error
stayed below 24 pixels. A separate 12-minute kinematic check using projected
image points showed repeatable bounded motion with the posture correction;
that longer check did not exercise image rendering or detection.

The complete folder also ran after being copied into `/tmp`, from a different
working directory. All 41 mesh references were checked. The image Jacobian
was checked against finite differences at a rotated base and nonzero arm pose,
and centred-target / lost-target feedback was checked to return zero velocity.
Interactive GUI windows were not exercised during these headless checks.
