# Camera cart visual servo

```sh
conda run --no-capture-output -n robotics python examples/visual_servo.py
```

A small procedural IBVS demo: one `main()` and one `interaction_matrix()` function.
All PyBullet calls use `p` directly. Parameters are constants at the top of
`examples/visual_servo.py`; there are no scene classes or project-module dependencies.

Click the **PyBullet window** for keyboard control. The camera window accepts
Escape only. Camera capture and control use 480 × 360 images at 30 Hz; native
buffer previews are disabled. A slow machine runs simulation time more slowly
instead of increasing the integration step.

| Key | Action |
| --- | --- |
| W / S or up / down | Move board along world +X / -X |
| A / D or left / right | Move board along world +Y / -Y |
| Q / E | Rotate board about world Z |
| Space | Pause / resume tracking |
| R | Reset cart and board |
| Escape | Exit |

Uppercase and lowercase keys work. Space and R act once per press. Keep the
printed face toward the camera; losing the marker stops the cart immediately.
Bringing it back resumes tracking. Closing either window exits.

## Read the code in order

The numbered blocks in `main()` follow this sequence:

1. Set camera intrinsics, AprilTag detector and desired image corners.
2. Create the cart and marker board with `p`.
3. Read keyboard input to move the board.
4. Render the camera, detect four corners, and estimate their depths with PnP.
5. Compute image error and solve for two wheel rates using IBVS.
6. Integrate wheel odometry to move the cart.
7. Show the image and wait until the next frame deadline.

Only rendered pixels enter the visual controller. Board world coordinates are
used to place the board, never to calculate wheel commands.

## The three matrices

`J_WHEELS` maps left/right wheel angular rates to forward speed and yaw rate:

```text
[v, omega] = J_WHEELS @ [left_rate, right_rate]
J_WHEELS = radius * [[1/2,       1/2],
                    [-1/track, 1/track]]                     (2×2)
```

`J_CAMERA` maps that axle motion to camera velocity. Optical axes are right,
down, forward; the cart's forward direction is optical +Z. Turning left gives
negative optical Y rotation and, because the camera is ahead of the axle,
negative optical X translation:

```text
camera_twist = J_CAMERA @ [v, omega]
J_CAMERA = [[0, -offset],
            [0,       0],
            [1,       0],
            [0,       0],
            [0,      -1],
            [0,       0]]                                    (6×2)
```

`L`, the Interaction matrix, maps camera velocity to image corner velocity.
For each normalized corner `(x, y)` at depth `Z`, it contributes two rows:

```text
[-1/Z,    0, x/Z,    xy, -(1+x²),  y]
[   0, -1/Z, y/Z, 1+y²,     -xy, -x]
```

Stack four corners to get `L` (8×6). Combine the three matrices in the order
that wheel motion reaches the image:

```text
J = L @ J_CAMERA @ J_WHEELS                    # 8×2
image_velocity = J @ wheel_rates
error = (measured_corners - desired_corners).ravel()
wheel_rates = -GAIN * solve(J.T @ J + DAMPING² * I, J.T @ error)
```

This damped least-squares solution asks the image error to shrink. There are
eight image coordinates but only two independent wheel velocities; when the
board tilts, all four corners need not reach their desired positions. This is
why a residual image error may remain even when the cart has settled.

Desired corners describe a frontal 0.40 m marker at `DISTANCE = 1.6 m`. This is
an image target, not a guarantee of exact physical distance for a tilted board.
Commands are scaled together to respect speed limits; a small image-error
threshold suppresses jitter. Pausing or invalid detection sets both rates to zero.

The cart uses ideal differential-drive odometry, without tire dynamics or
collision response. Generated marker texture and mesh are removed on exit.
The demo stays single-process and does not use the full project's display queues.

## Headless run

```sh
conda run --no-capture-output -n robotics python examples/visual_servo.py --headless --seconds 6
```

Optional `--snapshot /tmp/cart-camera.png` saves the final annotated image.
Requires NumPy, PyBullet, and OpenCV with `cv2.aruco.ArucoDetector` (and a GUI
backend for interactive use).
