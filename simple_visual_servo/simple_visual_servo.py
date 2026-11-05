#!/usr/bin/env python3
"""Learn PyBullet and image-based visual servoing with a wrist camera.

Run ``python simple_visual_servo.py`` to open the scene and camera windows;
press Q or Esc in the camera window to quit. Add ``--static-target`` to watch
the robot centre a stationary block, or ``--headless --duration 10`` to run
ten simulated seconds without windows. See README.md for installation.

PyBullet loads the robot and scene from URDF files, provides robot link poses,
and renders the camera image. OpenCV finds the blue block in that image.
The visual servo repeatedly measures the block's offset from the image centre
and converts that error into motion of the Summit base and seven WAM joints.
Only RGB pixels provide target feedback; the controller does not read the
target's world pose, depth image, or simulator segmentation.

This example teaches ideal kinematic control: commanded velocities are
integrated into poses directly, without wheel dynamics or motor torques.
Centring one image point controls neither target distance nor full camera pose.
"""

import argparse
import csv
from pathlib import Path
import time

import cv2
import numpy as np
import pybullet as p


# Resolve assets beside this script, independent of the working directory.
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

# Camera calibration: square pixels, vertical FOV in degrees, focal length in
# pixels. The desired image feature is the centre of the 640 x 480 frame.
WIDTH, HEIGHT = 640, 480
DT = 1.0 / 30.0  # Simulated seconds per control update (30 Hz).
FOV = 80.0  # PyBullet uses a vertical field of view.
FOCAL = HEIGHT / (2.0 * np.tan(np.deg2rad(FOV / 2.0)))
CENTRE = np.array([WIDTH / 2.0, HEIGHT / 2.0])
GAIN = 1.2  # Proportional gain: larger values request faster error correction.
DAMPING = 0.05  # Regularise the inverse when image motion is poorly conditioned.
DEPTH_ESTIMATE = 2.0  # Constant nominal depth, NOT measured simulator depth.
PIXEL_TOLERANCE = 2.0
# These values create/animate the scene; they are not target measurements used
# by the visual servo. Positions and offsets are in metres, angles in radians.
TARGET_POSITION = np.array([2.6, 0.5, 1.0])
TARGET_MOTION_PERIOD = 24.0  # Seconds per repeating, bounded figure-eight.
TARGET_MOTION_AMPLITUDE = np.array([0.35, 0.70, 0.22])  # Metres, x/y/z.
# Rigid camera mounting: translation in the wrist frame and a 90-degree roll.
CAMERA_OFFSET = np.array([-0.06, 0.0, 0.115])
# Optical axes: x right, y down, z forward, expressed in the wrist frame.
OPTICAL_ROTATION = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
# All ten-element vectors use [base x, base y, base yaw, J1, ..., J7].
# Corresponding rate units are [m/s, m/s, rad/s, rad/s, ..., rad/s].
SPEED_LIMITS = np.array([0.15, 0.15, 0.25] + [0.4] * 7)
# Give planar translation a larger share so both the base and arm visibly move.
MOBILITY = np.array([4.0, 4.0, 0.6] + [1.0] * 7)
POSTURE_GAIN = 0.15  # Gently resist drift during repeated target cycles.
# Start with J4 bent; the posture bias gently attracts the robot to this pose.
NEUTRAL_CONFIGURATION = np.array([0., 0., 0., 0., 0., 0., 1.572, 0., 0., 0.])


def rotation(quaternion):
    """Convert a PyBullet quaternion (x, y, z, w) into a 3 x 3 rotation matrix."""
    return np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)


def moving_target_position(sim_time):
    """Repeat a smooth figure-eight forever without drifting out of the scene."""
    phase = 2.0 * np.pi * (sim_time % TARGET_MOTION_PERIOD) / TARGET_MOTION_PERIOD
    return TARGET_POSITION + TARGET_MOTION_AMPLITUDE * np.array([
        np.sin(phase), np.sin(phase), np.sin(2.0 * phase)])


def add_floor_grid():
    """Give the external scene a stationary reference for base movement."""
    # Debug lines are visual guides, not collision geometry.
    for coordinate in np.arange(-5.0, 5.01, 0.5):
        p.addUserDebugLine([-5, coordinate, 0.005], [5, coordinate, 0.005],
                           [0.4, 0.4, 0.4])
        p.addUserDebugLine([coordinate, -5, 0.005], [coordinate, 5, 0.005],
                           [0.4, 0.4, 0.4])


def load_scene():
    """Load URDF bodies and look up the joints/links needed by the controller."""
    # A PyBullet connection must exist before making scene API calls.
    # Zero gravity and a fixed base support this direct-pose kinematic example.
    p.setGravity(0, 0, 0)
    p.setTimeStep(DT)
    # loadURDF returns a body ID used in subsequent queries and updates.
    # useFixedBase prevents dynamics from moving the root; explicit resets below
    # can still reposition it to demonstrate planar mobile-base motion.
    p.loadURDF(str(ASSETS / "ground.urdf"), useFixedBase=True)
    robot = p.loadURDF(str(ASSETS / "waterloo_steel_full.urdf"),
                       [0, 0, 0.13], useFixedBase=True)
    joints = {}
    links = {}
    for index in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, index)
        # getJointInfo includes the joint name at [1], type at [2], and child
        # link name at [12]. Names are bytes, so decode them for dictionary keys.
        # A joint index also identifies that joint's child link in PyBullet.
        joints[info[1].decode()] = index
        links[info[12].decode()] = index
        if info[2] != p.JOINT_FIXED:
            # Disable default motors: this demo writes joint positions itself.
            p.setJointMotorControl2(robot, index, p.VELOCITY_CONTROL, force=0)
    arm = [joints[f"wam/J{i}"] for i in range(1, 8)]
    for joint, angle in zip(arm, NEUTRAL_CONFIGURATION[3:]):
        p.resetJointState(robot, joint, float(angle))
    target = p.loadURDF(str(ASSETS / "blue_block.urdf"),
                        TARGET_POSITION.tolist(), useFixedBase=True)
    return robot, arm, links["wam/wrist_palm_link"], target


def camera_pose(robot, wrist):
    """Compose the wrist's world pose with the fixed camera mounting transform."""
    state = p.getLinkState(robot, wrist, computeForwardKinematics=True)
    # Entries [4] and [5] are the world pose of the URDF link frame, rather
    # than the centre-of-mass frame. Rotate the local offset before adding it.
    wrist_rotation = rotation(state[5])
    position = np.array(state[4]) + wrist_rotation @ CAMERA_OFFSET
    return position, wrist_rotation @ OPTICAL_ROTATION


def capture_rgb(position, optical_rotation):
    """Render one camera, discarding the simulator's depth/segmentation data."""
    # computeViewMatrix takes eye position, a look-at point, and an up vector.
    # Look along optical +z; optical +y points down, so negate it to get up.
    # The camera follows the wrist orientation, not the target's world pose.
    view = p.computeViewMatrix(position.tolist(),
                               (position + optical_rotation[:, 2]).tolist(),
                               (-optical_rotation[:, 1]).tolist())
    # Perspective projection uses vertical FOV, aspect ratio, and near/far
    # clipping distances in metres. TinyRenderer also works without a GUI.
    projection = p.computeProjectionMatrixFOV(FOV, WIDTH / HEIGHT, 0.02, 8.0)
    result = p.getCameraImage(WIDTH, HEIGHT, view, projection,
                              renderer=p.ER_TINY_RENDERER,
                              flags=p.ER_NO_SEGMENTATION_MASK)
    # getCameraImage returns width, height, RGBA, depth, and segmentation.
    # Keep only colour and convert to OpenCV's BGR channel order.
    rgba = np.asarray(result[2], dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)


def detect_blue(frame):
    """Use the largest blue image region, rather than a simulator object ID."""
    # HSV separates hue from brightness. OpenCV's 8-bit hue range is 0..179;
    # the saturation/value thresholds reject pale or very dark pixels.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 100, 45]), np.array([135, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    # m00 is contour area; reject tiny regions before dividing by it.
    # The centroid (u, v) is measured from the image's top-left corner.
    if moments["m00"] < 20:
        return None
    return np.array([moments["m10"] / moments["m00"],
                     moments["m01"] / moments["m00"]])


def camera_jacobian(robot, arm, wrist, camera_position, optical_rotation):
    """Map ten base/arm rates to camera [vx, vy, vz, wx, wy, wz].

    Rows 0..2 describe camera linear velocity; rows 3..5 describe angular
    velocity. Use PyBullet for the arm columns, add the ideal planar base
    columns, then express all velocities in optical axes.
    """
    # calculateJacobian expects positions for ALL movable joints, including
    # wheels and fingers, in degree-of-freedom order. Fixed joints are omitted.
    movable = [joint for joint in range(p.getNumJoints(robot))
               if p.getJointInfo(robot, joint)[3] >= 0]
    positions = [state[0] for state in p.getJointStates(robot, movable)]
    zeros = [0.0] * len(movable)
    # Evaluate at the camera offset on the wrist, not at the wrist origin.
    # The wrist's URDF link and inertial axes are aligned in this model.
    linear, angular = p.calculateJacobian(
        robot, wrist, CAMERA_OFFSET.tolist(), positions, zeros, zeros)
    arm_columns = [movable.index(joint) for joint in arm]

    base_position, base_orientation = p.getBasePositionAndOrientation(robot)
    # For this fixed-base model, PyBullet returns the joint Jacobian in base
    # axes. Rotate into world axes, accounting for the mobile base's yaw.
    base_rotation = rotation(base_orientation)
    jacobian = np.zeros((6, 10))
    jacobian[:3, 3:] = base_rotation @ np.asarray(linear)[:, arm_columns]
    jacobian[3:, 3:] = base_rotation @ np.asarray(angular)[:, arm_columns]

    # The base is repositioned explicitly, so its three commanded rates are
    # not URDF joint degrees of freedom returned by calculateJacobian.
    z_axis = np.array([0., 0., 1.])
    jacobian[:3, :2] = np.eye(3)[:, :2]  # World-frame mobile base vx, vy.
    # Rotation about an axis produces linear velocity axis x displacement
    # at the offset camera, as well as angular velocity along the axis.
    jacobian[:3, 2] = np.cross(z_axis, camera_position - base_position)
    jacobian[3:, 2] = z_axis
    # Rotation matrices are orthogonal: transpose maps world into camera axes.
    jacobian[:3] = optical_rotation.T @ jacobian[:3]
    jacobian[3:] = optical_rotation.T @ jacobian[3:]
    return jacobian


def feedback_velocity(centroid, jacobian, configuration=None):
    """Choose base/arm rates that drive the detected point to the image centre.

    For normalised image error e, e_dot = L @ J_camera @ q_dot. Proportional
    feedback asks for e_dot = -GAIN * e; a damped inverse solves for q_dot.
    """
    if centroid is None:
        return np.zeros(10)  # Stop immediately if the block leaves the image.
    error_pixels = centroid - CENTRE
    if np.linalg.norm(error_pixels) <= PIXEL_TOLERANCE:
        return np.zeros(10)
    # Normalised coordinates are dimensionless: divide pixel displacement by
    # focal length. Desired coordinates are (0, 0), so (x, y) is also the error.
    x, y = error_pixels / FOCAL
    z = DEPTH_ESTIMATE
    # The 2 x 6 point-feature interaction matrix L predicts image motion from
    # camera velocity for optical x right, y down, z forward. Its translation
    # terms need depth; here z is a fixed estimate, not a depth-buffer reading.
    interaction = np.array([
        [-1/z, 0, x/z, x*y, -(1+x*x), y],
        [0, -1/z, y/z, 1+y*y, -x*y, -x],
    ])
    image_jacobian = interaction @ jacobian
    # J_image is 2 x 10: two image constraints leave many possible robot rates.
    # Scaling columns by MOBILITY favours some motions when choosing a solution.
    scaled_jacobian = image_jacobian * MOBILITY
    # For A = J_image @ diag(MOBILITY), use A.T @ inv(A @ A.T + damping^2 I).
    # solve avoids explicitly forming a matrix inverse; damping limits its
    # amplification when a direction of image motion is difficult to produce.
    inverse = scaled_jacobian.T @ np.linalg.solve(
        scaled_jacobian @ scaled_jacobian.T + DAMPING**2 * np.eye(2), np.eye(2))
    velocity = -GAIN * MOBILITY * (inverse @ np.array([x, y]))
    if configuration is not None:
        # Use the robot's own pose for a secondary return-to-start bias.
        # Approximately remove the part that changes the image feature;
        # damping makes this projection approximate rather than an exact nullspace.
        bias = -POSTURE_GAIN * (configuration - NEUTRAL_CONFIGURATION)
        velocity += bias - MOBILITY * (inverse @ (image_jacobian @ bias))
    # Scale all rates together so the commanded image-motion direction survives.
    velocity /= max(1.0, float(np.max(np.abs(velocity) / SPEED_LIMITS)))
    return velocity


def apply_velocity(robot, arm, velocity):
    """Integrate ideal planar base and arm rates; enforce the URDF arm limits."""
    # Explicit Euler integration: next position = current position + DT * rate.
    # reset* calls set state immediately; they are not motor-control commands.
    position, orientation = p.getBasePositionAndOrientation(robot)
    yaw = p.getEulerFromQuaternion(orientation)[2] + DT * velocity[2]
    p.resetBasePositionAndOrientation(
        robot, [position[0] + DT * velocity[0],
                position[1] + DT * velocity[1], 0.13],
        p.getQuaternionFromEuler([0, 0, yaw]))
    for joint, rate in zip(arm, velocity[3:]):
        info = p.getJointInfo(robot, joint)
        current = p.getJointState(robot, joint)[0]
        # getJointInfo()[8:10] holds lower/upper limits in radians for these
        # revolute joints. Leave a small margin inside either limit.
        updated = np.clip(current + DT * rate, info[8] + 0.01, info[9] - 0.01)
        p.resetJointState(robot, joint, float(updated), targetVelocity=0)
    # Advance one fixed simulation step. The commanded motion above is already
    # applied; stepSimulation does not integrate the servo rates for us.
    p.stepSimulation()


def annotate(frame, centroid):
    """Overlay the desired centre, measured centroid, and pixel-error length."""
    cv2.drawMarker(frame, tuple(CENTRE.astype(int)), (0, 255, 0),
                   cv2.MARKER_CROSS, 24, 1)
    label = "Blue block lost - robot stopped"
    if centroid is not None:
        point = tuple(np.rint(centroid).astype(int))
        cv2.circle(frame, point, 6, (0, 255, 255), 2)
        cv2.line(frame, point, tuple(CENTRE.astype(int)), (0, 255, 255), 1)
        label = f"Centre error: {np.linalg.norm(centroid - CENTRE):.2f} px"
    cv2.putText(frame, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (20, 20, 20), 3)
    cv2.putText(frame, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    return frame


def main():
    """Connect, build the scene, repeat the feedback loop, then release resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without GUI windows")
    parser.add_argument("--duration", type=float, default=0,
                        help="Simulated seconds; 0 = run indefinitely until quit")
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--moving-target", dest="moving_target", action="store_true",
                        help="Continuously move the blue block (default)")
    motion.add_argument("--static-target", dest="moving_target", action="store_false",
                        help="Hold the block still for a centring demonstration")
    parser.set_defaults(moving_target=True)
    parser.add_argument("--log", type=Path,
                        help="Write centroid, pixel error, base pose and arm joints to CSV")
    parser.add_argument("--save-frame", type=Path, help="Save the final annotated camera PNG")
    args = parser.parse_args()
    if args.duration < 0 or not np.isfinite(args.duration):
        parser.error("--duration must be a finite, non-negative number")
    duration = args.duration or float("inf")
    # GUI opens PyBullet's external scene viewer; DIRECT runs without a window.
    # Both connections support the separately rendered wrist camera image.
    if p.connect(p.DIRECT if args.headless else p.GUI) < 0:
        raise RuntimeError("Could not connect to PyBullet")
    log_file = None
    frame = None
    try:
        robot, arm, wrist, target = load_scene()
        if not args.headless:
            # This external debug view helps observe robot motion. It is
            # separate from the wrist camera that supplies controller feedback.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(3.7, 50, -25, [1.1, 0, 0.8])
            add_floor_grid()
            cv2.namedWindow("Wrist camera - Q or Esc to quit", cv2.WINDOW_AUTOSIZE)
        writer = None
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log_file = args.log.open("w", newline="")
            writer = csv.writer(log_file)
            writer.writerow(["time_s", "u_px", "v_px", "error_u_px", "error_v_px", "error_norm_px",
                             "base_x_m", "base_y_m", "base_yaw_rad"]
                            + [f"arm_J{i}_rad" for i in range(1, 8)])
        print("Single wrist RGB camera -> blue centroid -> image error -> base + arm rates")
        print("Ideal kinematic control; nominal depth = 2 m; centring tolerance = 2 pixels.")
        print(f"Block repeats a {TARGET_MOTION_PERIOD:g}-second figure-eight."
              if args.moving_target else "Block is stationary.")
        if not args.headless:
            print("Scene trails: orange = mobile base, green = wrist. Q or Esc to quit.")
        initial_error = None
        final_error = None
        step = 0
        while p.isConnected() and step * DT <= duration:
            started = time.perf_counter()
            sim_time = step * DT
            # 1. Animate the scene independently of the robot's feedback law.
            if args.moving_target:
                position = moving_target_position(sim_time)
                p.resetBasePositionAndOrientation(target, position.tolist(), [0, 0, 0, 1])
            # 2. Observe: render the wrist image and measure the blue centroid.
            camera_position, optical_rotation = camera_pose(robot, wrist)
            frame = capture_rgb(camera_position, optical_rotation)
            centroid = detect_blue(frame)
            final_error = None if centroid is None else float(np.linalg.norm(centroid - CENTRE))
            if step == 0:
                initial_error = final_error
            # Robot pose is available for kinematics, posture bias, and logging;
            # target location still comes only from the camera's colour pixels.
            base_position, base_orientation = p.getBasePositionAndOrientation(robot)
            arm_positions = [p.getJointState(robot, joint)[0] for joint in arm]
            if writer:
                writer.writerow([f"{sim_time:.4f}"] + (["", "", "", "", ""] if centroid is None
                    else [*centroid, *(centroid - CENTRE), final_error])
                    + [*base_position[:2], p.getEulerFromQuaternion(base_orientation)[2]]
                    + arm_positions)
            if step % 30 == 0:
                print(f"t={sim_time:5.1f}s  " + ("block lost - stopped" if final_error is None
                      else f"centre error={final_error:7.2f} px"))
            # 3. Control: map image error through camera/robot kinematics to
            # simultaneous base and arm rates. No target-world-position query.
            jacobian = camera_jacobian(robot, arm, wrist, camera_position, optical_rotation)
            configuration = np.array([*base_position[:2],
                p.getEulerFromQuaternion(base_orientation)[2], *arm_positions])
            velocity = feedback_velocity(centroid, jacobian, configuration)
            frame = annotate(frame, centroid)
            if not args.headless:
                cv2.imshow("Wrist camera - Q or Esc to quit", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty("Wrist camera - Q or Esc to quit", cv2.WND_PROP_VISIBLE) < 1:
                    break
            # The last logged/rendered image describes the final robot state.
            if sim_time + DT > duration:
                break
            # 4. Act: integrate the rates, then measure a fresh image next time.
            apply_velocity(robot, arm, velocity)
            if not args.headless:
                new_base = p.getBasePositionAndOrientation(robot)[0]
                p.addUserDebugLine([*base_position[:2], 0.01], [*new_base[:2], 0.01],
                                   [1.0, 0.4, 0.0], lineWidth=3, lifeTime=8)
                new_camera_position, _ = camera_pose(robot, wrist)
                p.addUserDebugLine(camera_position.tolist(), new_camera_position.tolist(),
                                   [0.1, 0.8, 0.2], lineWidth=2, lifeTime=8)
            step += 1
            if not args.headless:
                # Pace the display near real time. Headless mode skips this
                # delay; simulated time still advances by DT per update.
                time.sleep(max(0.0, DT - (time.perf_counter() - started)))
        print(f"Initial / final pixel error: {initial_error} / {final_error}")
    except KeyboardInterrupt:
        pass
    finally:
        # Release the simulator connection, file, and windows even after Ctrl+C.
        if log_file:
            log_file.close()
        if p.isConnected():
            p.disconnect()
        if not args.headless:
            cv2.destroyAllWindows()
        if args.save_frame and frame is not None:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save_frame), frame):
                raise OSError(f"Could not save frame: {args.save_frame}")


if __name__ == "__main__":
    main()
