import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

# 1. Global Settings
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seconds", type=float, default=None)
parser.add_argument("--record", action="store_true", help="Record X motion data.")
parser.add_argument("--fps", type=float, default=15., help="Camera and visual control rate in Hz (default: 15).")
args = parser.parse_args()
gui = not args.headless
if args.record and not gui:
  parser.error("Manual recording requires keyboard input. Run with --record and omit --headless.")
seconds = args.seconds
if seconds is None and not gui:
  seconds = 5.

DT = 1 / 240
# Choose a camera rate that divides 240 Hz evenly, e.g. 7.5, 15, 30 or 60 Hz.
steps_per_frame = round(240 / args.fps)
FRAME_DT = steps_per_frame * DT
assets = Path(__file__).resolve().parents[2] / "assets"
width, height, fov = 640, 480, 65  # Pixels and vertical FOV in degrees.
load_half = 0.09  # Half-edge length of the 180 mm cube.
tag_size = 0.06   # Black border edge length in metres (75 mm including white margin).
base_position = [0., 0.95, 0.06]
camera_offset = np.array([-0.045, 0., 0.])  # 45 mm above the grasp point along the gripper's -X axis.
camera_mount_q = p.getQuaternionFromEuler([0., 0., -np.pi / 2])  # Keep the camera image upright.
target_distance = 0.28  # Distance from the lens to the tag face (m).
# Start with the lens level with the tag and the fingers opening horizontally.
home = np.array([-0.116980, -1.300000, -1.211193, -2.410256, -2.668033, 2.363015, -0.824374])
crane_home = np.array([0., 0., 0.58])  # Y, X, cable length (m). Load centre starts at Z=0.60 m.
crane_lower = np.array([-0.35, -0.35, 0.30])
crane_upper = np.array([0.35, 0.35, 0.88])
crane_speed = np.array([0.26, 0.26, 0.15])  # Y, X, hoist (m/s).
crane_accel = np.array([0.65, 0.65, 0.25])  # m/s² while a movement key is held.
crane_brake = np.array([3.25, 3.25, 0.25])  # m/s² while a movement key is released.
track_min = np.array([-0.22, -0.10, 0.44])  # Tag workspace bounds (m), including sway margin.
track_max = np.array([0.22, 0.30, 0.82])
gain, damping, max_joint_speed = 9., 0.012, 1.2
corner_tolerance_px = 0.15  # Hold at subpixel alignment.

client = p.connect(p.GUI if gui else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(DT)
p.setRealTimeSimulation(0)
p.setPhysicsEngineParameter(numSolverIterations=100)
p.loadURDF("plane.urdf")
if gui:
  p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
  for preview in (p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                  p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
    p.configureDebugVisualizer(preview, 0)

# 2. Gantry crane and suspended load with free sway joints.
crane = p.loadURDF(str(assets / "gantry_crane.urdf"), useFixedBase=True)
crane_names = {p.getJointInfo(crane, i)[1].decode(): i for i in range(p.getNumJoints(crane))}
crane_joints = [crane_names[name] for name in ("gantry_y", "trolley_x", "hoist")]
swing_joints = [crane_names["sway_x"], crane_names["sway_y"]]
for joint, value in zip(crane_joints, crane_home):
  p.resetJointState(crane, joint, value)
for joint in swing_joints:
  p.setJointMotorControl2(crane, joint, p.VELOCITY_CONTROL, force=0)
  p.changeDynamics(crane, joint, jointDamping=0.03)

collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[load_half] * 3)
visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[load_half] * 3, rgbaColor=[0.9, 0.4, 0.12, 1])
tag = p.createVisualShape(p.GEOM_MESH, fileName=str(assets / "visual_servo/tag.obj"),
                          meshScale=[tag_size / 0.8] * 3, rgbaColor=[1, 1, 1, 1])
hook = p.getLinkState(crane, crane_joints[2], computeForwardKinematics=True)
load_home = np.asarray(hook[4]) + [0, 0, -0.05 - load_half]
# Create a box object
load = p.createMultiBody(
  baseMass=0.15, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=load_home,
  linkMasses=[0], linkCollisionShapeIndices=[-1], linkVisualShapeIndices=[tag],
  linkPositions=[[0, load_half + 0.001, 0]],
  linkOrientations=[p.getQuaternionFromEuler([np.pi / 2, 0, np.pi])],
  linkInertialFramePositions=[[0, 0, 0]], linkInertialFrameOrientations=[[0, 0, 0, 1]],
  linkParentIndices=[0], linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 0]],
)
texture = p.loadTexture(str(assets / "visual_servo/apriltag_36h11_0.png"))
p.changeVisualShape(load, 0, textureUniqueId=texture, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
p.changeDynamics(load, -1, linearDamping=0.03, angularDamping=0.03)
# Convert the attachment pose to the hook's centre-of-mass frame.
relative = p.multiplyTransforms(*p.invertTransform(hook[0], hook[1]), load_home, [0, 0, 0, 1])
latch = p.createConstraint(crane, crane_joints[2], load, -1, p.JOINT_FIXED, [0, 0, 0],
                           relative[0], [0, 0, 0], parentFrameOrientation=relative[1])
p.changeConstraint(latch, maxForce=200)

# Draw the cable with short cylinder segments.
cable_count, segment_length = 16, 0.06
segment = p.createVisualShape(p.GEOM_CYLINDER, radius=0.003, length=segment_length,
                              rgbaColor=[0.8, 0.65, 0.12, 1])
cable = p.createMultiBody(
  baseMass=0, baseVisualShapeIndex=segment,
  linkMasses=[0] * cable_count, linkCollisionShapeIndices=[-1] * cable_count,
  linkVisualShapeIndices=[segment] * cable_count, linkPositions=[[0, 0, 0]] * cable_count,
  linkOrientations=[[0, 0, 0, 1]] * cable_count, linkInertialFramePositions=[[0, 0, 0]] * cable_count,
  linkInertialFrameOrientations=[[0, 0, 0, 1]] * cable_count,
  linkParentIndices=list(range(cable_count)), linkJointTypes=[p.JOINT_PRISMATIC] * cable_count,
  linkJointAxis=[[0, 0, 1]] * cable_count,
)

# 3. Fixed pedestal and Franka Panda arm.
collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.18, height=base_position[2])
visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.18, length=base_position[2],
                             rgbaColor=[0.22, 0.30, 0.36, 1])
pedestal = p.createMultiBody(0, collision, visual, [0, base_position[1], base_position[2] / 2])
robot = p.loadURDF("franka_panda/panda.urdf", base_position, useFixedBase=True)
info = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
arm_joints = [j[0] for j in info if j[2] == p.JOINT_REVOLUTE]
movable = [j[0] for j in info if j[3] >= 0]
tip = next(j[0] for j in info if j[12] == b"panda_grasptarget")
lower = np.array([info[i][8] for i in arm_joints]) + 0.08
upper = np.array([info[i][9] for i in arm_joints]) - 0.08
p.setCollisionFilterPair(robot, pedestal, -1, -1, 0)
for joint, value in zip(arm_joints, home):
  p.resetJointState(robot, joint, value)
for joint in movable[7:]:
  p.resetJointState(robot, joint, 0.015)
  p.setJointMotorControl2(robot, joint, p.POSITION_CONTROL, targetPosition=0.015, force=20)
# Visual-only camera housing, 24 x 32 x 80 mm.
housing = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012, 0.016, 0.04],
  visualFramePosition=[0, 0, -0.04], rgbaColor=[0.08, 0.30, 0.38, 1])
camera_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=housing)

# 4. Camera and AprilTag detector. Optical axes are X right, Y down, Z forward.
cv2.setNumThreads(2)
focal = height / (2 * np.tan(np.deg2rad(fov) / 2))
# TinyRenderer projects the optical axis to (width / 2, height / 2 - 1).
K = np.array([[focal, 0, width / 2], [0, focal, height / 2 - 1], [0, 0, 1.]])
projection = p.computeProjectionMatrixFOV(fov, width / height, 0.02, 3.)
h = tag_size / 2
marker_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
# Project a centred, front-facing tag into normalized camera coordinates (image Y points down).
desired = marker_points[:, :2] * [1., -1.] / target_distance

if gui:
  camera_window = "Panda wrist camera"
  camera_ui = cv2.currentUIFramework()
  cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(camera_window, width, height)
  p.resetDebugVisualizerCamera(
    cameraDistance=1.35,
    cameraYaw=230,
    cameraPitch=-26,
    cameraTargetPosition=[0, 0.35, 0.70],
  )
  controls = ("A/Left: -X   D/Right: +X   Esc: save and exit" if args.record else
              "W/S: Y   A/D: X   Q/E: lift   Space: pause   R: reset   Esc: exit")
  p.addUserDebugText(controls, [-0.8, 0, 1.62], [0.1, 0.1, 0.1], 1.1)
if args.record:
  print("Focus the simulation window. A/Left: -X, D/Right: +X. Release to brake. Esc saves and exits.", flush=True)
elif gui:
  print("Focus the simulation window. W/S: Y, A/D: X, Q/E: hoist, Space: pause, R: reset, Esc: exit.", flush=True)
held, paused = set(), False
crane_target, crane_velocity = crane_home.copy(), np.zeros(3)
arm_target, joint_velocity = home.copy(), np.zeros(7)
previous_tag_world = None
frame, status_id, last_status = 0, -1, ""
deadline = time.perf_counter()

# 5. Read input, measure image error, command joints, and step physics.
samples = []
try:
  while p.isConnected(client) and (seconds is None or frame * FRAME_DT < seconds):
    pressed, events = set(), {}
    for key, flags in p.getKeyboardEvents().items():
      key = key + 32 if ord("A") <= key <= ord("Z") else key
      events[key] = events.get(key, 0) | flags
    for key, flags in events.items():
      if flags & p.KEY_WAS_TRIGGERED and key not in held:
        pressed.add(key)
      repress = flags & p.KEY_IS_DOWN and flags & p.KEY_WAS_TRIGGERED
      if flags & p.KEY_WAS_RELEASED and not repress:
        held.discard(key)
      elif flags & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN):
        held.add(key)
    if 27 in pressed:
      break
    if ord(" ") in pressed:
      paused = not paused
    if ord("r") in pressed:
      for joint, value in zip(crane_joints + swing_joints, [*crane_home, 0, 0]):
        p.resetJointState(crane, joint, value)
      p.resetBasePositionAndOrientation(load, load_home, [0, 0, 0, 1])
      p.resetBaseVelocity(load, [0, 0, 0], [0, 0, 0])
      for joint, value in zip(arm_joints, home):
        p.resetJointState(robot, joint, value)
      crane_target, crane_velocity = crane_home.copy(), np.zeros(3)
      arm_target, joint_velocity = home.copy(), np.zeros(7)
      paused = False
      previous_tag_world = None
      held.clear()
    direction = np.array([
      int(bool(held & {ord("w"), p.B3G_UP_ARROW})) - int(bool(held & {ord("s"), p.B3G_DOWN_ARROW})),
      int(bool(held & {ord("d"), p.B3G_RIGHT_ARROW})) - int(bool(held & {ord("a"), p.B3G_LEFT_ARROW})),
      int(ord("e") in held) - int(ord("q") in held),
    ])

    if args.record:
      direction[0] = direction[2] = 0  # Only the trolley X axis is driven during recording.

    # Render the wrist camera image for visual feedback.
    state = p.getLinkState(robot, tip, computeForwardKinematics=True)
    eye, camera_q = p.multiplyTransforms(state[4], state[5], camera_offset, camera_mount_q)
    eye = np.asarray(eye)
    R = np.asarray(p.getMatrixFromQuaternion(camera_q)).reshape(3, 3)
    p.resetBasePositionAndOrientation(camera_body, eye, camera_q)
    view = p.computeViewMatrix(eye, eye + R[:, 2], -R[:, 1])
    rgba = p.getCameraImage(width, height, view, projection, renderer=p.ER_TINY_RENDERER,
      shadow=0, lightAmbientCoeff=0.9, lightDiffuseCoeff=0.1, lightSpecularCoeff=0,
      flags=p.ER_NO_SEGMENTATION_MASK)[2]
    if not isinstance(rgba, np.ndarray):
      rgba = np.frombuffer(bytes(rgba), dtype=np.uint8)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if gui:
      if camera_ui != "WAYLAND":  # Wayland does not report window visibility.
        try:
          if cv2.getWindowProperty(camera_window, cv2.WND_PROP_VISIBLE) < 1:
            break
        except cv2.error:  # Some backends remove a closed window immediately.
          break
      preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
      if ids is not None:
        cv2.aruco.drawDetectedMarkers(preview, corners, ids)
      # Show the image window.
      cv2.imshow(camera_window, preview)
      # PyBullet already processes Cocoa events. waitKey would consume the crane controls.
      if camera_ui != "COCOA" and cv2.waitKey(1) == 27:
        break
    q = np.array([s[0] for s in p.getJointStates(robot, movable)])
    previous_velocity = joint_velocity.copy()
    joint_velocity = np.zeros(7)
    status, error_px = "TARGET LOST - move crane back into view", float("nan")

    if ids is not None and 0 in ids.ravel():
      pixels = corners[int(np.flatnonzero(ids.ravel() == 0)[0])].reshape(4, 2).astype(np.float64)
      normalized = (pixels - K[:2, 2]) / focal
      # Estimate corner depths and tag position from the detected pixels.
      ok, rvec, tvec = cv2.solvePnP(marker_points, pixels, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
      if ok:
        points = marker_points @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()
        tag_world = eye + R @ tvec.ravel()
        error = (normalized - desired).ravel()
        error_px = float(np.sqrt(np.mean(error**2)) * focal)
        in_range = np.all(tag_world >= track_min) and np.all(tag_world <= track_max)
        status = "TRACKING" if in_range else "OUT OF RANGE - move crane back"
        if in_range and error_px > corner_tolerance_px and np.isfinite(points).all() and np.min(points[:, 2]) > 0.05:
          rows = []
          for (x, y), Z in zip(normalized, points[:, 2]):
            rows.extend(([-1/Z, 0, x/Z, x*y, -(1+x*x), y],
                         [0, -1/Z, y/Z, 1+y*y, -x*y, -x]))
          L = np.asarray(rows)  # 8x6 interaction matrix for four image corners.
          zeros = [0.] * len(q)
          linear, angular = p.calculateJacobian(robot, tip, camera_offset, q.tolist(), zeros, zeros)
          J = np.vstack((R.T @ np.asarray(linear)[:, :7], R.T @ np.asarray(angular)[:, :7]))
          A = L @ J             # 8x7 image Jacobian for seven arm joints.
          inverse = A.T @ np.linalg.solve(A @ A.T + damping**2 * np.eye(8), np.eye(8))
          image_velocity = -gain * error
          if previous_tag_world is not None:
            # Follow the tag's measured translation between camera frames.
            tag_velocity_camera = R.T @ (tag_world - previous_tag_world) / FRAME_DT
            image_velocity += L[:, :3] @ tag_velocity_camera
          joint_velocity = inverse @ image_velocity
          joint_velocity += (np.eye(7) - inverse @ A) @ (0.08 * (home - q[:7]))
          joint_velocity *= min(1., max_joint_speed / max(np.max(np.abs(joint_velocity)), 1e-9))
          joint_velocity = np.clip(joint_velocity, np.minimum(0, (lower - q[:7]) / FRAME_DT),
                                   np.maximum(0, (upper - q[:7]) / FRAME_DT))
    if paused:
      status, joint_velocity = "PAUSED - Space to resume", np.zeros(7)
    previous_tag_world = tag_world if status == "TRACKING" else None
    if not np.any(joint_velocity) and np.any(previous_velocity):
      arm_target = q[:7].copy()  # Hold the current pose when tracking stops.
    if status != last_status:
      print(status, flush=True)
      if gui:
        color = [0.1, 0.55, 0.2] if status == "TRACKING" else [0.8, 0.25, 0.1]
        status_id = p.addUserDebugText(status, [-0.8, 0, 1.49], color, 1.1, replaceItemUniqueId=status_id)
      last_status = status

    if args.record:
      trolley_x = p.getJointState(crane, crane_joints[1])[0]
      tag_x = p.getLinkState(load, 0, computeForwardKinematics=True)[4][0]
      last_camera = rgb, corners, ids  # Keep the image paired with the last logged sample.
      samples.append((frame * FRAME_DT, direction[1], trolley_x, tag_x, eye[0],
                      error_px if np.isfinite(error_px) else None))

    max_velocity_step = np.where(direction == 0, crane_brake, crane_accel) * DT
    for _ in range(steps_per_frame):
      crane_velocity += np.clip(direction * crane_speed - crane_velocity, -max_velocity_step, max_velocity_step)
      proposed = crane_target + crane_velocity * DT
      crane_target = np.clip(proposed, crane_lower, crane_upper)
      crane_velocity[proposed != crane_target] = 0
      p.setJointMotorControlArray(crane, crane_joints, p.POSITION_CONTROL,
        targetPositions=crane_target, targetVelocities=crane_velocity, forces=[200, 200, 80],
        positionGains=[0.5] * 3, velocityGains=[1.] * 3)
      actual = np.array([s[0] for s in p.getJointStates(robot, arm_joints)])
      arm_target = np.clip(arm_target + joint_velocity * DT, actual - 0.10, actual + 0.10)
      arm_target = np.clip(arm_target, lower, upper)
      p.setJointMotorControlArray(robot, arm_joints, p.POSITION_CONTROL,
        targetPositions=arm_target, targetVelocities=joint_velocity, forces=[87, 87, 87, 87, 12, 12, 12],
        positionGains=[0.3] * 7, velocityGains=[1.] * 7)
      p.stepSimulation()

    # Update camera and cable visuals.
    state = p.getLinkState(robot, tip, computeForwardKinematics=True)
    eye, camera_q = p.multiplyTransforms(state[4], state[5], camera_offset, camera_mount_q)
    p.resetBasePositionAndOrientation(camera_body, eye, camera_q)
    hook = p.getLinkState(crane, crane_joints[2], computeForwardKinematics=True)
    R_hook = np.asarray(p.getMatrixFromQuaternion(hook[5])).reshape(3, 3)
    cable_length = p.getJointState(crane, crane_joints[2])[0]
    p.resetBasePositionAndOrientation(cable, np.asarray(hook[4]) + R_hook[:, 2] * segment_length / 2, hook[5])
    p.resetJointStatesMultiDof(cable, list(range(cable_count)),
                              [[(cable_length - segment_length) / cable_count]] * cable_count)
    frame += 1
    if gui:
      # Redraw the main window once per vision update.
      p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
      deadline += FRAME_DT
      delay = deadline - time.perf_counter()
      if delay > 0:
        time.sleep(delay)
      elif delay < -0.1:
        deadline = time.perf_counter()
except p.error:
  if p.isConnected(client):
    raise
finally:
  if gui:
    cv2.destroyAllWindows()
  if p.isConnected(client):
    p.disconnect(client)

if args.record and samples:
  rate_label = f"{args.fps:g}Hz"
  output = Path(__file__).resolve().parent / "results" / f"tracking_{rate_label}.csv"
  output.parent.mkdir(exist_ok=True)
  # Prepare both files before replacing the previous recording.
  with TemporaryDirectory(dir=output.parent) as staging:
    recording = Path(staging) / output.name
    with recording.open("w", newline="", encoding="utf-8") as stream:
      writer = csv.writer(stream)
      writer.writerow(["time_s", "input_x", "trolley_x_m", "tag_x_m", "camera_x_m", "corner_rms_px"])
      writer.writerows(samples)
    rgb, corners, ids = last_camera
    final_view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if ids is not None:
      cv2.aruco.drawDetectedMarkers(final_view, corners, ids)
    snapshot = Path(staging) / f"final_camera_{rate_label}.png"
    if not cv2.imwrite(str(snapshot), final_view):
      raise OSError(f"Could not save {snapshot.name}")
    snapshot.replace(output.parent / snapshot.name)
    recording.replace(output)
  print(f"Saved {len(samples)} frames to {output}")
