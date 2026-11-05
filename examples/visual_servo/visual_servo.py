"""Minimal image-based visual servo for a kinematic differential-drive cart.

Run: conda run --no-capture-output -n robotics python examples/visual_servo.py
Click PyBullet: W/S (X), A/D (Y), Q/E (board yaw), Space (pause), R (reset).
Escape exits either window. Control uses rendered AprilTag corners, not board pose.
"""

import argparse
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

WIDTH, HEIGHT, FOV = 480, 360, 60
HZ = 30
DT = 1 / HZ
MARKER_SIZE, MARKER_ID = 0.40, 0  # Black square size [m], excluding white border.
DISTANCE = 1.6
CAMERA_OFFSET, CAMERA_HEIGHT = 0.18, 0.62
WHEEL_RADIUS, TRACK = 0.10, 0.46
GAIN, DAMPING = 1.8, 0.015
MAX_SPEED, MAX_YAW_RATE, MAX_WHEEL_RATE = 0.65, 1.25, 10.0
BOARD_SPEED = 0.35

# Optical axes: right (+x), down (+y), forward (+z).
# J_camera (6x2): axle [v, omega] -> camera [vx, vy, vz, wx, wy, wz].
J_CAMERA = np.array([[0, -CAMERA_OFFSET], [0, 0], [1, 0],
                     [0, 0], [0, -1], [0, 0]], dtype=float)
# J_wheels (2x2): left/right wheel rates [rad/s] -> axle [m/s, rad/s].
J_WHEELS = WHEEL_RADIUS * np.array([[0.5, 0.5], [-1 / TRACK, 1 / TRACK]])


def interaction_matrix(points):
  """L (8x6): camera velocity -> four normalized corner velocities."""
  rows = []
  for X, Y, Z in points:
    x, y = X / Z, Y / Z
    rows.extend(([-1 / Z, 0, x / Z, x * y, -(1 + x * x), y],
                 [0, -1 / Z, y / Z, 1 + y * y, -x * y, -x]))
  return np.asarray(rows)


def main(gui=True, seconds=None, snapshot=None):
  if not gui and seconds is None:
    seconds = 3.0
  p.connect(p.GUI if gui else p.DIRECT)
  try:
    with tempfile.TemporaryDirectory(prefix="cart_ibvs_") as directory:
      # 1. Camera calibration, detector and desired frontal image corners.
      cv2.setNumThreads(2)
      focal = HEIGHT / (2 * np.tan(np.deg2rad(FOV) / 2))
      K = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.]])
      projection = p.computeProjectionMatrixFOV(FOV, WIDTH / HEIGHT, 0.03, 30)
      h = MARKER_SIZE / 2
      marker_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
      desired = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) / DISTANCE
      dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
      parameters = cv2.aruco.DetectorParameters()
      parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
      detector = cv2.aruco.ArucoDetector(dictionary, parameters)

      # 2. Visual-only scene. Motion below uses wheel odometry, not tire physics.
      p.setAdditionalSearchPath(pybullet_data.getDataPath())
      p.loadURDF("plane.urdf")
      p.setRealTimeSimulation(0)
      blue, dark = [0.08, 0.37, 0.63, 1], [0.08, 0.09, 0.11, 1]
      chassis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.28, 0.19, 0.075], rgbaColor=blue)
      wheel = p.createVisualShape(p.GEOM_CYLINDER, radius=WHEEL_RADIUS, length=0.05,
                                    visualFrameOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]), rgbaColor=dark)
      caster = p.createVisualShape(p.GEOM_SPHERE, radius=0.045, rgbaColor=dark)
      mast = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.018, 0.018, 0.16], rgbaColor=[0.65, 0.68, 0.7, 1])
      camera = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.045, 0.065, 0.035], rgbaColor=dark)
      lens = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.012,
                                   visualFrameOrientation=p.getQuaternionFromEuler([0, np.pi / 2, 0]),
                                   rgbaColor=[0.15, 0.75, 0.8, 1])
      robot = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=chassis, basePosition=[0, 0, 0.20],
        linkMasses=[0] * 7, linkCollisionShapeIndices=[-1] * 7,
        linkVisualShapeIndices=[wheel, wheel, caster, caster, mast, camera, lens],
        linkPositions=[[0, TRACK / 2, -0.10], [0, -TRACK / 2, -0.10],
                       [-0.22, 0, -0.155], [0.22, 0, -0.155],
                       [CAMERA_OFFSET - 0.06, 0, 0.245], [CAMERA_OFFSET - 0.06, 0, 0.42],
                       [CAMERA_OFFSET - 0.01, 0, 0.42]],
        linkOrientations=[[0, 0, 0, 1]] * 7, linkInertialFramePositions=[[0, 0, 0]] * 7,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * 7, linkParentIndices=[0] * 7,
        linkJointTypes=[p.JOINT_REVOLUTE] * 2 + [p.JOINT_FIXED] * 5,
        linkJointAxis=[[0, 1, 0]] * 7,
      )
      # Runtime-only marker asset: white quiet zone and a UV-mapped flat quad.
      image = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, 400)
      texture_path, mesh_path = Path(directory) / "tag.png", Path(directory) / "tag.obj"
      if not cv2.imwrite(str(texture_path), cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)):
        raise OSError("Could not write marker texture")
      mesh_path.write_text("v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\n"
                           "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
      tag = p.createVisualShape(p.GEOM_MESH, fileName=str(mesh_path), meshScale=[MARKER_SIZE * 1.25] * 3,
                                  rgbaColor=[1, 1, 1, 1])
      panel = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.32, 0.31], rgbaColor=[0.93, 0.94, 0.95, 1])
      post = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.15], rgbaColor=dark)
      foot = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.20, 0.35, 0.025], rgbaColor=dark)
      target = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=panel,
        linkMasses=[0] * 4, linkCollisionShapeIndices=[-1] * 4,
        linkVisualShapeIndices=[tag, post, post, foot],
        linkPositions=[[-0.026, 0, 0], [0, -0.25, -0.45], [0, 0.25, -0.45], [0, 0, -0.595]],
        linkOrientations=[p.getQuaternionFromEuler([np.pi / 2, 0, -np.pi / 2])] + [[0, 0, 0, 1]] * 3,
        linkInertialFramePositions=[[0, 0, 0]] * 4, linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
        linkParentIndices=[0] * 4, linkJointTypes=[p.JOINT_FIXED] * 4, linkJointAxis=[[0, 0, 1]] * 4,
      )
      p.changeVisualShape(target, 0, textureUniqueId=p.loadTexture(str(texture_path)),
                            flags=p.VISUAL_SHAPE_DOUBLE_SIDED)

      cart = np.zeros(3)  # World [x, y, yaw].
      board = np.array([2.3, 0., 0.])
      wheel_angles = np.zeros(2)
      if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        for preview in (p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                        p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
          p.configureDebugVisualizer(preview, 0)
        p.resetDebugVisualizerCamera(4.3, -45, -32, [1.4, 0, 0.35])
        cv2.namedWindow("Cart visual servo", cv2.WINDOW_NORMAL)
      print("Click PyBullet: W/S X, A/D Y, Q/E yaw; Space pause; R reset; Esc exit.")
      down, released = set(), set()
      frame, detections, paused = 0, 0, False
      image, error, initial_error = None, None, None
      rates = np.zeros(2)
      deadline = time.perf_counter()
      while p.isConnected() and (seconds is None or frame < round(seconds * HZ)):
        # 3. Retain held keys between polls; actions trigger once per press.
        pressed, events = set(), {}
        for key, flags in p.getKeyboardEvents().items():
          key = key + 32 if ord("A") <= key <= ord("Z") else key
          events[key] = events.get(key, 0) | flags
        for key, flags in events.items():
          was_down = key in down
          repress = flags & p.KEY_IS_DOWN and flags & p.KEY_WAS_TRIGGERED
          if flags & p.KEY_WAS_RELEASED and not repress:
            down.discard(key)
            released.add(key)
          elif flags & p.KEY_WAS_TRIGGERED:
            down.add(key)
            released.discard(key)
          elif flags & p.KEY_IS_DOWN and key not in released:
            down.add(key)  # Ignore delayed DOWN-only events after a release.
          if flags & p.KEY_WAS_TRIGGERED and not was_down:
            pressed.add(key)
        if 27 in pressed:
          break
        if ord(" ") in pressed:
          paused = not paused
        if ord("r") in pressed:
          cart[:] = 0
          board[:] = [2.3, 0, 0]
          wheel_angles[:] = 0
        movement = np.array([
          bool(down & {ord("w"), p.B3G_UP_ARROW}) - bool(down & {ord("s"), p.B3G_DOWN_ARROW}),
          bool(down & {ord("a"), p.B3G_LEFT_ARROW}) - bool(down & {ord("d"), p.B3G_RIGHT_ARROW})])
        board[:2] += BOARD_SPEED * DT * movement / max(1, np.linalg.norm(movement))
        board[2] += 0.6 * DT * ((ord("q") in down) - (ord("e") in down))
        for body, pose, z in ((robot, cart, 0.20), (target, board, CAMERA_HEIGHT)):
          p.resetBasePositionAndOrientation(body, [*pose[:2], z], p.getQuaternionFromEuler([0, 0, pose[2]]))
        for index, angle in enumerate(wheel_angles):
          p.resetJointState(robot, index, angle)

        # 4. Render the onboard camera and detect actual image corners.
        forward = np.array([np.cos(cart[2]), np.sin(cart[2]), 0])
        eye = np.r_[cart[:2], CAMERA_HEIGHT] + CAMERA_OFFSET * forward
        view = p.computeViewMatrix(eye, eye + forward, [0, 0, 1])
        rgba = p.getCameraImage(WIDTH, HEIGHT, view, projection, renderer=p.ER_TINY_RENDERER,
                                lightAmbientCoeff=0.9, lightDiffuseCoeff=0.1, lightSpecularCoeff=0,
                                shadow=0, flags=p.ER_NO_SEGMENTATION_MASK)[2]
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3]
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        rates, error, pixels, depths = np.zeros(2), None, None, None
        if ids is not None and MARKER_ID in ids:
          pixels = corners[int(np.flatnonzero(ids.ravel() == MARKER_ID)[0])].reshape(4, 2).astype(float)
          # PnP supplies only corner depths; reject behind-camera / poor solutions.
          result = cv2.solvePnPGeneric(marker_points, pixels, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
          best_error = 3.0
          for rvec, tvec in zip(result[1], result[2]):
            points = marker_points @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()
            projected = cv2.projectPoints(marker_points, rvec, tvec, K, None)[0].reshape(4, 2)
            reprojection = np.sqrt(np.mean((pixels - projected) ** 2))
            if np.isfinite(points).all() and np.min(points[:, 2]) > 0.05 and reprojection < best_error:
              best_error, depths = reprojection, points[:, 2]

        # 5. IBVS: wheel rates -> axle velocity -> camera twist -> image motion.
        if depths is not None:
          normalized = (pixels - K[:2, 2]) / focal
          error = (normalized - desired).ravel()  # Eight measured image errors.
          L = interaction_matrix(np.column_stack((normalized, np.ones(4))) * depths[:, None])
          J = L @ J_CAMERA @ J_WHEELS  # (8x6) @ (6x2) @ (2x2) = 8x2.
          # Solve J @ rates ~= -GAIN * error, with damping near singularities.
          rates = -GAIN * np.linalg.solve(J.T @ J + DAMPING**2 * np.eye(2), J.T @ error)
          v, omega = J_WHEELS @ rates
          rates /= max(1., abs(v) / MAX_SPEED, abs(omega) / MAX_YAW_RATE,
                       np.max(np.abs(rates)) / MAX_WHEEL_RATE)
          if np.sqrt(np.mean(error**2)) < 0.0025 or (min(depths) < 0.55 and v > 0):
            rates[:] = 0
          detections += 1
          if initial_error is None:
            initial_error = float(np.linalg.norm(error))
        if paused:
          rates[:] = 0
        # Missing/invalid detections leave rates zero: no ground-truth fallback.

        # 6. Integrate differential-drive motion for the next camera frame.
        v, omega = J_WHEELS @ rates
        angle = omega * DT
        travel = v * DT * np.sinc(angle / (2 * np.pi))
        cart += [travel * np.cos(cart[2] + angle / 2),
                 travel * np.sin(cart[2] + angle / 2), angle]
        wheel_angles += rates * DT

        # 7. Display and pace at 30 Hz. Slow rendering never enlarges the time step.
        if gui or snapshot is not None:
          image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
          goal_pixels = (desired * focal + K[:2, 2]).astype(np.int32)
          cv2.polylines(image, [goal_pixels], True, (0, 190, 255), 1)
          if pixels is not None:
            cv2.polylines(image, [pixels.astype(np.int32)], True, (0, 255, 0), 1)
          status = "PAUSED" if paused else "TRACKING" if depths is not None else "TAG LOST - STOP"
          lines = [status, f"v {v:+.2f} m/s | yaw {omega:+.2f} rad/s",
                   "PyBullet: WASD / arrows, Q/E | Space, R, Esc"]
          for i, line in enumerate(lines):
            cv2.putText(image, line, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (30, 220, 30), 1, cv2.LINE_AA)
        if gui:
          cv2.imshow("Cart visual servo", image)
          if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Cart visual servo", cv2.WND_PROP_VISIBLE) < 1:
            break
        frame += 1
        if gui:
          deadline += DT
          remaining = deadline - time.perf_counter()
          if remaining > 0:
            time.sleep(remaining)
          elif remaining < -0.1:
            deadline = time.perf_counter()
      if snapshot is not None and image is not None:
        path = Path(snapshot)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
          raise OSError(f"Could not save {path}")
      print(f"Frames: {frame}, detections: {detections}, final image error: {error}")
      return {"frames": frame, "detections": detections, "error": error,
              "initial_error": initial_error, "cart": cart, "rates": rates}
  except KeyboardInterrupt:
    return None
  except p.error:
    if p.isConnected():
      raise
    return None
  finally:
    if gui:
      cv2.destroyAllWindows()
    if p.isConnected():
      p.disconnect()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--headless", action="store_true", help="DIRECT renderer, no windows")
  parser.add_argument("--seconds", type=float, help="Stop after this many simulation seconds")
  parser.add_argument("--snapshot", help="Save final annotated camera image")
  args = parser.parse_args()
  if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
    parser.error("--seconds must be positive and finite")
  main(gui=not args.headless, seconds=args.seconds, snapshot=args.snapshot)
