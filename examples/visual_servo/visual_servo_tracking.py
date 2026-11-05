import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

# 1. 参数。物理 240 Hz，图像和键盘 15 Hz；为 640×480 相机和主窗口绘制留出时间。
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seconds", type=float, default=None)
args = parser.parse_args()
gui = not args.headless
seconds = args.seconds if args.seconds is not None else (None if gui else 5.)
assets = Path(__file__).resolve().parents[1] / "assets"
dt, steps_per_frame = 1 / 240, 16
frame_dt = dt * steps_per_frame
width, height, fov = 640, 480, 65  # 处理分辨率；fov 是垂直视场角，非对角视场角。
load_half = 0.09  # 吊物半边长，完整尺寸 180 × 180 × 180 mm。
tag_size = 0.06   # 36h11 黑框外边长 60 mm；含白边的完整贴纸边长为 75 mm。
# base_position = [0., 1.05, 0.30]
base_position = [0., 0.95, 0.30]
camera_offset = np.array([0.055, 0., 0.])  # panda_grasptarget 坐标系中的光心位置。
# 此关节姿态的光心相对基座约为 [0, -0.609, 0.420]，光轴向下倾斜约 19 度。
# 当前基座 Y=0.95 时，初始光心约 [0, 0.341, 0.720]，离标记中心约 0.28 m。
# 直接从该姿态开始，避免将未收敛的 IK 解截断后当作有效姿态。
home = np.array([-0.073224, -0.702598, -1.312661, -2.272324, -2.284241, 2.595941, -2.173273])
crane_home = np.array([0., 0., 0.58])  # Y、X、绳长；吊物中心初始高度 0.60 m。
crane_lower = np.array([-0.35, -0.35, 0.30])
crane_upper = np.array([0.35, 0.35, 0.88])
crane_speed = np.array([0.15, 0.15, 0.15])
track_min = np.array([-0.22, -0.10, 0.44])  # 标记中心的允许范围，包含正常摆动余量，m。
track_max = np.array([0.22, 0.30, 0.82])
gain, damping, max_joint_speed = 4., 0.02, 1.2

client = p.connect(p.GUI if gui else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(dt)
p.setRealTimeSimulation(0)
p.setPhysicsEngineParameter(numSolverIterations=100)
p.loadURDF("plane.urdf")
if gui:
  p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
  for preview in (p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                  p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
    p.configureDebugVisualizer(preview, 0)

# 2. 龙门吊与已经挂好的吊物。两根摆动轴不加电机力，允许自然摆动。
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
# createConstraint 的父坐标系是连杆质心，因此先换算固定连接的位置。
relative = p.multiplyTransforms(*p.invertTransform(hook[0], hook[1]), load_home, [0, 0, 0, 1])
latch = p.createConstraint(crane, crane_joints[2], load, -1, p.JOINT_FIXED, [0, 0, 0],
                           relative[0], [0, 0, 0], parentFrameOrientation=relative[1])
p.changeConstraint(latch, maxForce=200)

# 绳子仅作显示：短圆柱首尾相接。更新其位置不会阻塞主窗口的 debug-line 绘制。
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

# 3. 圆柱底座和机械臂。底座不动，正常跟踪时只给关节电机下达指令。
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
# 相机组件外观示意尺寸 24 × 32 × 80 mm；仅作显示，未模拟真实相机的质量和碰撞。
housing = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012, 0.016, 0.04],
  visualFramePosition=[0, 0, -0.04], rgbaColor=[0.08, 0.30, 0.38, 1])
camera_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=housing)

# 4. 相机标定和 AprilTag 检测。光学坐标：X 向右、Y 向下、Z 向前。
cv2.setNumThreads(2)
focal = height / (2 * np.tan(np.deg2rad(fov) / 2))
K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])
projection = p.computeProjectionMatrixFOV(fov, width / height, 0.02, 3.)
h = tag_size / 2
marker_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
desired = None  # 启动时拍到的角点就是参考图像，随后保持这个位置、大小和形状。

if gui:
  p.resetDebugVisualizerCamera(2.6, 135, -26, [0, 0.35, 0.70])
  p.addUserDebugText("W/S: Y   A/D: X   Q/E: lift   Space: pause   R: reset   Esc: exit",
                     [-0.8, 0, 1.62], [0.1, 0.1, 0.1], 1.1)
print("点击主窗口后操作：W/S 前后，A/D 左右，Q/E 升降；空格暂停跟踪，R 复位，Esc 退出。", flush=True)
held, paused = set(), False
crane_target, crane_velocity = crane_home.copy(), np.zeros(3)
arm_target, joint_velocity = home.copy(), np.zeros(7)
frame, status_id, last_status = 0, -1, ""
deadline = time.perf_counter()

# 5. 主循环：键盘 → 相机图像 → 图像误差 → 关节速度 → 物理步进。
try:
  while p.isConnected(client) and (seconds is None or frame * frame_dt < seconds):
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
      paused, desired = False, None
      held.clear()
    direction = np.array([
      int(bool(held & {ord("w"), p.B3G_UP_ARROW})) - int(bool(held & {ord("s"), p.B3G_DOWN_ARROW})),
      int(bool(held & {ord("d"), p.B3G_RIGHT_ARROW})) - int(bool(held & {ord("a"), p.B3G_LEFT_ARROW})),
      int(ord("e") in held) - int(ord("q") in held),
    ])

    # 相机固连在末端，用机械臂自身位姿渲染实际 RGB；不读取吊物真值控制机械臂。
    state = p.getLinkState(robot, tip, computeForwardKinematics=True)
    R = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    eye = np.asarray(state[4]) + R @ camera_offset
    p.resetBasePositionAndOrientation(camera_body, eye, state[5])
    view = p.computeViewMatrix(eye, eye + R[:, 2], -R[:, 1])
    rgba = p.getCameraImage(width, height, view, projection, renderer=p.ER_TINY_RENDERER,
      shadow=0, lightAmbientCoeff=0.9, lightDiffuseCoeff=0.1, lightSpecularCoeff=0,
      flags=p.ER_NO_SEGMENTATION_MASK)[2]
    if not isinstance(rgba, np.ndarray):
      rgba = np.frombuffer(bytes(rgba), dtype=np.uint8)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    q = np.array([s[0] for s in p.getJointStates(robot, movable)])
    previous_velocity = joint_velocity.copy()
    joint_velocity = np.zeros(7)
    status, error_px = "TARGET LOST - move crane back into view", float("nan")

    if ids is not None and 0 in ids.ravel():
      pixels = corners[int(np.flatnonzero(ids.ravel() == 0)[0])].reshape(4, 2).astype(np.float64)
      normalized = (pixels - K[:2, 2]) / focal
      if desired is None:
        desired = normalized.copy()
      # ITERATIVE 对接近正面的平面标记也稳定；PnP 这里只提供四角点的深度。
      ok, rvec, tvec = cv2.solvePnP(marker_points, pixels, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
      if ok:
        points = marker_points @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()
        tag_world = eye + R @ tvec.ravel()
        error = (normalized - desired).ravel()
        error_px = float(np.sqrt(np.mean(error**2)) * focal)
        in_range = np.all(tag_world >= track_min) and np.all(tag_world <= track_max)
        status = "TRACKING" if in_range else "OUT OF RANGE - move crane back"
        if in_range and np.isfinite(points).all() and np.min(points[:, 2]) > 0.05:
          rows = []
          for (x, y), Z in zip(normalized, points[:, 2]):
            rows.extend(([-1/Z, 0, x/Z, x*y, -(1+x*x), y],
                         [0, -1/Z, y/Z, 1+y*y, -x*y, -x]))
          L = np.asarray(rows)  # 8×6：相机速度 → 四角点的归一化图像速度。
          zeros = [0.] * len(q)
          linear, angular = p.calculateJacobian(robot, tip, camera_offset, q.tolist(), zeros, zeros)
          J = np.vstack((R.T @ np.asarray(linear)[:, :7], R.T @ np.asarray(angular)[:, :7]))
          A = L @ J             # 8×7：关节速度 → 图像速度。
          inverse = A.T @ np.linalg.solve(A @ A.T + damping**2 * np.eye(8), np.eye(8))
          joint_velocity = -gain * inverse @ error
          joint_velocity += (np.eye(7) - inverse @ A) @ (0.08 * (home - q[:7]))
          joint_velocity *= min(1., max_joint_speed / max(np.max(np.abs(joint_velocity)), 1e-9))
          joint_velocity = np.clip(joint_velocity, np.minimum(0, (lower - q[:7]) / frame_dt),
                                   np.maximum(0, (upper - q[:7]) / frame_dt))
    if paused:
      status, joint_velocity = "PAUSED - Space to resume", np.zeros(7)
    if not np.any(joint_velocity) and np.any(previous_velocity):
      arm_target = q[:7].copy()  # 丢失、越界或暂停时，只捕获一次保持位置。
    if status != last_status:
      print(status, flush=True)
      if gui:
        color = [0.1, 0.55, 0.2] if status == "TRACKING" else [0.8, 0.25, 0.1]
        status_id = p.addUserDebugText(status, [-0.8, 0, 1.49], color, 1.1, replaceItemUniqueId=status_id)
      last_status = status

    for _ in range(steps_per_frame):
      crane_velocity += np.clip(direction * crane_speed - crane_velocity, -0.25 * dt, 0.25 * dt)
      proposed = crane_target + crane_velocity * dt
      crane_target = np.clip(proposed, crane_lower, crane_upper)
      crane_velocity[proposed != crane_target] = 0
      p.setJointMotorControlArray(crane, crane_joints, p.POSITION_CONTROL,
        targetPositions=crane_target, targetVelocities=crane_velocity, forces=[200, 200, 80],
        positionGains=[0.5] * 3, velocityGains=[1.] * 3)
      actual = np.array([s[0] for s in p.getJointStates(robot, arm_joints)])
      arm_target = np.clip(arm_target + joint_velocity * dt, actual - 0.10, actual + 0.10)
      arm_target = np.clip(arm_target, lower, upper)
      p.setJointMotorControlArray(robot, arm_joints, p.POSITION_CONTROL,
        targetPositions=arm_target, targetVelocities=joint_velocity, forces=[87, 87, 87, 87, 12, 12, 12],
        positionGains=[0.3] * 7, velocityGains=[1.] * 7)
      p.stepSimulation()

    # 更新相机外壳和绳子的显示。不在循环里反复创建 debug line/text。
    state = p.getLinkState(robot, tip, computeForwardKinematics=True)
    R = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    p.resetBasePositionAndOrientation(camera_body, np.asarray(state[4]) + R @ camera_offset, state[5])
    hook = p.getLinkState(crane, crane_joints[2], computeForwardKinematics=True)
    R_hook = np.asarray(p.getMatrixFromQuaternion(hook[5])).reshape(3, 3)
    cable_length = p.getJointState(crane, crane_joints[2])[0]
    p.resetBasePositionAndOrientation(cable, np.asarray(hook[4]) + R_hook[:, 2] * segment_length / 2, hook[5])
    p.resetJointStatesMultiDof(cable, list(range(cable_count)),
                              [[(cable_length - segment_length) / cable_count]] * cable_count)
    frame += 1
    if gui:
      # 每轮更新完成后只绘制一次，避免相机渲染与主窗口持续重绘争用时间。
      p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
      deadline += frame_dt
      delay = deadline - time.perf_counter()
      if delay > 0:
        time.sleep(delay)
      elif delay < -0.1:
        deadline = time.perf_counter()
except p.error:
  # 原生窗口可能在一次 Bullet 调用中途关闭；真正的仿真错误仍直接报出。
  if p.isConnected(client):
    raise
finally:
  if p.isConnected(client):
    p.disconnect(client)
