"""URDF 龙门吊：完全由滑条和键盘操作，Matplotlib 显示 X/Y 滑轨实际速度。

运行：conda run -n robotics python examples/crane.py
W/S 或 ↑/↓：门架 Y；A/D 或 ←/→：小车 X；Q/E：升钩/降钩。
F：就近挂住方块；R：脱钩；H：显示/隐藏滑条面板与世界坐标轴。
C：保存当前视角的 RGB、深度和分割数据。
钩子通过固定约束连接独立刚体，挂载后仍保留两个自由摆动关节。
"""

import math
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

DT = 1 / 240
FRAME_STEPS = 4  # Simulation Steps per Frame
TABLE_TOP_Z = 0.625
BLOCK_HALF = 0.025
CRANE_BASE = [0.0, 0.0, 0.0]
GIRDER_Z = 1.32
L_RETRACT = 0.18
L_GRASP = GIRDER_Z - (TABLE_TOP_Z + BLOCK_HALF + 0.04)
VMAX_TRAVEL = 0.45  # X/Y 最大速度，m/s
ACCEL_TRAVEL = 1.2  # X/Y 加减速度，m/s²
VMAX_HOIST = 0.25  # Z 起升最大速度，m/s
ACCEL_HOIST = 0.6  # Z 加减速度，m/s²
TRAVEL_FORCE = 80.0
HOIST_FORCE = 40.0

_URDF = Path(__file__).resolve().parents[1] / "assets" / "gantry_crane.urdf"
_CAPTURE_DIR = Path(__file__).resolve().parents[1] / "tmp" / "captures"

def _advance(position, velocity, target, speed, acceleration):
  """手动控制的限速、限加速度参考，逐物理步更新。"""
  diff = target - position
  desired = math.copysign(min(speed, math.sqrt(2 * acceleration * abs(diff)), 4 * abs(diff)), diff)
  velocity += max(-acceleration * DT, min(acceleration * DT, desired - velocity))
  next_position = position + velocity * DT
  # 最后一个步长落在目标上，避免在极限位置越界。
  if diff * (target - next_position) <= 0:
    return target, 0.0
  return next_position, velocity


def _key_down(keys, *codes):
  return any(keys.get(code, 0) & p.KEY_IS_DOWN for code in codes)


def main(gui=True, max_frames=None):
  """GUI 默认持续运行；DIRECT 默认运行 60 帧，供无输入的场景检查。"""
  if not gui and max_frames is None:
    max_frames = 60
  fig = None
  position = None
  client = p.connect(p.GUI if gui else p.DIRECT)
  if client < 0:
    raise RuntimeError("PyBullet 连接失败")
  try:
    # 场景与模型直接展开；所有关节按 URDF 名称查找。
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(DT)
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", CRANE_BASE)
    crane = p.loadURDF(str(_URDF), CRANE_BASE, useFixedBase=True)
    joints = {p.getJointInfo(crane, i)[1].decode(): i for i in range(p.getNumJoints(crane))}
    names = ("gantry_y", "trolley_x", "hoist")
    actuated = [joints[name] for name in names]
    swing = [joints["sway_x"], joints["sway_y"]]
    limits = [p.getJointInfo(crane, joint)[8:10] for joint in actuated]
    speeds = (VMAX_TRAVEL, VMAX_TRAVEL, VMAX_HOIST)
    accelerations = (ACCEL_TRAVEL, ACCEL_TRAVEL, ACCEL_HOIST)
    forces = (TRAVEL_FORCE, TRAVEL_FORCE, HOIST_FORCE)
    for joint in swing:
      p.setJointMotorControl2(crane, joint, p.VELOCITY_CONTROL, force=0)
      p.changeDynamics(crane, joint, linearDamping=0.02, angularDamping=0.04, jointDamping=0.01)
    p.changeDynamics(crane, joints["hoist"], linearDamping=0.02, angularDamping=0.04)

    # The Item Block
    pick = [CRANE_BASE[0], CRANE_BASE[1], TABLE_TOP_Z + BLOCK_HALF]
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[BLOCK_HALF] * 3)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[BLOCK_HALF] * 3, rgbaColor=[0.92, 0.42, 0.15, 1])
    block = p.createMultiBody(
      baseMass=0.05, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=pick
    )
    p.changeDynamics(block, -1, lateralFriction=2.0)
    # 门架与小车均从 URDF 零位启动，吊钩和方块位于龙门中心。
    targets = [0.0, 0.0, L_RETRACT]
    for joint, target in zip(actuated, targets):
      p.resetJointState(crane, joint, target)
    refs = [(target, 0.0) for target in targets]
    latch = None
    cable_id = -1

    panel_visible = False
    if gui:
      # PyBullet 的世界坐标轴与原生 GUI 共用开关；默认隐藏，H 可打开滑条。
      p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
      p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
      p.resetDebugVisualizerCamera(2.0, 48, -32, [CRANE_BASE[0], CRANE_BASE[1], 0.90])
      sliders = [
        p.addUserDebugParameter(label, *bounds, target)
        for label, bounds, target in zip(("gantry Y [m]", "trolley X [m]", "cable length [m]"), limits, targets)
      ]
      print("手动控制已就绪。点击 PyBullet 窗口后按键；H 显示/隐藏滑条面板。")
      print("W/S 或 ↑↓：门架 Y；A/D 或 ←→：小车 X；Q/E：升/降；F：挂载；R：脱钩。")
      print("C：采集当前视角的 RGB、深度和分割数据，保存至 captures/。")
    last_sliders = list(targets)
    held = [False] * 3
    braking = [False] * 3
    slider_warning = False

    def attach():
      nonlocal latch
      if latch is not None:
        return True
      hook_state = p.getLinkState(crane, joints["hoist"], computeForwardKinematics=1)
      block_pos, block_orn = p.getBasePositionAndOrientation(block)
      # 以钩尖而非小车位置判断距离，禁止远距离吸附。
      tip, _ = p.multiplyTransforms(hook_state[4], hook_state[5], [0, 0, -0.04], [0, 0, 0, 1])
      if math.dist(tip, block_pos) > BLOCK_HALF + 0.035:
        print("挂载失败：请把钩尖移到方块附近，再按 F。")
        return False
      # createConstraint 的父坐标系是连杆质心；保留当前相对姿态，不瞬移方块。
      rel_pos, rel_orn = p.multiplyTransforms(*p.invertTransform(hook_state[0], hook_state[1]), block_pos, block_orn)
      latch = p.createConstraint(
        crane,
        joints["hoist"],
        block,
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        rel_pos,
        [0, 0, 0],
        parentFrameOrientation=rel_orn,
        childFrameOrientation=[0, 0, 0, 1],
      )
      p.changeConstraint(latch, maxForce=300)
      print("已挂住方块。")
      return True

    # 曲线采样 60 Hz，绘图刷新 10 Hz，避免每个物理步重绘拖慢仿真。
    speed_t = deque(maxlen=480)
    speed_x = deque(maxlen=480)
    speed_y = deque(maxlen=480)
    frame_count = 0
    position = tuple(pick)
    if gui:
      plt.ion()
      fig, ax = plt.subplots(figsize=(7, 3.2))
      ax.set_title("Rail velocity")
      ax.set_xlabel("simulation time [s]")
      ax.set_ylabel("velocity [m/s]")
      ax.set_ylim(-VMAX_TRAVEL * 1.3, VMAX_TRAVEL * 1.3)
      ax.set_xlim(0, 8)
      ax.axhline(0, color="0.6", linewidth=0.8)
      ax.grid(True, alpha=0.3)
      (line_x,) = ax.plot([], [], color="#c43c3c", label="X / trolley")
      (line_y,) = ax.plot([], [], color="#2a6fdb", label="Y / gantry")
      ax.legend(loc="upper right")
      fig.tight_layout()
      plt.show(block=False)

    while p.isConnected(client) and (max_frames is None or frame_count < max_frames):
      frame_start = time.perf_counter()
      if gui:
        read_recovered = slider_warning
        sliders_valid = True
        try:
          sliders_now = [p.readUserDebugParameter(i) for i in sliders]
          slider_warning = False
        except p.error:
          if not p.isConnected(client):
            break
          sliders_valid = False
          sliders_now = list(last_sliders)
          if panel_visible and not slider_warning:
            print("滑条暂时无法读取，保留上次目标；键盘控制仍可用。")
          slider_warning = True

        # KeyBoard Events
        keys: dict = p.getKeyboardEvents()
        if any(keys.get(code, 0) & p.KEY_WAS_TRIGGERED for code in (ord("c"), ord("C"))):
          try:
            folder = capture_current_view(_CAPTURE_DIR)
            print(f"相机数据已保存：{folder}")
          except (OSError, ValueError, p.error) as exc:
            if not p.isConnected(client):
              raise
            print(f"相机采集失败：{exc}")
        panel_toggled = any(keys.get(code, 0) & p.KEY_WAS_TRIGGERED for code in (ord("h"), ord("H")))
        if panel_toggled:
          panel_visible = not panel_visible
          p.configureDebugVisualizer(p.COV_ENABLE_GUI, int(panel_visible))

        directions = [
          int(_key_down(keys, ord("w"), ord("W"), p.B3G_UP_ARROW)) - int(_key_down(keys, ord("s"), ord("S"), p.B3G_DOWN_ARROW)),
          int(_key_down(keys, ord("d"), ord("D"), p.B3G_RIGHT_ARROW)) - int(_key_down(keys, ord("a"), ord("A"), p.B3G_LEFT_ARROW)),
          int(_key_down(keys, ord("e"), ord("E"))) - int(_key_down(keys, ord("q"), ord("Q"))),
        ]
        # 隐藏面板、刚恢复读取、切换面板时只同步读值基线，不能下发旧目标。
        accept_sliders = panel_visible and sliders_valid and not read_recovered and not panel_toggled
        changed = [accept_sliders and abs(a - b) > 1e-4 for a, b in zip(sliders_now, last_sliders)]
        grab = any(keys.get(code, 0) & p.KEY_WAS_TRIGGERED for code in (ord("f"), ord("F")))
        release = any(keys.get(code, 0) & p.KEY_WAS_TRIGGERED for code in (ord("r"), ord("R")))
        for i, direction in enumerate(directions):
          if changed[i]:
            targets[i] = sliders_now[i]
            braking[i] = False
          if direction:
            targets[i] = limits[i][1 if direction > 0 else 0]
            held[i] = True
            braking[i] = False
          elif held[i]:
            # 松键后保留速度，沿当前方向减速；停止后再锁定位置。
            braking[i] = not changed[i]
            held[i] = False
        if release and latch is not None:
          p.removeConstraint(latch)
          latch = None
          print("已脱钩。")
        elif grab:
          attach()
        last_sliders = sliders_now

      # Physical Simulation
      for _ in range(FRAME_STEPS):
        for i, joint in enumerate(actuated):
          targets[i] = max(limits[i][0], min(limits[i][1], targets[i]))
          if braking[i]:
            position_ref, velocity_ref = refs[i]
            next_velocity = math.copysign(max(0.0, abs(velocity_ref) - accelerations[i] * DT), velocity_ref)
            next_position = position_ref + (velocity_ref + next_velocity) * DT / 2
            bounded_position = max(limits[i][0], min(limits[i][1], next_position))
            if bounded_position != next_position:
              next_velocity = 0.0
            refs[i] = (bounded_position, next_velocity)
            targets[i] = bounded_position
            if next_velocity == 0:
              # 刹停后锁定实际位置，避免追赶参考位置造成微小反向纠正。
              targets[i] = p.getJointState(crane, joint)[0]
              refs[i] = (targets[i], 0.0)
              braking[i] = False
          else:
            refs[i] = _advance(*refs[i], targets[i], speeds[i], accelerations[i])
          p.setJointMotorControl2(
            crane,
            joint,
            p.POSITION_CONTROL,
            targetPosition=refs[i][0],
            targetVelocity=refs[i][1],
            force=forces[i],
            positionGain=0.5,
            velocityGain=1.0,
          )
        for joint in swing:
          p.setJointMotorControl2(crane, joint, p.VELOCITY_CONTROL, force=0)
        p.stepSimulation()

      frame_count += 1
      position = p.getBasePositionAndOrientation(block)[0]
      if gui:
        top = p.getLinkState(crane, joints["trolley_x"], computeForwardKinematics=1)[4]
        bottom = p.getLinkState(crane, joints["hoist"], computeForwardKinematics=1)[4]
        cable_id = p.addUserDebugLine(top, bottom, [0.95, 0.78, 0.25], lineWidth=10, replaceItemUniqueId=cable_id)
        if fig is not None and plt.fignum_exists(fig.number):
          elapsed = frame_count * FRAME_STEPS * DT
          speed_t.append(elapsed)
          # getJointState 的第二项是实际关节速度，不是控制目标速度。
          speed_x.append(p.getJointState(crane, joints["trolley_x"])[1])
          speed_y.append(p.getJointState(crane, joints["gantry_y"])[1])
          if frame_count % 6 == 0:
            line_x.set_data(list(speed_t), list(speed_x))
            line_y.set_data(list(speed_t), list(speed_y))
            ax.set_xlim(max(0, elapsed - 8), max(8, elapsed))
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        # 60 Hz 刷新 UI，240 Hz 物理；扣除计算时间，避免逐步 sleep 累积变慢。
        delay = FRAME_STEPS * DT - (time.perf_counter() - frame_start)
        if delay > 0:
          time.sleep(delay)
    return position
  except p.error as exc:
    # 关闭 GUI 可能发生在 isConnected 检查与下一次调用之间。
    if not p.isConnected(client) or "Not connected to physics server" in str(exc):
      return position
    raise
  except KeyboardInterrupt:
    return position
  finally:
    if fig is not None:
      plt.close(fig)
    if p.isConnected(client):
      p.disconnect(client)


def capture_current_view(directory, renderer=p.ER_BULLET_HARDWARE_OPENGL):
  camera = p.getDebugVisualizerCamera()
  width, height = camera[:2]
  view, projection = camera[2:4]
  if width <= 0 or height <= 0:
    raise ValueError("无法获取相机分辨率，请在 GUI 模式下采集。")

  _, _, rgba, depth, mask = p.getCameraImage(
    width,
    height,
    viewMatrix=view,
    projectionMatrix=projection,
    renderer=renderer,
    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
  )
  rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
  depth = np.asarray(depth, dtype=np.float32).reshape(height, width)
  mask = np.asarray(mask, dtype=np.int32).reshape(height, width)

  # OpenGL 透视投影：深度缓冲为非线性的 [0, 1]，转换为相机 Z 深度（米）。
  a, b = projection[10], projection[14]
  near, far = b / (a - 1), b / (a + 1)
  depth_m = far * near / (far - (far - near) * depth)
  # 背景像素不代表真实表面，标为 NaN；分割背景继续保留 -1。
  depth_m[mask == -1] = np.nan

  folder = Path(directory) / str(time.time_ns())
  folder.mkdir(parents=True, exist_ok=False)
  Image.fromarray(rgb).save(folder / "rgb.png")
  np.save(folder / "depth_m.npy", depth_m)
  np.save(folder / "segmentation.npy", mask)
  np.savez(
    folder / "camera.npz",
    width=width,
    height=height,
    view=np.asarray(view),
    projection=np.asarray(projection),
    near=near,
    far=far,
  )
  return folder


if __name__ == "__main__":
  main()
