"""Standalone four-wheel cart carrying a Franka Panda; no project imports.

Run: conda run -n robotics python examples/visual_servo_control/car.py
Click PyBullet: W/S forward/back, A/D turn, Space brake, R reset, Esc exit.
Arrow keys also work. Release movement keys to stop.
"""

import time

import numpy as np
import pybullet as p
import pybullet_data

DT = 1 / 60
WHEEL_RADIUS, TRACK = 0.09, 0.38  # Wheel radius and wheel separation [m].
BODY_HEIGHT, ARM_HEIGHT = 0.18, 0.25  # Body center and top mounting surface [m].
ARM_HOME = [0, -0.6, 0, -2.2, 0, 1.6, 0.8]
SPEED, YAW_RATE = 0.65, 1.25  # Keyboard commands [m/s, rad/s].

# Left/right wheel rates [rad/s] -> body forward speed and yaw rate.
J_WHEELS = WHEEL_RADIUS * np.array([[0.5, 0.5], [-1 / TRACK, 1 / TRACK]])


def main(gui=True, seconds=None):
  p.connect(p.GUI if gui else p.DIRECT)
  try:
    # 1. Independent copy of the src scene dimensions: box body and four wheels.
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setRealTimeSimulation(0)
    chassis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.20, 0.18, 0.07], rgbaColor=[0.15, 0.35, 0.5, 1])
    wheel = p.createVisualShape(
      p.GEOM_CYLINDER,
      radius=WHEEL_RADIUS,
      length=0.04,
      visualFrameOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]),
      rgbaColor=[0.12, 0.12, 0.12, 1],
    )
    robot_car = p.createMultiBody(
      baseMass=0,
      baseVisualShapeIndex=chassis,
      basePosition=[0, 0, BODY_HEIGHT],
      linkMasses=[0] * 4,
      linkCollisionShapeIndices=[-1] * 4,
      linkVisualShapeIndices=[wheel] * 4,
      # Front left/right, rear left/right. Wheels touch z=0.
      linkPositions=[[x, y, WHEEL_RADIUS - BODY_HEIGHT] for x in (0.16, -0.16) for y in (TRACK / 2, -TRACK / 2)],
      linkOrientations=[[0, 0, 0, 1]] * 4,
      linkInertialFramePositions=[[0, 0, 0]] * 4,
      linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
      linkParentIndices=[0] * 4,
      linkJointTypes=[p.JOINT_REVOLUTE] * 4,
      linkJointAxis=[[0, 1, 0]] * 4,
    )
    panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, ARM_HEIGHT], useFixedBase=True)
    for joint, angle in enumerate(ARM_HOME):
      p.resetJointState(panda, joint, angle)
    for joint in (9, 10):
      p.resetJointState(panda, joint, 0.03)
    # loadURDF takes the root link pose; resetBase takes its inertial frame pose.
    inertial_position, inertial_rotation = p.getDynamicsInfo(panda, -1)[3:5]

    if gui:
      p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
      p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
      p.resetDebugVisualizerCamera(3.0, -45, -30, [0, 0, 0.55])

    print("Click PyBullet: W/S forward/back, A/D turn; Space brake; R reset; Esc exit.")
    pose = np.zeros(3)  # World [x, y, yaw].
    wheel_angles = np.zeros(2)
    down, released = set(), set()
    frame = 0
    debug_text_id = p.addUserDebugText(
      text="Car with Frank Panda",
      textPosition=[0, 0, 1],
      textColorRGB=[1, 0, 0],
      textSize=1,
      lifeTime=0,  # 0 = 永久
      parentObjectUniqueId=robot_car,  # >=0 时挂到物体局部坐标系
      parentLinkIndex=-1,
      replaceItemUniqueId=-1,  # 传回上次返回的 id 可原地更新文字
    )
    deadline = time.perf_counter()
    while p.isConnected() and (seconds is None or frame < round(seconds / DT)):
      p.addUserDebugText(
        text="Car with Frank Panda",
        textPosition=[0, 0, 1],
        textColorRGB=[1, 0, 0],
        textSize=1,
        lifeTime=0,
        parentObjectUniqueId=robot_car,
        parentLinkIndex=-1,
        replaceItemUniqueId=debug_text_id,
      )
      # 2. Keep held keys across polls, normalize uppercase, ignore delayed DOWN events.
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
      if ord("r") in pressed:
        pose[:] = 0
        wheel_angles[:] = 0
      forward = bool(down & {ord("w"), p.B3G_UP_ARROW}) - bool(down & {ord("s"), p.B3G_DOWN_ARROW})
      turn = bool(down & {ord("a"), p.B3G_LEFT_ARROW}) - bool(down & {ord("d"), p.B3G_RIGHT_ARROW})
      v, omega = SPEED * forward, YAW_RATE * turn
      if ord(" ") in down or ord("r") in pressed:
        v, omega = 0.0, 0.0

      # 3. Ideal skid steering: front/rear wheels on each side share one rate.
      left_rate = (v - omega * TRACK / 2) / WHEEL_RADIUS
      right_rate = (v + omega * TRACK / 2) / WHEEL_RADIUS
      rates = np.array([left_rate, right_rate])
      v, omega = J_WHEELS @ rates

      # 4. Integrate the circular arc over DT (sinc also handles straight motion).
      angle = omega * DT
      travel = v * DT * np.sinc(angle / (2 * np.pi))
      heading = pose[2] + angle / 2
      pose += [travel * np.cos(heading), travel * np.sin(heading), angle]
      wheel_angles += rates * DT

      # Ideal kinematics: directly place the cart, without tire/contact dynamics.
      orientation = p.getQuaternionFromEuler([0, 0, pose[2]])
      p.resetBasePositionAndOrientation(robot_car, [*pose[:2], BODY_HEIGHT], orientation)
      for index in range(4):
        p.resetJointState(robot_car, index, wheel_angles[index % 2])
      # Carry the arm with the platform; its joint posture stays fixed in this demo.
      arm_position, arm_rotation = p.multiplyTransforms(
        [*pose[:2], ARM_HEIGHT], orientation, inertial_position, inertial_rotation
      )
      p.resetBasePositionAndOrientation(panda, arm_position, arm_rotation)
      frame += 1
      if gui:
        deadline += DT
        remaining = deadline - time.perf_counter()
        if remaining > 0:
          time.sleep(remaining)
        elif remaining < -0.1:
          deadline = time.perf_counter()
    return pose
  except KeyboardInterrupt:
    pass
  except p.error:
    if p.isConnected():
      raise
  finally:
    if p.isConnected():
      p.disconnect()


if __name__ == "__main__":
  main()
