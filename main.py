import time

import numpy as np
import pybullet as p
import pybullet_data

DT = 1 / 240

TABLE_TOP_Z = 0.625
ROBOT_BASE_Z = 0.630

ARM_JOINTS = [1, 2, 3, 4, 5, 6]
DRIVE_JOINT = 8

MIMIC_JOINTS = {
    9: [1, 0, 0],
    10: [-1, 0, 0],
    11: [1, 0, 0],
    12: [-1, 0, 0],
    13: [1, 0, 0],
}
END_EFFECTOR = 6

FINGER_OFFSET = 0.17
OPEN_ANGLE = 0.0
CLOSE_ANGLE = 0.55

HOME = [0.0, -0.5, -0.7, 0.0, 1.3, 0.0]
BLOCK_HALF = 0.025


def main(gui=True, hold=True):
  client = p.connect(p.GUI if gui else p.DIRECT)
  try:
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(DT)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=150)
    if gui:
      p.resetDebugVisualizerCamera(
        cameraDistance=1.35,
        cameraYaw=48,
        cameraPitch=-32,
        cameraTargetPosition=[0.45, 0.0, 0.75],
      )

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.55, 0.0, 0.0])
    robot = _load_robot()
    block = _load_block([0.42, 0.0, TABLE_TOP_Z + BLOCK_HALF])

    pick = [0.42, 0.0, TABLE_TOP_Z + BLOCK_HALF]
    place = [0.58, 0.0, TABLE_TOP_Z + BLOCK_HALF]
    _move(robot, _above(pick, 0.12), OPEN_ANGLE, 1.2, gui, grip_force=20)
    _move(robot, pick, OPEN_ANGLE, 1.0, gui, grip_force=20)
    _move(robot, pick, CLOSE_ANGLE, 0.7, gui, grip_force=15)
    _move(robot, _above(pick, 0.15), CLOSE_ANGLE, 1.0, gui, grip_force=20)
    _move(robot, _above(place, 0.15), CLOSE_ANGLE, 1.4, gui, grip_force=20)
    _move(robot, _above(place, 0.01), CLOSE_ANGLE, 1.0, gui, grip_force=20)
    _move(robot, _above(place, 0.01), OPEN_ANGLE, 0.5, gui, grip_force=20)
    _move(robot, _above(place, 0.15), OPEN_ANGLE, 0.8, gui, grip_force=20)

    position = p.getBasePositionAndOrientation(block)[0]
    print(
      "方块放下后的位置: "
      f"x={position[0]:.3f} m, y={position[1]:.3f} m, z={position[2]:.3f} m"
    )

    if gui and hold:
      print("夹取完成。关闭仿真窗口即可退出。")
      while p.isConnected():
        p.stepSimulation()
        time.sleep(DT)
    return position
  except KeyboardInterrupt:
    raise
  finally:
    if p.isConnected(client):
      p.disconnect(client)


def _load_robot():
  robot = p.loadURDF(
    "xarm/xarm6_with_gripper.urdf",
    [0.0, 0.0, ROBOT_BASE_Z],
    useFixedBase=True,
  )
  for joint_id, position in zip(ARM_JOINTS, HOME):
    p.resetJointState(robot, joint_id, position)
  for joint_id, axis in MIMIC_JOINTS.items():
    constraint = p.createConstraint(
      robot,
      DRIVE_JOINT,
      robot,
      joint_id,
      p.JOINT_GEAR,
      jointAxis=axis,
      parentFramePosition=[0, 0, 0],
      childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(constraint, gearRatio=-1, erp=0.8, maxForce=500)
    p.setJointMotorControl2(robot, joint_id, p.VELOCITY_CONTROL, force=0)  # passive; a motor here fights the gear
    p.changeDynamics(robot, joint_id, lateralFriction=2.0)
  p.changeDynamics(robot, DRIVE_JOINT, lateralFriction=2.0)
  return robot


def _load_block(position):
  collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[BLOCK_HALF] * 3)
  visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[BLOCK_HALF] * 3,
    rgbaColor=[0.92, 0.42, 0.15, 1.0],
  )
  block = p.createMultiBody(
    baseMass=0.05,
    baseCollisionShapeIndex=collision,
    baseVisualShapeIndex=visual,
    basePosition=position,
  )
  p.changeDynamics(block, -1, lateralFriction=2.0)
  return block


def _above(point, height):
  return [point[0], point[1], point[2] + height]


def _inverse_kinematics(robot, tcp):
  link_position = [tcp[0], tcp[1], tcp[2] + FINGER_OFFSET]
  orientation = p.getQuaternionFromEuler([0.0, np.pi, 0.0])  # gripper +z points down
  solution = p.calculateInverseKinematics(
    robot,
    END_EFFECTOR,
    link_position,
    orientation,
    lowerLimits=[-6.2, -2.0, -3.9, -6.2, -1.6, -6.2],
    upperLimits=[6.2, 2.0, 0.15, 6.2, 3.1, 6.2],
    jointRanges=[12.4, 4.0, 4.05, 12.4, 4.7, 12.4],
    restPoses=HOME,
    maxNumIterations=200,
    residualThreshold=1e-4,
  )
  return np.array(solution[:6])


def _move(robot, tcp, gripper, seconds, gui, grip_force):
  start = np.array([p.getJointState(robot, joint_id)[0] for joint_id in ARM_JOINTS])
  goal = _inverse_kinematics(robot, tcp)
  steps = max(1, int(seconds / DT))
  for step in range(steps):
    alpha = (step + 1) / steps
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    target = start * (1.0 - alpha) + goal * alpha
    for joint_id, position in zip(ARM_JOINTS, target):
      p.setJointMotorControl2(
        robot,
        joint_id,
        p.POSITION_CONTROL,
        targetPosition=float(position),
        force=250,
        positionGain=0.04,
      )
    p.setJointMotorControl2(
      robot,
      DRIVE_JOINT,
      p.POSITION_CONTROL,
      targetPosition=gripper,
      force=grip_force,
    )
    p.stepSimulation()
    if gui:
      # Do some delay here to see every step.
      time.sleep(DT/10)


if __name__ == "__main__":
  main()
