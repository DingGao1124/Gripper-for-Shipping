"""Panda kinematics, joint motor commands and kinematic cart motion."""

from dataclasses import dataclass

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from .vision import pose


@dataclass
class Robot:
  body: int
  arm: list
  movable: list
  camera: int
  hand: int
  magnet: int
  lower: np.ndarray
  upper: np.ndarray
  home: np.ndarray
  targets: np.ndarray
  xy: np.ndarray
  base_velocity: np.ndarray
  base_command: np.ndarray
  joint_command: np.ndarray
  hold: bool = False
  base_inertial: np.ndarray | None = None


def link_pose(body, link):
  state = p.getLinkState(body, link, computeForwardKinematics=True)
  return pose(state[4], state[5])


def initialize_robot(scene, cfg):
  body = scene.panda
  names = {p.getJointInfo(body, i)[12].decode(): i for i in range(p.getNumJoints(body))}
  arm = [next(i for i in range(p.getNumJoints(body)) if p.getJointInfo(body, i)[1].decode() == f'panda_joint{j}')
         for j in range(1, 8)]
  movable = [i for i in range(p.getNumJoints(body)) if p.getJointInfo(body, i)[3] >= 0]
  lower = np.array([p.getJointInfo(body, i)[8] for i in arm])
  upper = np.array([p.getJointInfo(body, i)[9] for i in arm])
  home = np.array(cfg.initial_joints)
  for i, value in zip(arm, home):
    p.resetJointState(body, i, value)
  for i in movable[7:]:
    p.resetJointState(body, i, 0)
    p.setJointMotorControl2(body, i, p.POSITION_CONTROL, targetPosition=0, force=20)
  camera = names['camera']
  tag_position = np.array(cfg.marker_position) + [0, 0, cfg.load_size / 2]
  camera_position = tag_position + [0, cfg.observation_distance, cfg.initial_camera_rise]
  forward = tag_position - camera_position
  forward /= np.linalg.norm(forward)
  right = np.cross(forward, [0, 0, 1])
  right /= np.linalg.norm(right)
  rotation = np.column_stack((right, np.cross(forward, right), forward))
  quaternion = Rotation.from_matrix(rotation).as_quat()
  # Reset joints only during initialization; operational servoing uses motors.
  for _ in range(12):
    solution = p.calculateInverseKinematics(body, camera, camera_position, quaternion,
       maxNumIterations=200, residualThreshold=1e-6)
    for i, value, lo, hi in zip(arm, solution[:7], lower, upper):
      p.resetJointState(body, i, np.clip(value, lo + cfg.joint_margin, hi - cfg.joint_margin))
  targets = np.array([p.getJointState(body, i)[0] for i in arm])
  robot = Robot(body, arm, movable, camera, names['panda_hand'], names['magnet'], lower, upper,
                targets.copy(), targets.copy(), np.array([0., cfg.base_offset]), np.zeros(2),
                np.zeros(2), np.zeros(7))
  robot.base_inertial = np.asarray(p.getDynamicsInfo(robot.body, -1)[3])
  apply_joint_targets(robot)
  return robot


def camera_pose(robot):
  return link_pose(robot.body, robot.camera)


def camera_jacobian(robot):
  positions = [state[0] for state in p.getJointStates(robot.body, robot.movable)]
  zeros = [0.] * len(positions)
  linear, angular = p.calculateJacobian(robot.body, robot.camera, [0, 0, 0], positions, zeros, zeros)
  return np.vstack((np.asarray(linear)[:, :7], np.asarray(angular)[:, :7]))


def magnet_pose(robot, cfg):
  hand = link_pose(robot.body, robot.hand)
  result = hand.copy()
  result[:3, 3] += hand[:3, :3] @ np.asarray(cfg.magnet_tip)
  return result


def magnet_velocity(robot, cfg):
  camera = camera_pose(robot)
  qdot = np.array([state[1] for state in p.getJointStates(robot.body, robot.arm)])
  twist = camera_jacobian(robot) @ qdot
  tip = magnet_pose(robot, cfg)[:3, 3]
  return twist[:3] + np.cross(twist[3:], tip - camera[:3, 3]) + np.r_[robot.base_velocity, 0]


def apply_joint_targets(robot):
  p.setJointMotorControlArray(robot.body, robot.arm, p.POSITION_CONTROL, targetPositions=robot.targets,
    targetVelocities=robot.joint_command, forces=[87, 87, 87, 87, 12, 12, 12],
    positionGains=[0.25] * 7, velocityGains=[1.0] * 7)


def advance_robot(robot, scene, cfg):
  difference = robot.base_command - robot.base_velocity
  norm = np.linalg.norm(difference)
  if norm > cfg.base_acceleration * cfg.dt:
    difference *= cfg.base_acceleration * cfg.dt / norm
  robot.base_velocity += difference
  if robot.hold:
    robot.base_velocity[:] = 0
  robot.xy += robot.base_velocity * cfg.dt
  p.resetBasePositionAndOrientation(scene.cart, [*robot.xy, 0.18], [0, 0, 0, 1])
  # Bullet resets the inertial frame, whereas loadURDF positions the root link.
  p.resetBasePositionAndOrientation(robot.body, np.r_[robot.xy, cfg.base_height] + robot.base_inertial, [0, 0, 0, 1])
  for body, x, y in scene.wheels:
    p.resetBasePositionAndOrientation(body, [robot.xy[0] + x, robot.xy[1] + y, 0.09],
                                      p.getQuaternionFromEuler([np.pi / 2, 0, 0]))
  robot.targets = np.clip(robot.targets + robot.joint_command * cfg.dt,
                         robot.lower + cfg.joint_margin, robot.upper - cfg.joint_margin)
  apply_joint_targets(robot)


def stop_arm(robot):
  # Capture a hold target once; repeated recaptures let gravity drift accumulate.
  if np.any(robot.joint_command):
    robot.targets = np.array([state[0] for state in p.getJointStates(robot.body, robot.arm)])
  robot.joint_command[:] = 0


def obstacle_risk(robot, scene, cfg):
  clearance = cfg.obstacle_margin + np.linalg.norm(robot.base_velocity) ** 2 / (2 * cfg.base_acceleration)
  if p.getClosestPoints(robot.body, scene.crane, clearance):
    return 'crane clearance'
  if p.getClosestPoints(scene.cart, scene.crane, clearance):
    return 'cart clearance'
  if any(point[3] >= 0 for point in p.getClosestPoints(robot.body, scene.cart, cfg.obstacle_margin)):
    return 'arm / cart clearance'
  points = p.getClosestPoints(robot.body, scene.ground, cfg.obstacle_margin)
  if any(point[3] > 0 for point in points):
    return 'ground clearance'
  actual = np.array([state[0] for state in p.getJointStates(robot.body, robot.arm)])
  if np.max(np.abs(actual - robot.targets)) > 0.25:
    return 'joint tracking / reach limit'
  return ''
