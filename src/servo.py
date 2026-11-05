"""Point-feature IBVS with camera-frame robot Jacobian and base compensation."""

import numpy as np
from scipy.spatial.transform import Rotation

from .robot import camera_jacobian, camera_pose
from .vision import camera_parameters, marker_points, marker_to_load


def limited(vector, maximum):
  norm = np.linalg.norm(vector)
  return vector * min(1., maximum / max(norm, 1e-12))


def damped_inverse(matrix, damping):
  return matrix.T @ np.linalg.solve(matrix @ matrix.T + damping ** 2 * np.eye(matrix.shape[0]),
                                   np.eye(matrix.shape[0]))


def interaction_matrix(points):
  rows = []
  for X, Y, Z in points:
    x, y = X / Z, Y / Z
    rows.extend(([-1/Z, 0, x/Z, x*y, -(1+x*x), y],
                 [0, -1/Z, y/Z, 1+y*y, -x*y, -x]))
  return np.array(rows)


def _docking_pose(cfg):
  # Desired tag pose in the tilted camera when the tool touches the pad.
  facing = np.diag([1., -1., -1.])
  tag_load = marker_to_load(cfg)
  pad_marker = tag_load[:3, :3] @ np.asarray(cfg.pad_position) + tag_load[:3, 3]
  mount = np.asarray(cfg.camera_mount) - np.asarray(cfg.magnet_tip)
  rotation = (facing @ Rotation.from_euler('xyz', cfg.camera_rpy).as_matrix()).T
  return rotation, -rotation @ (pad_marker + facing @ mount)


def docking_distance(cfg):
  return _docking_pose(cfg)[1][2]


def desired_features(distance, cfg):
  rotation, translation = _docking_pose(cfg)
  # Change distance along the pad normal, preserving the tool's docking alignment.
  retreat = -rotation[:, 2]
  translation += retreat * ((distance - translation[2]) / retreat[2])
  points = marker_points(cfg) @ rotation.T + translation
  return (points[:, :2] / points[:, 2, None]).ravel()


def base_follow(robot, estimate, cfg):
  # Lead the slow centroid during transport; the arm handles remaining swing.
  desired = estimate.slow_position[:2] + 0.20 * estimate.tracking_velocity[:2] + [0, cfg.base_offset]
  return limited(2 * (desired - robot.xy) + estimate.tracking_velocity[:2] * 0.7, cfg.base_speed)


def visual_servo(robot, observation, estimate, distance, cfg):
  intrinsic = camera_parameters(cfg)
  current = ((observation.corners - intrinsic[:2, 2]) / intrinsic[0, 0]).ravel()
  error = current - desired_features(distance, cfg)
  points = marker_points(cfg) @ observation.camera_tag[:3, :3].T + observation.camera_tag[:3, 3]
  L = interaction_matrix(points)
  rotation = camera_pose(robot)[:3, :3].T
  transform = np.zeros((6, 6))
  transform[:3, :3] = transform[3:, 3:] = rotation
  J = transform @ camera_jacobian(robot)
  feedback = -cfg.servo_gain * damped_inverse(L, 0.005) @ error
  feedback[:3] = limited(feedback[:3], cfg.camera_speed)
  feedback[3:] = limited(feedback[3:], cfg.camera_angular_speed)
  # Moving-target translation and cart motion are expressed in camera coordinates.
  feedback[:3] += rotation @ (estimate.tracking_velocity - np.r_[robot.base_velocity, 0])
  A = L @ J
  pinv = damped_inverse(A, cfg.servo_damping)
  command = pinv @ (L @ feedback)
  actual = robot.targets
  center = (robot.lower + robot.upper) / 2
  posture = cfg.home_gain * (robot.home - actual)
  proximity = np.maximum(0., cfg.joint_margin * 2 - np.minimum(actual - robot.lower, robot.upper - actual))
  posture += 0.4 * np.sign(center - actual) * proximity
  command += (np.eye(7) - pinv @ A) @ posture
  command = np.clip(command, -cfg.joint_speed, cfg.joint_speed)
  command = np.where((actual < robot.lower + cfg.joint_margin + 0.01) & (command < 0), 0, command)
  command = np.where((actual > robot.upper - cfg.joint_margin - 0.01) & (command > 0), 0, command)
  return command, float(np.sqrt(np.mean(error ** 2)) * intrinsic[0, 0])
