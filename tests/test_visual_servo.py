"""Verify the Jacobian chain and the procedural demo using actual rendered pixels."""

import numpy as np
import pybullet as p

from examples import visual_servo as demo


def test_interaction_and_wheel_jacobians_against_projection():
  pose = np.array([0.1, -0.05, 0.12])
  h = demo.MARKER_SIZE / 2
  marker = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
  rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
  world_points = marker @ rotation.T + [2.2, 0.25, demo.CAMERA_HEIGHT]

  def project(cart):
    c, s = np.cos(cart[2]), np.sin(cart[2])
    optical_rotation = np.array([[s, 0, c], [-c, 0, s], [0, -1, 0]])
    eye = np.r_[cart[:2], demo.CAMERA_HEIGHT] + demo.CAMERA_OFFSET * np.array([c, s, 0])
    points = (world_points - eye) @ optical_rotation
    return points, (points[:, :2] / points[:, 2, None]).ravel()

  points, _ = project(pose)
  analytic = demo.interaction_matrix(points) @ demo.J_CAMERA @ demo.J_WHEELS
  numerical = np.zeros((8, 2))
  eps = 1e-5
  for j in range(2):
    v, omega = demo.J_WHEELS @ np.eye(2)[j]
    displacement = np.array([v * np.cos(pose[2]), v * np.sin(pose[2]), omega]) * eps
    numerical[:, j] = (project(pose + displacement)[1] - project(pose - displacement)[1]) / (2 * eps)
  np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=1e-6)


def test_rgb_loop_converges_tracks_and_stops_on_marker_loss(monkeypatch):
  frame = -1
  poses, wheel_angles = [], [[], []]
  reset_pose, reset_joint = p.resetBasePositionAndOrientation, p.resetJointState

  def keyboard():
    nonlocal frame
    frame += 1
    if frame == 0:
      return {ord('A'): p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN}
    if frame == 30:
      return {ord('A'): p.KEY_WAS_RELEASED}
    if frame == 31:
      return {ord('A'): p.KEY_IS_DOWN}  # Delayed DOWN must not restart motion.
    if frame == 120:
      return {ord('w'): p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN}
    if frame == 180:
      return {ord('w'): p.KEY_WAS_RELEASED}
    return {}

  def move_body(body, position, orientation):
    if body == 1:  # Plane, cart, board are created in this order.
      poses.append([*position[:2], p.getEulerFromQuaternion(orientation)[2]])
    elif body == 2 and 360 <= frame < 370:
      position = [position[0], position[1] + 10, position[2]]
    reset_pose(body, position, orientation)

  def move_wheel(body, joint, angle):
    wheel_angles[joint].append(angle)
    reset_joint(body, joint, angle)

  monkeypatch.setattr(p, 'getKeyboardEvents', keyboard)
  monkeypatch.setattr(p, 'resetBasePositionAndOrientation', move_body)
  monkeypatch.setattr(p, 'resetJointState', move_wheel)
  result = demo.main(gui=False, seconds=14)
  assert result['detections'] == 410  # Ten genuinely out-of-view rendered images.
  assert np.linalg.norm(result['error']) < 0.025
  assert np.linalg.norm(result['error']) < result['initial_error'] / 8
  assert result['cart'][0] > 0.5
  assert abs(result['cart'][1]) > 0.03
  np.testing.assert_allclose(poses[360:371], np.tile(poses[360], (11, 1)), atol=1e-12)
  velocities = np.diff(poses, axis=0) / demo.DT
  assert np.max(np.linalg.norm(velocities[:, :2], axis=1)) <= demo.MAX_SPEED + 1e-9
  assert np.max(np.abs(velocities[:, 2])) <= demo.MAX_YAW_RATE + 1e-9
  assert np.max(np.abs(np.diff(wheel_angles, axis=1) / demo.DT)) <= demo.MAX_WHEEL_RATE + 1e-9
