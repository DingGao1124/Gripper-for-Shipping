"""Visual readiness and atomic transfer of the load from hook to Panda."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from .crane import rail_state, release_hook
from .robot import magnet_pose, magnet_velocity, stop_arm
from .servo import docking_distance


@dataclass
class Task:
  phase: str = 'WAIT_LIFT'
  reason: str = ''
  distance: float = 0.45
  samples: deque = field(default_factory=deque)
  latch: int | None = None
  stable_ready: bool = False


def _clear(task, cfg, phase='TRACK', reason=''):
  task.phase, task.reason = phase, reason
  task.samples.clear()
  task.stable_ready = False
  task.distance = cfg.observation_distance


def update_task(task, observation, estimate, crane, robot, cfg, now, risk=''):
  if task.phase == 'HOLD':
    return
  if now - estimate.time > cfg.lost_seconds or estimate.valid_frames < cfg.recovery_frames:
    _clear(task, cfg, 'LOST', 'marker unavailable')
    return
  if risk:
    _clear(task, cfg, 'PAUSED', risk)
    return
  if crane.latch is None or estimate.position[2] < cfg.load_size / 2 + 0.08:
    _clear(task, cfg, 'WAIT_LIFT', 'manual hook / lift required')
    return
  if task.phase in ('WAIT_LIFT', 'LOST', 'PAUSED'):
    _clear(task, cfg)
  _, velocity = rail_state(crane)
  inside = np.linalg.norm(estimate.position[:2] - cfg.target_xy) <= cfg.target_radius
  height = cfg.capture_height[0] <= estimate.position[2] <= cfg.capture_height[1]
  stopped = np.max(np.abs(velocity)) < cfg.crane_stop_speed and not any(crane.held)
  slow = np.linalg.norm(estimate.velocity) < cfg.stable_speed
  if not (inside and height and stopped and slow):
    _clear(task, cfg, reason='transport / outside handoff conditions')
    return
  if task.phase == 'APPROACH':
    task.distance = max(docking_distance(cfg), task.distance - cfg.approach_speed * cfg.dt * cfg.control_steps)
    return
  task.phase, task.reason = 'SETTLE', 'waiting for small world-frame motion'
  # One independent sample per image, not per controller update.
  if not task.samples or observation.time != task.samples[-1][0]:
    task.samples.append((observation.time, estimate.position.copy()))
  while task.samples and now - task.samples[0][0] > cfg.stable_seconds + cfg.camera_steps * cfg.dt:
    task.samples.popleft()
  if task.samples and task.samples[-1][0] - task.samples[0][0] >= cfg.stable_seconds - 1e-6:
    positions = np.array([sample[1][:2] for sample in task.samples])
    if np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1)) <= cfg.stable_radius:
      task.stable_ready = True
      if np.linalg.norm(robot.base_velocity) < 0.005:
        task.phase, task.reason = 'APPROACH', 'aligning magnetic tool'


def try_handoff(task, observation, estimate, crane, robot, scene, cfg):
  if task.phase != 'APPROACH' or not observation.valid:
    return False
  tool = magnet_pose(robot, cfg)
  desired_pad = observation.world_load[:3, 3] + observation.world_load[:3, :3] @ np.asarray(cfg.pad_position)
  normal = observation.world_load[:3, :3] @ [0, 1, 0]
  angle = np.rad2deg(np.arccos(np.clip(np.dot(tool[:3, 2], -normal), -1, 1)))
  relative_speed = np.linalg.norm(magnet_velocity(robot, cfg) - estimate.velocity)
  if np.linalg.norm(tool[:3, 3] - desired_pad) > cfg.lock_distance or angle > cfg.lock_angle or relative_speed > cfg.lock_speed:
    return False
  # Only this final physical gate reads the true load pose and velocity.
  position, orientation = p.getBasePositionAndOrientation(scene.load)
  true_pad, _ = p.multiplyTransforms(position, orientation, cfg.pad_position, [0, 0, 0, 1])
  rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
  true_velocity, angular = p.getBaseVelocity(scene.load)
  pad_velocity = np.asarray(true_velocity) + np.cross(angular, rotation @ np.asarray(cfg.pad_position))
  true_angle = np.rad2deg(np.arccos(np.clip(np.dot(tool[:3, 2], -(rotation @ [0, 1, 0])), -1, 1)))
  if np.linalg.norm(tool[:3, 3] - true_pad) > cfg.lock_distance or true_angle > cfg.lock_angle:
    return False
  if np.linalg.norm(magnet_velocity(robot, cfg) - pad_velocity) > cfg.lock_speed:
    return False
  parent = p.getLinkState(robot.body, robot.hand, computeForwardKinematics=True)
  relative = p.multiplyTransforms(*p.invertTransform(parent[0], parent[1]), position, orientation)
  latch = p.createConstraint(robot.body, robot.hand, scene.load, -1, p.JOINT_FIXED,
    [0, 0, 0], relative[0], [0, 0, 0], parentFrameOrientation=relative[1])
  p.changeConstraint(latch, maxForce=150)
  # No simulation step occurs between establishing support and releasing the hook.
  release_hook(crane)
  task.latch, task.phase, task.reason = latch, 'HOLD', 'handoff complete'
  robot.base_command[:] = 0
  robot.base_velocity[:] = 0
  robot.hold = True
  stop_arm(robot)
  print('Handoff complete: Panda holds the load; crane hook released.')
  return True
