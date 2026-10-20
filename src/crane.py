"""Manual crane references and proximity-checked physical hook attachment."""

from dataclasses import dataclass, field
import math

import numpy as np
import pybullet as p

from .robot import link_pose
from .keyboard import KeyboardState, update_keyboard


@dataclass
class Crane:
  body: int
  joints: dict
  actuated: list
  limits: np.ndarray
  targets: np.ndarray
  positions: np.ndarray
  velocities: np.ndarray
  braking: list = field(default_factory=lambda: [False] * 3)
  held: list = field(default_factory=lambda: [False] * 3)
  latch: int | None = None
  panel_visible: bool = False
  keyboard: KeyboardState = field(default_factory=KeyboardState)


def initialize_crane(scene, gui):
  names = {p.getJointInfo(scene.crane, i)[1].decode(): i for i in range(p.getNumJoints(scene.crane))}
  active = [names[name] for name in ('gantry_y', 'trolley_x', 'hoist')]
  limits = np.array([p.getJointInfo(scene.crane, i)[8:10] for i in active])
  initial = np.array([0., 0., 0.18])
  for i, value in zip(active, initial):
    p.resetJointState(scene.crane, i, value)
  for name in ('sway_x', 'sway_y'):
    p.setJointMotorControl2(scene.crane, names[name], p.VELOCITY_CONTROL, force=0)
    p.changeDynamics(scene.crane, names[name], jointDamping=0.025, angularDamping=0.04)
  crane = Crane(scene.crane, names, active, limits, initial.copy(), initial.copy(), np.zeros(3))
  if gui:
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    for preview in (p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
      p.configureDebugVisualizer(preview, 0)
  return crane


def set_targets(crane, values):
  crane.targets[:] = np.clip(values, crane.limits[:, 0], crane.limits[:, 1])
  crane.braking[:] = [False] * 3


def release_hook(crane):
  if crane.latch is not None:
    p.removeConstraint(crane.latch)
    crane.latch = None


def attach_hook(crane, scene, cfg):
  if crane.latch is not None:
    return True
  hook = link_pose(crane.body, crane.joints['hoist'])
  tip = hook[:3, 3] + hook[:3, :3] @ np.asarray(cfg.hook_tip)
  position, quaternion = p.getBasePositionAndOrientation(scene.load)
  rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
  local_tip = rotation.T @ (tip - position)
  half = cfg.load_size / 2
  gap = local_tip[2] - half
  outside = max(0., float(np.max(np.abs(local_tip[:2])) - half))
  # Accept the whole top face; allow 1 cm penetration because the hook is visual-only.
  if outside > 1e-6 or not -0.01 <= gap <= cfg.hook_attach_distance:
    print(f'Hook not near the top face: height gap {gap * 100:+.1f} cm '
          f'(allowed -1.0 to {cfg.hook_attach_distance * 100:.1f}), '
          f'outside top edge {outside * 100:.1f} cm. Align over the top and adjust Q/E.')
    return False
  state = p.getLinkState(crane.body, crane.joints['hoist'], computeForwardKinematics=True)
  relative = p.multiplyTransforms(*p.invertTransform(state[0], state[1]), position, quaternion)
  crane.latch = p.createConstraint(crane.body, crane.joints['hoist'], scene.load, -1, p.JOINT_FIXED,
    [0, 0, 0], relative[0], [0, 0, 0], parentFrameOrientation=relative[1])
  p.changeConstraint(crane.latch, maxForce=150)
  print('Hook attached.')
  return True


def read_input(crane, scene, cfg, keys=None):
  if keys is None:
    keys = update_keyboard(crane.keyboard, p.getKeyboardEvents())
  def triggered(key):
    return any(keys.get(ord(code), 0) & p.KEY_WAS_TRIGGERED for code in (key.lower(), key.upper()))
  if triggered('h'):
    crane.panel_visible = not crane.panel_visible
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, int(crane.panel_visible))
  def down(*codes):
    variants = list(codes) + [code - 32 for code in codes if ord('a') <= code <= ord('z')]
    return any(keys.get(code, 0) & p.KEY_IS_DOWN for code in variants)
  directions = [int(down(ord('w'), p.B3G_UP_ARROW)) - int(down(ord('s'), p.B3G_DOWN_ARROW)),
    int(down(ord('d'), p.B3G_RIGHT_ARROW)) - int(down(ord('a'), p.B3G_LEFT_ARROW)),
    int(down(ord('e'))) - int(down(ord('q')))]
  for i, direction in enumerate(directions):
    if direction:
      crane.targets[i] = crane.limits[i, 1 if direction > 0 else 0]
      crane.held[i], crane.braking[i] = True, False
    elif crane.held[i]:
      crane.held[i], crane.braking[i] = False, True
  if triggered('r'):
    release_hook(crane)
  elif triggered('f'):
    attach_hook(crane, scene, cfg)
  return triggered('c')


def advance_crane(crane, cfg):
  for i, (speed, acceleration, force) in enumerate(zip((0.45, 0.45, 0.25), (1.2, 1.2, 0.6), (80, 80, 40))):
    position, velocity = crane.positions[i], crane.velocities[i]
    if crane.braking[i]:
      next_velocity = math.copysign(max(0., abs(velocity) - acceleration * cfg.dt), velocity)
      position += (velocity + next_velocity) * cfg.dt / 2
      velocity = next_velocity
      crane.targets[i] = position
      if velocity == 0:
        position = p.getJointState(crane.body, crane.actuated[i])[0]
        crane.targets[i] = position
        crane.braking[i] = False
    else:
      target = np.clip(crane.targets[i], *crane.limits[i])
      diff = target - position
      desired = math.copysign(min(speed, math.sqrt(2 * acceleration * abs(diff)), 4 * abs(diff)), diff)
      velocity += np.clip(desired - velocity, -acceleration * cfg.dt, acceleration * cfg.dt)
      next_position = position + velocity * cfg.dt
      if diff * (target - next_position) <= 0:
        next_position, velocity = target, 0.
      position = next_position
    bounded = np.clip(position, *crane.limits[i])
    if bounded != position:
      velocity = 0.
    crane.positions[i], crane.velocities[i] = bounded, velocity
    p.setJointMotorControl2(crane.body, crane.actuated[i], p.POSITION_CONTROL,
      targetPosition=bounded, targetVelocity=velocity, force=force, positionGain=0.5, velocityGain=1.)


def rail_state(crane):
  states = p.getJointStates(crane.body, crane.actuated)
  return np.array([s[0] for s in states]), np.array([s[1] for s in states])
