"""Rendered RGB -> tag corners -> camera pose; no object-state shortcut."""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import time

import cv2
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from .scene import marker_rotation
from .config import Config


def pose(position, quaternion):
  result = np.eye(4)
  result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
  result[:3, 3] = position
  return result


def inverse(transform):
  result = np.eye(4)
  result[:3, :3] = transform[:3, :3].T
  result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
  return result


def marker_to_load(cfg):
  result = np.eye(4)
  result[:3, :3] = marker_rotation()
  result[:3, 3] = cfg.marker_position
  return inverse(result)


def marker_points(cfg):
  h = cfg.marker_size / 2
  return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


@dataclass
class Observation:
  time: float
  valid: bool = False
  corners: np.ndarray | None = None
  camera_tag: np.ndarray | None = None
  world_tag: np.ndarray | None = None
  world_load: np.ndarray | None = None
  reprojection_error: float = float('inf')


@dataclass
class Estimate:
  time: float = -1.0
  position: np.ndarray | None = None
  slow_position: np.ndarray | None = None
  velocity: np.ndarray | None = None
  tracking_velocity: np.ndarray | None = None
  valid_frames: int = 0
  rotation: np.ndarray | None = None
  history: deque = field(default_factory=deque)


def camera_parameters(cfg):
  focal = cfg.height / (2 * np.tan(np.deg2rad(cfg.fov / 2)))
  return np.array([[focal, 0, cfg.width / 2], [0, focal, cfg.height / 2], [0, 0, 1.]])


def render(camera_pose, cfg: Config, gui=False, api=p):
  origin = camera_pose[:3, 3]
  rotation = camera_pose[:3, :3]
  view = p.computeViewMatrix(origin, origin + rotation[:, 2], -rotation[:, 1])
  projection = p.computeProjectionMatrixFOV(cfg.fov, cfg.width / cfg.height, cfg.near, cfg.far)
  image = api.getCameraImage(cfg.width, cfg.height, view, projection,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER,
                           lightAmbientCoeff=0.8, lightDiffuseCoeff=0.2, lightSpecularCoeff=0, shadow=0,
                           flags=p.ER_NO_SEGMENTATION_MASK)
  rgba = image[2]
  if not isinstance(rgba, np.ndarray):
    # Non-NumPy Bullet builds return a flat tuple of bytes; avoid scalar casting.
    rgba = np.frombuffer(bytes(rgba), dtype=np.uint8)
  return np.asarray(rgba, dtype=np.uint8).reshape(cfg.height, cfg.width, 4)[:, :, :3].copy()


def make_detector():
  cv2.setNumThreads(2)
  dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
  parameters = cv2.aruco.DetectorParameters()
  parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
  return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect(rgb, camera_pose, timestamp, detector, cfg, previous=None):
  observation = Observation(timestamp)
  corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
  if ids is None or cfg.marker_id not in ids.ravel():
    return observation
  pixels = corners[int(np.flatnonzero(ids.ravel() == cfg.marker_id)[0])].reshape(4, 2).astype(np.float64)
  points, intrinsic = marker_points(cfg), camera_parameters(cfg)
  result = cv2.solvePnPGeneric(points, pixels, intrinsic, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
  candidates = []
  for rvec, tvec in zip(result[1], result[2]):
    rotation = cv2.Rodrigues(rvec)[0]
    translation = tvec.ravel()
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
      continue
    depths = (points @ rotation.T + translation)[:, 2]
    if np.min(depths) <= cfg.near or np.dot(rotation[:, 2], translation) >= 0:
      continue
    projected = cv2.projectPoints(points, rvec, tvec, intrinsic, None)[0].reshape(4, 2)
    error = float(np.sqrt(np.mean((pixels - projected) ** 2)))
    camera_tag = np.eye(4)
    camera_tag[:3, :3], camera_tag[:3, 3] = rotation, translation
    world_tag = camera_pose @ camera_tag
    continuity = 0.0
    if previous is not None and previous.valid and timestamp - previous.time < cfg.lost_seconds:
      continuity = 8 * np.linalg.norm(world_tag[:3, 3] - previous.world_tag[:3, 3])
      continuity += 0.1 * Rotation.from_matrix(previous.world_tag[:3, :3].T @ world_tag[:3, :3]).magnitude()
    candidates.append((error + continuity, error, camera_tag, world_tag))
  if not candidates:
    return observation
  _, error, camera_tag, world_tag = min(candidates, key=lambda candidate: candidate[0])
  if error > 3.0 or camera_tag[2, 3] > cfg.far:
    return observation
  return Observation(timestamp, True, pixels, camera_tag, world_tag, world_tag @ marker_to_load(cfg), error)


def update_estimate(estimate: Estimate, observation: Observation, cfg: Config):
  if not observation.valid:
    estimate.valid_frames = 0
    return
  delta = observation.time - estimate.time
  reset = estimate.position is None or delta > cfg.lost_seconds
  raw_rotation = observation.world_tag[:3, :3]
  if reset:
    estimate.rotation = raw_rotation.copy()
    estimate.history.clear()
  else:
    correction = Rotation.from_matrix(estimate.rotation.T @ raw_rotation).as_rotvec()
    alpha = delta / (cfg.rotation_filter + delta)
    estimate.rotation = estimate.rotation @ Rotation.from_rotvec(alpha * correction).as_matrix()
  filtered_tag = observation.world_tag.copy()
  filtered_tag[:3, :3] = estimate.rotation
  observation.world_load = filtered_tag @ marker_to_load(cfg)
  measured = observation.world_load[:3, 3]
  if reset:
    estimate.position = measured.copy()
    estimate.slow_position = measured.copy()
    estimate.velocity = np.zeros(3)
    estimate.tracking_velocity = np.zeros(3)
    estimate.valid_frames = 0
  else:
    alpha = delta / (cfg.measurement_filter + delta)
    estimate.position += alpha * (measured - estimate.position)
    alpha = delta / (cfg.position_filter + delta)
    estimate.slow_position += alpha * (estimate.position - estimate.slow_position)
  estimate.history.append((observation.time, estimate.position.copy()))
  while estimate.history and observation.time - estimate.history[0][0] > cfg.velocity_window:
    estimate.history.popleft()
  if len(estimate.history) >= 3:
    times = np.array([item[0] for item in estimate.history])
    positions = np.array([item[1] for item in estimate.history])
    centered = times - times.mean()
    slope = centered @ (positions - positions.mean(axis=0)) / max(centered @ centered, 1e-8)
    alpha = delta / (cfg.velocity_filter + delta)
    estimate.velocity += alpha * (slope - estimate.velocity)
    recent = times >= times[-1] - cfg.tracking_window
    fast_times, fast_positions = times[recent], positions[recent]
    if len(fast_times) >= 3:
      centered = fast_times - fast_times.mean()
      fast_slope = centered @ (fast_positions - fast_positions.mean(axis=0)) / max(centered @ centered, 1e-8)
      alpha = delta / (0.05 + delta)
      estimate.tracking_velocity += alpha * (fast_slope - estimate.tracking_velocity)
  estimate.time = observation.time
  estimate.valid_frames += 1


def overlay(rgb, observation, status, error, distance, base_velocity):
  image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
  if observation.valid:
    cv2.polylines(image, [observation.corners.astype(np.int32)], True, (0, 220, 0), 2)
  lines = [status, f"tag: {'visible' if observation.valid else 'lost'}   error: {error:.1f} px",
           f"camera distance: {distance:.3f} m   cart speed: {np.linalg.norm(base_velocity):.3f} m/s"]
  for i, text in enumerate(lines):
    cv2.putText(image, text, (12, 25 + i * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 220, 30), 1, cv2.LINE_AA)
  return image


def save_capture(rgb, camera_pose, cfg):
  directory = Path(cfg.capture_dir) / str(time.time_ns())
  directory.mkdir(parents=True)
  cv2.imwrite(str(directory / 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
  np.savez(directory / 'camera.npz', intrinsic=camera_parameters(cfg), world_camera=camera_pose,
           width=cfg.width, height=cfg.height, marker_size=cfg.marker_size)
  return directory
