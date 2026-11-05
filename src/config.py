"""SI units; transforms use column vectors and OpenCV camera axes."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
  dt: float = 1 / 240
  control_steps: int = 4
  camera_steps: int = 8
  plot_hz: float = 10.0
  keyboard_heartbeat: float = 0.10
  keyboard_timeout: float = 0.35  # Wall-clock timeout, independent of simulation speed.
  width: int = 640
  height: int = 480
  fov: float = 65.0
  near: float = 0.02
  far: float = 5.0
  marker_id: int = 0
  marker_size: float = 0.08
  load_size: float = 0.20
  load_mass: float = 0.20
  hook_tip: tuple = (0.0, 0.0, -0.05)  # Bottom of hook_link's visible cylinder.
  hook_attach_distance: float = 0.035
  marker_position: tuple = (-0.045, 0.102, 0.015)
  pad_position: tuple = (0.055, 0.101, -0.015)
  camera_mount: tuple = (0.046, 0.0, 0.030)  # +X side face, centered along the hand plate.
  camera_rpy: tuple = (0.25, 0.30, 0.0)  # Aim toward the tag beside the magnetic pad.
  magnet_tip: tuple = (0.0, 0.0, 0.105)
  base_height: float = 0.25
  base_offset: float = 1.00
  initial_joints: tuple = (0.0, 1.2, -0.6, -2.5, -2.5, 2.0, 0.0)
  base_speed: float = 0.45
  base_acceleration: float = 0.8
  position_filter: float = 0.35
  measurement_filter: float = 0.08
  rotation_filter: float = 0.15
  velocity_window: float = 0.4
  tracking_window: float = 0.15
  velocity_filter: float = 0.20
  observation_distance: float = 0.35
  initial_camera_rise: float = 0.18
  servo_gain: float = 1.4
  servo_damping: float = 0.025
  joint_speed: float = 1.5
  camera_speed: float = 0.35
  camera_angular_speed: float = 1.2
  home_gain: float = 0.04
  joint_margin: float = 0.08
  target_xy: tuple = (0.25, 0.25)
  target_radius: float = 0.08
  capture_height: tuple = (0.45, 0.85)
  crane_stop_speed: float = 0.02
  stable_seconds: float = 2.0
  stable_radius: float = 0.03
  stable_speed: float = 0.05
  approach_speed: float = 0.03
  lock_distance: float = 0.015
  lock_angle: float = 10.0
  lock_speed: float = 0.03
  lost_seconds: float = 0.20
  recovery_frames: int = 3
  obstacle_margin: float = 0.015
  capture_dir: Path = ROOT / "tmp" / "captures"
