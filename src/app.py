"""Single-clock simulation orchestration and lightweight live displays."""

import argparse
from collections import deque
from dataclasses import dataclass
import time

import numpy as np
import pybullet as p

from .config import Config
from .display import start_display, publish_display, close_display, read_display_input, view_state
from .crane import Crane, advance_crane, initialize_crane, rail_state, read_input
from .keyboard import KeyboardState
from .robot import Robot, advance_robot, camera_pose, initialize_robot, obstacle_risk, stop_arm
from .scene import Scene, build_scene, draw_scene
from .servo import base_follow, visual_servo
from .task import Task, try_handoff, update_task
from .vision import Estimate, Observation, detect, make_detector, render, save_capture, update_estimate


@dataclass
class Simulation:
  client: int
  cfg: Config
  gui: bool
  scene: Scene
  crane: Crane
  robot: Robot
  task: Task
  detector: object
  observation: Observation
  estimate: Estimate
  rgb: np.ndarray | None = None
  image_pose: np.ndarray | None = None
  previous: Observation | None = None
  ticks: int = 0
  error: float = 0.0
  risk: str = ""


def create_simulation(cfg=None, gui=True):
  cfg = cfg or Config()
  client = p.connect(p.GUI if gui else p.DIRECT)
  if client < 0:
    raise RuntimeError("Cannot connect to PyBullet.")
  try:
    scene = build_scene(cfg)
    crane = initialize_crane(scene, gui)
    robot = initialize_robot(scene, cfg)
    task = Task(distance=cfg.observation_distance)
    if gui:
      draw_scene(scene, cfg)
    sim = Simulation(client, cfg, gui, scene, crane, robot, task, make_detector(), Observation(0), Estimate())
    return sim
  except Exception:
    p.disconnect(client)
    raise


def step_simulation(sim: Simulation):
  cfg, now = sim.cfg, sim.ticks * sim.cfg.dt
  if sim.ticks % cfg.camera_steps == 0:
    camera = camera_pose(sim.robot)
    sim.rgb = render(camera, cfg, sim.gui)
    sim.image_pose = camera.copy()
    sim.observation = detect(sim.rgb, camera, now, sim.detector, cfg, sim.previous)
    if sim.observation.valid:
      sim.previous = sim.observation
    update_estimate(sim.estimate, sim.observation, cfg)
  if sim.ticks % cfg.control_steps == 0:
    sim.risk = obstacle_risk(sim.robot, sim.scene, cfg)
    update_task(sim.task, sim.observation, sim.estimate, sim.crane, sim.robot, cfg, now, sim.risk)
    usable = sim.observation.valid and now - sim.estimate.time < cfg.lost_seconds and not sim.risk
    if sim.task.phase == "HOLD":
      sim.robot.base_command[:] = 0
      sim.robot.joint_command[:] = 0
    elif usable and sim.task.phase in ("TRACK", "SETTLE", "APPROACH"):
      following = sim.task.phase in ("TRACK", "SETTLE") and not sim.task.stable_ready
      sim.robot.base_command = base_follow(sim.robot, sim.estimate, cfg) if following else np.zeros(2)
      sim.robot.joint_command, sim.error = visual_servo(
        sim.robot, sim.observation, sim.estimate, sim.task.distance, cfg
      )
      try_handoff(sim.task, sim.observation, sim.estimate, sim.crane, sim.robot, sim.scene, cfg)
    else:
      sim.robot.base_command[:] = 0
      stop_arm(sim.robot)
  advance_crane(sim.crane, cfg)
  advance_robot(sim.robot, sim.scene, cfg)
  p.stepSimulation()
  sim.ticks += 1


def close_simulation(sim):
  if p.isConnected(sim.client):
    p.disconnect(sim.client)


def main(gui=True, seconds=None):
  sim = create_simulation(gui=False)
  display = None
  keyboard = KeyboardState()
  samples = deque(maxlen=480)
  try:
    if gui:
      print('W/S: gantry Y | A/D: trolley X | Q/E: raise/lower | F: hook | R: release | H: panel | C: capture')
      print('Lift load center to 0.45-0.85 m, move into green target and stop; Panda takes over automatically.')
      display = start_display(sim.cfg)
    if not gui and seconds is None:
      seconds = 3.0
    frame_dt = sim.cfg.control_steps * sim.cfg.dt
    deadline = time.perf_counter()
    while p.isConnected(sim.client) and (seconds is None or sim.ticks * sim.cfg.dt < seconds):
      if gui:
        processes, _, _, stopped = display
        if stopped.is_set() or any(not process.is_alive() for process in processes):
          break
        keys = read_display_input(display, keyboard, sim.cfg.keyboard_timeout)
        if read_input(sim.crane, sim.scene, sim.cfg, keys) and sim.rgb is not None:
          print(f'Saved camera capture: {save_capture(sim.rgb, sim.image_pose, sim.cfg)}')
      for _ in range(sim.cfg.control_steps):
        step_simulation(sim)
      if gui:
        current = sim.ticks * sim.cfg.dt
        status = f"{sim.task.phase}: {sim.task.reason or 'visual following'}"
        _, speed = rail_state(sim.crane)
        samples.append((current, speed[1], speed[0], np.linalg.norm(sim.robot.base_velocity), sim.error))
        distance = sim.observation.camera_tag[2, 3] if sim.observation.valid else float('nan')
        publish_display(display, (sim.rgb, sim.observation, status, sim.error, distance,
                                  sim.robot.base_velocity.copy(), list(samples), view_state(sim)))
        # Absolute deadlines let the light frame after a camera update catch up.
        # Physics steps, observations and controller updates are never skipped.
        deadline += frame_dt
        delay = deadline - time.perf_counter()
        if delay > 0:
          time.sleep(delay)
    return sim.task.phase
  except KeyboardInterrupt:
    return sim.task.phase
  except p.error:
    if p.isConnected(sim.client):
      raise
    return sim.task.phase
  finally:
    if display is not None:
      close_display(display)
    close_simulation(sim)


def cli():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--headless", action="store_true", help="run without display windows")
  parser.add_argument("--seconds", type=float, help="optional duration in simulation seconds")
  args = parser.parse_args()
  main(gui=not args.headless, seconds=args.seconds)
