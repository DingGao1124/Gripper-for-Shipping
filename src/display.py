"""A read-only GUI replica keeps window work outside the physics process."""

from collections import deque
import multiprocessing as mp
from queue import Empty, Full
import time

import numpy as np
import pybullet as p

from .crane import initialize_crane
from .keyboard import KeyboardState, update_keyboard
from .scene import build_scene, draw_scene, create_cable, update_cable
from .vision import overlay

def start_display(cfg):
  context = mp.get_context('spawn')
  views, frames, inputs = context.Queue(maxsize=2), context.Queue(maxsize=2), context.Queue()
  stopped = context.Event()
  # Cocoa event pumps in OpenCV/Matplotlib must not consume Bullet's key events.
  viewer = context.Process(target=_viewer_loop, args=(views, inputs, stopped, cfg), daemon=True)
  monitor = context.Process(target=_monitor_loop, args=(frames, stopped, cfg), daemon=True)
  viewer.start()
  try:
    monitor.start()
  except Exception:
    stopped.set()
    viewer.join(timeout=2)
    if viewer.is_alive():
      viewer.terminate()
      viewer.join()
    raise
  return (viewer, monitor), (views, frames), inputs, stopped


def publish_display(display, frame):
  views, frames = display[1]
  _publish_latest(views, (frame[2], frame[-1]))
  _publish_latest(frames, frame[:-1])


def _publish_latest(frames, frame):
  try:
    frames.put_nowait(frame)
  except Full:
    try:
      frames.get_nowait()
      frames.put_nowait(frame)
    except (Empty, Full):
      pass  # Only display snapshots may be dropped, never sensor/control updates.


def read_display_input(display, state, timeout, now=None):
  inputs = display[2]
  now = time.monotonic() if now is None else now
  triggered = {}
  while True:
    try:
      timestamp, keys = inputs.get_nowait()
    except Empty:
      break
    if timestamp < state.timestamp or now - timestamp > timeout:
      continue
    state.timestamp = timestamp
    state.down = {code for code, flags in keys.items() if flags & p.KEY_IS_DOWN}
    for code, flags in keys.items():
      if flags & p.KEY_WAS_TRIGGERED:
        triggered[code] = p.KEY_WAS_TRIGGERED
  if now - state.timestamp > timeout:
    state.down.clear()
    return {}
  held = {code: p.KEY_IS_DOWN for code in state.down}
  for code, flags in triggered.items():
    held[code] = held.get(code, 0) | flags
  return held


def close_display(display):
  processes, frames, inputs, stopped = display
  stopped.set()
  for process in processes:
    process.join(timeout=2)
    if process.is_alive():
      process.terminate()
      process.join(timeout=2)
  for queue in (*frames, inputs):
    queue.cancel_join_thread()
    queue.close()


def view_state(sim):
  scene = sim.scene
  bodies = [scene.load, scene.cart, scene.panda] + [wheel[0] for wheel in scene.wheels]
  poses = [p.getBasePositionAndOrientation(body) for body in bodies]
  crane = [state[0] for state in p.getJointStates(scene.crane, list(range(p.getNumJoints(scene.crane))))]
  arm = [state[0] for state in p.getJointStates(scene.panda, sim.robot.movable)]
  return poses, crane, arm


def _apply_view(scene, joints, cable, state):
  poses, crane, arm = state
  bodies = [scene.load, scene.cart, scene.panda] + [wheel[0] for wheel in scene.wheels]
  for body, (position, orientation) in zip(bodies, poses):
    p.resetBasePositionAndOrientation(body, position, orientation)
  p.resetJointStatesMultiDof(scene.crane, joints[0], [[value] for value in crane])
  p.resetJointStatesMultiDof(scene.panda, joints[1], [[value] for value in arm])
  if cable:
    names = joints[2]
    top = p.getLinkState(scene.crane, names['trolley_x'], computeForwardKinematics=True)[4]
    bottom = p.getLinkState(scene.crane, names['hoist'], computeForwardKinematics=True)[4]
    update_cable(cable, top, bottom)


def _latest_frame(frames):
  try:
    frame = frames.get(timeout=0.005)
  except Empty:
    return None
  while True:
    try:
      frame = frames.get_nowait()
    except Empty:
      return frame


def _viewer_loop(frames, inputs, stopped, cfg):
  client = -1
  try:
    client = p.connect(p.GUI)
    if client < 0:
      raise RuntimeError('Cannot open PyBullet viewer.')
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    scene = build_scene(cfg)
    crane = initialize_crane(scene, gui=True)
    draw_scene(scene, cfg)
    cable = create_cable(scene)
    joints = [list(range(p.getNumJoints(scene.crane))),
              [i for i in range(p.getNumJoints(scene.panda)) if p.getJointInfo(scene.panda, i)[3] >= 0],
              crane.joints]
    keyboard = KeyboardState()
    last_keys, last_status, panel = {}, '', False
    last_input, label = float('-inf'), -1
    deadline = time.perf_counter()
    while not stopped.is_set() and p.isConnected(client):
      keys = update_keyboard(keyboard, p.getKeyboardEvents())
      if keys.get(ord('h'), 0) & p.KEY_WAS_TRIGGERED:
        panel = not panel
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, int(panel))
      keys.pop(ord('h'), None)
      now = time.monotonic()
      if keys != last_keys or now - last_input >= cfg.keyboard_heartbeat:
        inputs.put_nowait((now, keys))
        last_keys, last_input = keys, now
      frame = _latest_frame(frames)
      if frame is not None:
        status, state = frame
        _apply_view(scene, joints, cable, state)
        if status != last_status:
          label = p.addUserDebugText(status, [0, 0, 1.6], [0.1, 0.2, 0.25], 1.1, replaceItemUniqueId=label)
          last_status = status
      deadline += cfg.control_steps * cfg.dt
      delay = deadline - time.perf_counter()
      if delay > 0:
        time.sleep(delay)
      elif delay < -0.1:
        deadline = time.perf_counter()
  except p.error:
    if p.isConnected(client):
      raise
  finally:
    stopped.set()
    if client >= 0 and p.isConnected(client):
      p.disconnect(client)


def _monitor_loop(frames, stopped, cfg):
  import cv2
  import matplotlib.pyplot as plt

  figure = None
  try:
    cv2.setNumThreads(1)
    cv2.namedWindow('Panda wrist camera', cv2.WINDOW_NORMAL)
    plt.ion()
    figure, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    curves = [axes[0].plot([], [], label=name)[0] for name in ('trolley X', 'gantry Y', 'cart speed')]
    error_curve, = axes[1].plot([], [], color='tab:orange', label='visual error')
    axes[0].set_ylim(-0.6, 0.6)
    axes[0].set_ylabel('velocity [m/s]')
    axes[1].set_ylim(0, 160)
    axes[1].set_ylabel('error [px]')
    axes[1].set_xlabel('time relative to latest sample [s]')
    axes[1].set_xlim(-8, 0)
    for axis in axes:
      axis.grid(alpha=0.3)
      axis.legend(loc='upper right')
    figure.tight_layout()
    artists = [*curves, error_curve]
    backgrounds = []
    if figure.canvas.supports_blit:
      for artist in artists:
        artist.set_animated(True)
      def cache_background(event):
        backgrounds[:] = [figure.canvas.copy_from_bbox(axis.bbox) for axis in axes]
      figure.canvas.mpl_connect('draw_event', cache_background)
    figure.canvas.draw()
    plt.show(block=False)
    samples = deque(maxlen=480)
    last_sample, last_image, next_plot = -1., -1., 0.
    last_window_check = 0.
    plot_open = True
    deadline = time.perf_counter()
    while not stopped.is_set():
      frame = _latest_frame(frames)
      if frame is not None:
        rgb, observation, status, error, distance, base_velocity, rows = frame
        for row in rows:
          if row[0] > last_sample:
            samples.append(row)
            last_sample = row[0]
        if rgb is not None and observation.time != last_image:
          cv2.imshow('Panda wrist camera', overlay(rgb, observation, status, error, distance, base_velocity))
          if cv2.waitKey(1) & 0xFF == 27:
            break
          last_image = observation.time
      now = time.perf_counter()
      if now - last_window_check >= 0.25:
        if cv2.getWindowProperty('Panda wrist camera', cv2.WND_PROP_VISIBLE) < 1:
          break
        plot_open = plt.fignum_exists(figure.number)
        last_window_check = now
      if plot_open and samples and now >= next_plot:
        data = np.asarray(samples)
        x = data[:, 0] - data[-1, 0]
        for i, curve in enumerate(curves):
          curve.set_data(x, data[:, i + 1])
        error_curve.set_data(x, data[:, 4])
        peak = max(160., np.ceil(np.max(data[:, 4]) / 80) * 80)
        if peak > axes[1].get_ylim()[1]:
          axes[1].set_ylim(0, peak)
          backgrounds.clear()
        if backgrounds:
          for axis, background in zip(axes, backgrounds):
            figure.canvas.restore_region(background)
          for artist in artists:
            artist.axes.draw_artist(artist)
          for axis in axes:
            figure.canvas.blit(axis.bbox)
        else:
          figure.canvas.draw_idle()
        figure.canvas.flush_events()
        next_plot = now + 1 / cfg.plot_hz
      deadline += cfg.control_steps * cfg.dt
      delay = deadline - time.perf_counter()
      if delay > 0:
        time.sleep(delay)
      elif delay < -0.1:
        deadline = time.perf_counter()
  finally:
    stopped.set()
    cv2.destroyAllWindows()
    if figure is not None:
      plt.close(figure)
