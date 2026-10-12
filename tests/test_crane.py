import matplotlib
import pytest

matplotlib.use("Agg")

from examples.crane import CRANE_BASE, TABLE_TOP_Z, BLOCK_HALF, main


def test_crane_stays_idle_without_input():
  position = main(gui=False, max_frames=240)
  assert abs(position[0] - CRANE_BASE[0]) < 0.001
  assert abs(position[1] - 0.0) < 0.05
  assert abs(position[2] - (TABLE_TOP_Z + BLOCK_HALF)) < 0.05


def test_manual_pick_place(monkeypatch):
  """在 DIRECT 模式注入 GUI 输入，验证滑条、按键和挂载的完整物理流程。"""
  import pybullet as p
  from examples import crane

  connect = p.connect
  close_plot = crane.plt.close
  create_body = p.createMultiBody
  create_constraint = p.createConstraint
  state = {"frame": -1, "block": None, "attachments": 0, "peak": 0, "final": None}

  def create(**kwargs):
    body = create_body(**kwargs)
    if kwargs.get("baseMass") == 0.05:
      state["block"] = body
    return body

  def attach(*args, **kwargs):
    state["attachments"] += 1
    return create_constraint(*args, **kwargs)

  def read_slider(index):
    frame = state["frame"] + 1
    if index == 0:
      return 0
    if index == 1:
      return 0.0 if frame < 420 else 0.03
    return crane.L_GRASP if 1 <= frame < 201 or frame >= 640 else crane.L_RETRACT

  def keyboard():
    state["frame"] += 1
    frame = state["frame"]
    position = p.getBasePositionAndOrientation(state["block"])[0]
    state["peak"] = max(state["peak"], position[2])
    if frame == 0:
      return {ord("h"): p.KEY_WAS_TRIGGERED}
    if frame in (20, 200):  # 第一次距离太远，第二次钩尖已到方块旁。
      return {ord("f"): p.KEY_WAS_TRIGGERED}
    if 380 <= frame < 410:
      return {ord("d"): p.KEY_IS_DOWN}
    if frame == 480:
      state["curve_peak"] = max(
        abs(v) for line in crane.plt.gcf().axes[0].lines
        if line.get_label() == "X / trolley" for v in line.get_ydata()
      )
    if frame == 840:
      return {ord("r"): p.KEY_WAS_TRIGGERED}
    if frame == 960:
      state["final"] = position
      raise KeyboardInterrupt
    return {}

  def close(fig):
    if not hasattr(fig, "axes"):
      return close_plot(fig)
    state["curves"] = {
      line.get_label(): (list(line.get_xdata()), list(line.get_ydata()))
      for line in fig.axes[0].lines
      if line.get_label() in ("X / trolley", "Y / gantry")
    }
    close_plot(fig)

  monkeypatch.setattr(crane.plt, "close", close)
  labels = {"gantry Y [m]": 0, "trolley X [m]": 1, "cable length [m]": 2}
  monkeypatch.setattr(p, "connect", lambda mode: connect(p.DIRECT))
  monkeypatch.setattr(p, "createMultiBody", create)
  monkeypatch.setattr(p, "createConstraint", attach)
  monkeypatch.setattr(p, "addUserDebugParameter", lambda label, *args: labels[label])
  monkeypatch.setattr(p, "readUserDebugParameter", read_slider)
  monkeypatch.setattr(p, "getKeyboardEvents", keyboard)
  monkeypatch.setattr(crane.time, "sleep", lambda seconds: None)
  monkeypatch.setattr(crane.plt, "show", lambda **kwargs: None)
  crane.main(gui=True)

  assert state["attachments"] == 1  # 远距离挂载没有创建约束。
  assert state["peak"] > 1.0       # 独立方块实际跟随钩子升起。
  assert abs(state["final"][0] - (CRANE_BASE[0] + 0.03)) < 0.05
  assert abs(state["final"][1]) < 0.05
  assert abs(state["final"][2] - (TABLE_TOP_Z + BLOCK_HALF)) < 0.05
  assert not p.isConnected()
  assert set(state["curves"]) == {"X / trolley", "Y / gantry"}
  assert all(len(times) == len(speeds) == 480 for times, speeds in state["curves"].values())
  assert state["curve_peak"] > 0.01


def test_close_during_physics_step(monkeypatch):
  import pybullet as p
  from examples import crane

  def close_mid_step():
    p.disconnect()
    raise p.error("Not connected to physics server.")

  monkeypatch.setattr(p, "stepSimulation", close_mid_step)
  assert crane.main(gui=False) is not None
  assert not p.isConnected()


def test_unrelated_physics_error_is_not_hidden(monkeypatch):
  import pytest
  import pybullet as p
  from examples import crane

  def fail_step():
    raise p.error("unexpected physics failure")

  monkeypatch.setattr(p, "stepSimulation", fail_step)
  with pytest.raises(p.error, match="unexpected physics failure"):
    crane.main(gui=False)
  assert not p.isConnected()


def test_slider_failure_keeps_keyboard_control(monkeypatch):
  import pybullet as p
  from examples import crane

  connect = p.connect
  disconnect = p.disconnect
  result = {}

  def read_slider(index):
    raise p.error("Failed to read parameter.")

  def capture_disconnect(client):
    # 场景顺序：plane=0、table=1、crane=2，trolley_x 是关节 1。
    result["x"] = p.getJointState(2, 1)[0]
    disconnect(client)

  monkeypatch.setattr(p, "connect", lambda mode: connect(p.DIRECT))
  monkeypatch.setattr(p, "disconnect", capture_disconnect)
  monkeypatch.setattr(p, "addUserDebugParameter", lambda *args: -1)
  monkeypatch.setattr(p, "readUserDebugParameter", read_slider)
  monkeypatch.setattr(p, "getKeyboardEvents", lambda: {ord("d"): p.KEY_IS_DOWN})
  monkeypatch.setattr(crane.time, "sleep", lambda seconds: None)
  monkeypatch.setattr(crane.plt, "show", lambda **kwargs: None)
  crane.main(gui=True, max_frames=30)
  assert result["x"] > 0.05
  assert not p.isConnected()


@pytest.mark.parametrize("key,joint,sign", [("w", 0, 1), ("s", 0, -1), ("d", 1, 1), ("a", 1, -1)])
@pytest.mark.parametrize("held_frames", [8, 35])
def test_keyboard_release_brakes_without_rebound(monkeypatch, key, joint, sign, held_frames):
  import pybullet as p
  from examples import crane

  connect, step = p.connect, p.stepSimulation
  state = {"frame": -1, "samples": []}
  labels = {"gantry Y [m]": 0, "trolley X [m]": 1, "cable length [m]": 2}

  def keys():
    state["frame"] += 1
    return {ord(key): p.KEY_IS_DOWN} if state["frame"] < held_frames else {}

  def measure_step():
    step()
    state["samples"].append((state["frame"], *p.getJointState(2, joint)[:2]))

  monkeypatch.setattr(p, "connect", lambda mode: connect(p.DIRECT))
  monkeypatch.setattr(p, "stepSimulation", measure_step)
  monkeypatch.setattr(p, "addUserDebugParameter", lambda label, *args: labels[label])
  monkeypatch.setattr(p, "readUserDebugParameter", lambda i: [0, 0, crane.L_RETRACT][i])
  monkeypatch.setattr(p, "getKeyboardEvents", keys)
  monkeypatch.setattr(crane.time, "sleep", lambda seconds: None)
  monkeypatch.setattr(crane.plt, "show", lambda **kwargs: None)
  crane.main(gui=True, max_frames=100)

  samples = [s for s in state["samples"] if s[0] >= held_frames]
  assert samples[0][2] * sign > 0.05  # 松键时仍有速度，随后逐步减速。
  assert min(sign * s[2] for s in samples) >= -1e-6
  assert max(sign * s[1] for s in samples) - sign * samples[-1][1] < 1e-6
  assert abs(samples[-1][2]) < 1e-6


def test_hoist_does_not_return_to_stale_slider_after_read_recovery(monkeypatch):
  import pybullet as p
  from examples import crane

  connect, step = p.connect, p.stepSimulation
  state = {"frame": -1, "lengths": []}
  labels = {"gantry Y [m]": 0, "trolley X [m]": 1, "cable length [m]": 2}

  def read_slider(index):
    frame = state["frame"] + 1
    if frame == 200:
      raise p.error("Failed to read parameter.")
    if index < 2:
      return 0
    # 面板读取恢复时重新报告初始值，模拟旧滑条值覆盖键盘位置。
    return 0.4 if 1 <= frame < 200 else crane.L_RETRACT

  def keys():
    state["frame"] += 1
    if state["frame"] == 0:
      return {ord("h"): p.KEY_WAS_TRIGGERED}
    return {ord("e"): p.KEY_IS_DOWN} if 100 <= state["frame"] < 130 else {}

  def measure_step():
    step()
    state["lengths"].append((state["frame"], p.getJointState(2, 4)[0]))

  monkeypatch.setattr(p, "connect", lambda mode: connect(p.DIRECT))
  monkeypatch.setattr(p, "stepSimulation", measure_step)
  monkeypatch.setattr(p, "addUserDebugParameter", lambda label, *args: labels[label])
  monkeypatch.setattr(p, "readUserDebugParameter", read_slider)
  monkeypatch.setattr(p, "getKeyboardEvents", keys)
  monkeypatch.setattr(crane.time, "sleep", lambda seconds: None)
  monkeypatch.setattr(crane.plt, "show", lambda **kwargs: None)
  crane.main(gui=True, max_frames=600)

  lengths = [length for frame, length in state["lengths"] if frame >= 180]
  assert min(lengths) > 0.45
  assert max(lengths) - min(lengths) < 1e-5
