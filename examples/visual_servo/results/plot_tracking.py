"""Plot tag/camera tracking and synchronized trolley motion from one recording."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


# After key release, treat speeds up to this magnitude as rest to allow for simulation jitter.
REST_SPEED_MM_S = 0.1


def plot_trolley(
  ax: Axes,
  time_s: NDArray[np.float64],
  trolley_mm: NDArray[np.float64],
  command: NDArray[np.float64],
) -> None:
  """Draw the trolley's X velocity."""
  dt_s = np.diff(time_s)
  if not dt_s.size:
    ax.text(0.5, 0.5, "Two samples are needed to estimate velocity.",
            transform=ax.transAxes, ha="center", va="center")
    return
  if np.any(dt_s <= 0):
    raise ValueError("time_s must increase strictly")

  speed = np.diff(trolley_mm) / dt_s
  command = command[:-1]  # Classify each interval by the command sampled at its start.
  released = command == 0
  moving = np.abs(speed) > REST_SPEED_MM_S
  # Separate initial rest from later stops, even after the operator reverses direction.
  motion_seen = np.logical_or.accumulate((command != 0) | moving)
  phases = [
    ("Initial rest", "#e3e4e6", released & ~moving & ~motion_seen),
    ("+X command", "#b3d9cc", command > 0),
    ("−X command", "#cfbddd", command < 0),
    ("Released, still moving", "#f4cf8d", released & moving),
    ("Rest after motion", "#cfdeeb", released & ~moving & motion_seen),
  ]
  for label, color, mask in phases:
    if mask.any():
      ax.stairs(mask.astype(int), time_s, baseline=0, fill=True, transform=ax.get_xaxis_transform(),
                color=color, linewidth=0, zorder=0, label=label)

  ax.axhline(0, color="#76818a", linewidth=0.7, linestyle=":", zorder=2)
  ax.stairs(speed, time_s, baseline=None, color="#25343c", linewidth=1.65, zorder=3)
  if moving.any():
    fastest = np.argmax(np.abs(speed))
    mid_t = (time_s[fastest] + time_s[fastest + 1]) / 2
    above = speed[fastest] > 0
    ax.annotate(f"{speed[fastest]:+.0f} mm/s", (mid_t, speed[fastest]),
                xytext=(0, 4 if above else -4), textcoords="offset points", ha="center",
                va="bottom" if above else "top", fontsize=8, color="#25343c")
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncols=5, frameon=False,
            fontsize=8.5, handlelength=1.25, columnspacing=1.3)
  ax.margins(y=0.16)


def main():
  folder = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--fps", type=float, default=15, help="Recording frame rate in Hz (default: 15).")
  parser.add_argument("-o", "--output", help="PNG filename (default: same base name as the input CSV).")
  args = parser.parse_args()
  recording = folder / f"tracking_{args.fps:g}Hz.csv"
  if not recording.is_file():
    parser.error(f"Recording not found: {recording}")
  name = args.output or recording.stem
  output = folder / (name if name.lower().endswith(".png") else name + ".png")
  data = np.genfromtxt(recording, delimiter=",", names=True, ndmin=1)
  time_s = data["time_s"]
  if not time_s.size:
    raise ValueError(f"{recording.name} contains no samples")
  tag_mm = 1000 * data["tag_x_m"]
  camera_mm = 1000 * data["camera_x_m"]

  fig, (position, trolley) = plt.subplots(2, 1, sharex=True, figsize=(8.3, 5.65),
                                           gridspec_kw={"height_ratios": [2.1, 1]}, layout="constrained")
  # Distinct line styles also identify the two curves in a grayscale printout.
  position.plot(time_s, tag_mm, color="#0072b2", linewidth=2.2, label="AprilTag")
  position.plot(time_s, camera_mm, color="#d55e00", linewidth=2, linestyle="--", label="Wrist camera")
  position.legend(loc="upper left", frameon=False, ncols=2, fontsize=9, handlelength=2.7)

  gap_mm = np.abs(tag_mm - camera_mm)
  peak = np.argmax(gap_mm)
  peak_t = time_s[peak]
  midpoint = (tag_mm[peak] + camera_mm[peak]) / 2
  position.plot([peak_t, peak_t], [tag_mm[peak], camera_mm[peak]], color="#25343c",
                linewidth=1.6, marker="_", markersize=8, markeredgewidth=1.6, zorder=4)
  label_right = peak_t < time_s[0] + 0.25 * (time_s[-1] - time_s[0])
  position.annotate(f"Max |ΔX| = {gap_mm[peak]:.2f} mm\nat {peak_t:.2f} s",
                    xy=(peak_t, midpoint), xytext=(17 if label_right else -17,
                                                   -24 if midpoint > np.mean(position.get_ylim()) else 24),
                    textcoords="offset points", ha="left" if label_right else "right", va="center",
                    fontsize=9, color="#25343c", bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="none"),
                    arrowprops=dict(arrowstyle="-", color="#53616a", linewidth=0.85))

  plot_trolley(trolley, time_s, 1000 * data["trolley_x_m"], data["input_x"])
  position.set_title("(a)   Tag and camera tracking", loc="left", fontsize=11, fontweight="bold")
  position.set_title(f"Camera rate  {args.fps:g} Hz", loc="right", fontsize=9, color="#505e66")
  trolley.set_title("(b)   Trolley motion", loc="left", fontsize=11, fontweight="bold")
  trolley.set_title(f"Interval-mean velocity   ·   Rest threshold {REST_SPEED_MM_S:g} mm/s",
                    loc="right", fontsize=8.5, color="#505e66")
  position.set_ylabel("World X position (mm)")
  trolley.set_ylabel("X velocity (mm/s)")
  trolley.set_xlabel("Simulation time (s)")
  if time_s.size > 1:
    trolley.set_xlim(time_s[0], time_s[-1])
  for ax in (position, trolley):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#819098")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#84939b", linewidth=0.6, alpha=0.24)
    ax.tick_params(labelsize=9, width=0.7, length=3)
    ax.yaxis.set_label_coords(-0.082, 0.5)

  fig.savefig(output, dpi=300, facecolor="white")
  plt.close(fig)
  print(output)


if __name__ == "__main__":
  main()
