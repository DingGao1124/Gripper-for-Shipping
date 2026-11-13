"""Compare the four camera-rate recordings and save tracking_fps_errors.png."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
  folder = Path(__file__).resolve().parent
  rates = (7.5, 15, 30, 60)
  styles = ("-", "--", "-.", ":")
  fig, (load, camera, image) = plt.subplots(3, 1, sharex=True, figsize=(9, 7), layout="constrained")
  shared_frames = []

  for fps, style in zip(rates, styles):
    data = np.genfromtxt(folder / f"tracking_{fps:g}Hz.csv", delimiter=",", names=True, ndmin=1)
    time = data["time_s"]
    tag = data["tag_x_m"]
    deflection = 1000 * (tag - data["trolley_x_m"])
    offset = 1000 * (tag - data["camera_x_m"])
    line, = camera.plot(time, offset, style, label=f"{fps:g} Hz ({len(time):,} frames)")
    load.plot(time, deflection, style, color=line.get_color())
    image.plot(time, data["corner_rms_px"], style, color=line.get_color())
    # Compare the runs at the capture times of the slowest camera.
    shared_frames.append(np.column_stack((time, deflection))[::round(fps / rates[0])])

  reference = shared_frames[0]
  if all(run.shape == reference.shape and np.allclose(run, reference, rtol=0, atol=1e-8)
         for run in shared_frames[1:]):
    load.clear()
    load.plot(reference[:, 0], reference[:, 1], color="#354b57")
    load.set_title(f"All rates coincide at {len(reference)} shared frames", loc="right", fontsize=9)

  for ax, title, unit in (
    (load, "(a)  Load deflection relative to the trolley", "Tag − trolley (mm)"),
    (camera, "(b)  Lateral camera tracking error", "Tag − camera (mm)"),
    (image, "(c)  Image-plane corner error", "Corner RMS (px)"),
  ):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(unit)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0)

  image.set_xlabel("Simulation time (s)")
  image.set_ylim(bottom=0)
  fig.legend(*camera.get_legend_handles_labels(), loc="outside lower center", ncols=4, frameon=False)
  fig.align_ylabels()
  fig.savefig(folder / "tracking_fps_errors.png", dpi=240)
  plt.close(fig)


if __name__ == "__main__":
  main()
