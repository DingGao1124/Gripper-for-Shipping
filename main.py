import time

import pybullet as p
import pybullet_data


client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

dt = 1 / 30
p.setGravity(0, 0, -9.81)
p.setTimeStep(dt)

p.loadURDF("plane.urdf")
p.loadURDF("cube_small.urdf", basePosition=[0, 0, 1])
p.resetDebugVisualizerCamera(1.5, 45, -30, [0, 0, 0.3])

try:
  while p.isConnected(client):
    p.stepSimulation()
    time.sleep(dt)
except KeyboardInterrupt:
  pass
except p.error:
  if p.isConnected(client):
    raise
finally:
  if p.isConnected(client):
    p.disconnect(client)
