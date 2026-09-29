import time

import pybullet as p
import pybullet_data

dt = 1 / 240
client = p.connect(p.GUI)  # 无界面运行可改成 p.DIRECT

try:
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.setTimeStep(dt)
  p.setGravity(0, 0, -9.81)

  p.loadURDF("plane.urdf", [0, 0, -0.3])
  robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

  # 先查看模型有哪些关节，不要假定所有机器人的关节编号相同
  for joint_id in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, joint_id)
    print(joint_id, info[1].decode("utf-8"))

  # 让第 0 个关节转到约 0.8 弧度
  p.setJointMotorControl2(
    robot,
    jointIndex=0,
    controlMode=p.POSITION_CONTROL,
    targetPosition=0.8,
    force=100,
  )

  for _ in range(480):  # 以 240 Hz 推进约 2 秒仿真时间
    p.stepSimulation()
    time.sleep(dt)     # 只为让 GUI 中的动作接近实时

  angle = p.getJointState(robot, 0)[0]
  print(f"关节 0 的实际角度：{angle:.3f} rad")
except KeyboardInterrupt as e:
  raise
else:
  print()
finally:
  p.disconnect(client)
