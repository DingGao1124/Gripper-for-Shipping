"""Build rigid-body assets; the cart is an explicitly kinematic platform."""

from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import pybullet_data

from .config import ROOT, Config


@dataclass
class Scene:
  crane: int
  load: int
  cart: int
  panda: int
  wheels: list
  ground: int


def marker_rotation():
  # Marker +Z faces world +Y; marker +Y points up.
  return np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


def _box(half, position, color, mass=0, api=p):
  collision = api.createCollisionShape(api.GEOM_BOX, halfExtents=half)
  visual = api.createVisualShape(api.GEOM_BOX, halfExtents=half, rgbaColor=color)
  return api.createMultiBody(mass, collision, visual, position)


def _panda(cfg, position, api=p):
  directory = Path(pybullet_data.getDataPath()) / "franka_panda"
  tree = ET.parse(directory / "panda.urdf")
  root = tree.getroot()
  for mesh in root.iter("mesh"):
    mesh.set("filename", str(directory / mesh.get("filename").removeprefix("package://")))
  # Keep finger joints, but replace their geometry with a magnetic tool.
  for name in ("panda_leftfinger", "panda_rightfinger"):
    link = root.find(f"link[@name='{name}']")
    for element in list(link):
      if element.tag in ("visual", "collision"):
        link.remove(element)
  # A low side pad seats the camera at the middle of the hand plate.
  hand = root.find("link[@name='panda_hand']")
  x, y, z = cfg.camera_mount
  hand.append(ET.fromstring(f'''<visual>
    <origin xyz="{x - 0.012} {y} {z}"/>
    <geometry><box size="0.006 0.026 0.026"/></geometry>
    <material name="camera_bracket"><color rgba="0.22 0.25 0.28 1"/></material>
  </visual>'''))
  for name, xyz, geometry, color in (
    ("magnet", (0, 0, 0.085), '<cylinder radius="0.02" length="0.04"/>', "0.15 0.25 0.35 1"),
    ("camera", cfg.camera_mount, '<box size="0.024 0.022 0.008"/>', "0.1 0.1 0.1 1"),
  ):
    collision = f'<collision><geometry>{geometry}</geometry></collision>' if name == 'magnet' else ''
    link = ET.SubElement(root, "link", name=name)
    holder = ET.fromstring(f'''<holder>
      <inertial><mass value="0.02"/><inertia ixx="0.00001" iyy="0.00001" izz="0.00001" ixy="0" ixz="0" iyz="0"/></inertial>
      <visual><geometry>{geometry}</geometry><material name="{name}_color"><color rgba="{color}"/></material></visual>
      {collision}
    </holder>''')
    link.extend(holder)
    if name == 'camera':
      ET.SubElement(link.find('visual'), 'origin', xyz='0 0 -0.004')
      link.append(ET.fromstring('''<visual>
        <origin xyz="0 0 0.002"/>
        <geometry><cylinder radius="0.004" length="0.004"/></geometry>
        <material name="camera_lens"><color rgba="0.08 0.18 0.24 1"/></material>
      </visual>'''))
    joint = ET.SubElement(root, "joint", name=f"{name}_mount", type="fixed")
    ET.SubElement(joint, "parent", link="panda_hand")
    ET.SubElement(joint, "child", link=name)
    origin = ET.SubElement(joint, "origin", xyz=" ".join(map(str, xyz)))
    if name == 'camera':
      origin.set('rpy', ' '.join(map(str, cfg.camera_rpy)))
  with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as file:
    path = Path(file.name)
    tree.write(file, encoding="utf-8", xml_declaration=True)
  try:
    return api.loadURDF(str(path), position, useFixedBase=True)
  finally:
    path.unlink()


def build_scene(cfg: Config, api=p):
  api.setAdditionalSearchPath(pybullet_data.getDataPath())
  api.setGravity(0, 0, -9.81)
  api.setTimeStep(cfg.dt)
  api.setRealTimeSimulation(0)
  api.setPhysicsEngineParameter(numSolverIterations=150)
  ground = api.loadURDF("plane.urdf")
  crane = api.loadURDF(str(ROOT / "assets" / "gantry_crane.urdf"), useFixedBase=True)
  half = cfg.load_size / 2
  shape = api.createCollisionShape(api.GEOM_BOX, halfExtents=[half] * 3)
  visual = api.createVisualShape(api.GEOM_BOX, halfExtents=[half] * 3, rgbaColor=[0.85, 0.4, 0.15, 1])
  tag = api.createVisualShape(api.GEOM_MESH, fileName=str(ROOT / "assets" / "visual_servo" / "tag.obj"),
                            meshScale=[cfg.marker_size / 0.8] * 3, rgbaColor=[1, 1, 1, 1])
  pad = api.createVisualShape(api.GEOM_CYLINDER, radius=0.023, length=0.003, rgbaColor=[0.65, 0.7, 0.75, 1])
  tag_q = api.getQuaternionFromEuler([np.pi / 2, 0, np.pi])
  load = api.createMultiBody(
    cfg.load_mass, shape, visual, [0, 0, half],
    linkMasses=[0, 0], linkCollisionShapeIndices=[-1, -1], linkVisualShapeIndices=[tag, pad],
    linkPositions=[cfg.marker_position, cfg.pad_position], linkOrientations=[tag_q, tag_q],
    linkInertialFramePositions=[[0, 0, 0]] * 2, linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
    linkParentIndices=[0, 0], linkJointTypes=[api.JOINT_FIXED] * 2, linkJointAxis=[[0, 0, 0]] * 2,
  )
  texture = api.loadTexture(str(ROOT / "assets" / "visual_servo" / "apriltag_36h11_0.png"))
  api.changeVisualShape(load, 0, textureUniqueId=texture, flags=api.VISUAL_SHAPE_DOUBLE_SIDED)
  api.changeDynamics(load, -1, lateralFriction=0.8, linearDamping=0.02, angularDamping=0.02)
  cart = _box([0.20, 0.18, 0.07], [0, cfg.base_offset, 0.18], [0.15, 0.35, 0.5, 1], api=api)
  wheel_shape = api.createVisualShape(api.GEOM_CYLINDER, radius=0.09, length=0.04, rgbaColor=[0.12, 0.12, 0.12, 1])
  wheels = []
  for x in (-0.16, 0.16):
    for y in (-0.19, 0.19):
      body = api.createMultiBody(0, -1, wheel_shape, [x, cfg.base_offset + y, 0.09],
                              api.getQuaternionFromEuler([np.pi / 2, 0, 0]))
      wheels.append((body, x, y))
  panda = _panda(cfg, [0, cfg.base_offset, cfg.base_height], api=api)
  # Only the bolted root may touch the cart; the moving arm must clear it.
  api.setCollisionFilterPair(panda, cart, -1, -1, 0)
  return Scene(crane, load, cart, panda, wheels, ground)


def draw_scene(scene, cfg):
  p.resetDebugVisualizerCamera(2.7, 45, -25, [0, 0.30, 0.65])
  center = np.array([*cfg.target_xy, 0.003])
  for angle in np.linspace(0, 2 * np.pi, 49)[:-1]:
    a = center + cfg.target_radius * np.array([np.cos(angle), np.sin(angle), 0])
    b = center + cfg.target_radius * np.array([np.cos(angle + np.pi / 24), np.sin(angle + np.pi / 24), 0])
    p.addUserDebugLine(a, b, [0.1, 0.8, 0.25], 3)
  p.addUserDebugText("HANDOFF", center + [0, 0, 0.02], [0.1, 0.65, 0.15], 1.1)


def create_cable(scene):
  # Batch the visual-only segments as one body; the viewer never steps physics.
  length = 0.05
  maximum = next(p.getJointInfo(scene.crane, i)[9] for i in range(p.getNumJoints(scene.crane))
                 if p.getJointInfo(scene.crane, i)[1] == b'hoist')
  shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.0025, length=length,
                              rgbaColor=[0.9, 0.7, 0.15, 1])
  count = int(np.ceil(maximum / length)) - 1
  body = p.createMultiBody(
    0, -1, shape, linkMasses=[0] * count, linkCollisionShapeIndices=[-1] * count,
    linkVisualShapeIndices=[shape] * count, linkPositions=[[0, 0, 0]] * count,
    linkOrientations=[[0, 0, 0, 1]] * count, linkInertialFramePositions=[[0, 0, 0]] * count,
    linkInertialFrameOrientations=[[0, 0, 0, 1]] * count, linkParentIndices=list(range(count)),
    linkJointTypes=[p.JOINT_PRISMATIC] * count, linkJointAxis=[[0, 0, 1]] * count,
  )
  return body, list(range(count))


def update_cable(cable, top, bottom):
  if not cable:
    return
  start, end = np.asarray(top), np.asarray(bottom)
  delta = end - start
  length = np.linalg.norm(delta)
  direction = delta / max(length, 1e-9)
  if direction[2] < -0.999999:
    rotation = [1, 0, 0, 0]
  else:
    cross = np.cross([0, 0, 1], direction)
    rotation = np.r_[cross, 1 + direction[2]]
    rotation /= np.linalg.norm(rotation)
  body, joints = cable
  p.resetBasePositionAndOrientation(body, start + direction * 0.025, rotation)
  spacing = max(0., length - 0.05) / len(joints)
  p.resetJointStatesMultiDof(body, joints, [[spacing]] * len(joints))
