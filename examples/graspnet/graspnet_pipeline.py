"""GraspNet 抓取全流程: 相机取图 -> 模型推理 -> 位姿转换 -> 机械臂执行。

流程分四段, 每段都能单独替换:

    (1) capture    RGB-D 图 + 相机内参          本文件, 或换成你的相机 SDK
    (2) predict    -> GraspGroup (相机坐标系)   graspnet-baseline, 需要 CUDA
    (3) transform  -> 基座坐标系 + 路点          本文件, 纯几何
    (4) execute    -> 机械臂动作                 本文件定义接口, 实现按你的机器人

只有 (2) 需要 CUDA —— graspnet-baseline 的 pointnet2 算子是无条件的
CUDAExtension。(1)(3)(4) 在任何机器上都能跑, 配合 --skip-inference 可以先把
几何和路点逻辑验证清楚, 再上有卡的机器。

抓取位姿的坐标约定, 从 graspnetAPI 的夹爪几何实测反推(见 verify_grasp_frame):

    R[:, 0]      接近方向, 夹爪沿此方向插入, 尾部在反向
    R[:, 1]      闭合方向, 两指沿此轴张开 width
    translation  抓取中心点
    depth        指尖沿接近方向超出中心点的距离

用法:
    # 不接机器人也不推理, 只验证几何/路点(任何机器)
    python examples/grasp_pipeline.py --self-test

    # 用已有的 GraspGroup npy 走后半程
    python examples/grasp_pipeline.py --grasp-npy gg.npy --dry-run

    # 完整跑(Linux + CUDA + 权重)
    python examples/grasp_pipeline.py \\
        --data-dir /path/to/rgbd \\
        --baseline-root ~/code/graspnet-baseline \\
        --checkpoint ~/ckpt/checkpoint-rs.tar \\
        --hand-eye hand_eye.npy --dry-run
"""

import argparse
import copy
import os
import sys

import numpy as np

# 模型输出的夹爪宽度上限(米)。graspnetAPI 的评测也是按这个值裁剪的
MODEL_MAX_WIDTH = 0.1

# 沿接近方向的后退距离(米), 预抓取点取在这里
DEFAULT_STANDOFF = 0.10

# 抬起高度(米)
DEFAULT_LIFT = 0.15

# 抓取系 -> 机器人 TCP 系的固定旋转。
# GraspNet 的接近方向是局部 +x, 而多数机械臂的 TCP 是 +z 朝外, 差一个旋转。
# 这里把局部 x 转到 z: (x,y,z)_grasp -> (z,y,-x)_tcp。
# 你的机器人如果不是 z 朝外, 改这个矩阵, 不要去改路点计算。
GRASP_TO_TCP = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
])


# ---------------------------------------------------------------- (1) 取图


class Frame(object):
    """一帧 RGB-D 数据, 这是推理的全部输入。"""

    def __init__(self, color, depth, intrinsic, factor_depth, workspace_mask=None):
        # color: (H,W,3) float [0,1]; depth: (H,W) 原始整数值
        # intrinsic: (3,3); factor_depth: 深度值转米的除数, 常见 1000
        self.color = color
        self.depth = depth
        self.intrinsic = np.asarray(intrinsic, dtype=np.float64)
        self.factor_depth = float(factor_depth)
        self.workspace_mask = workspace_mask

    @property
    def shape(self):
        return self.depth.shape


def load_frame_from_dir(data_dir):
    """读 graspnet-baseline demo 那套四件套。

    目录里需要 color.png / depth.png / meta.mat, workspace_mask.png 可选。
    meta.mat 里要有 intrinsic_matrix 和 factor_depth。
    格式参考 graspnet-baseline/doc/example_data/。
    """
    import scipy.io as scio
    from PIL import Image

    color = np.array(Image.open(os.path.join(data_dir, "color.png")),
                     dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, "depth.png")))

    meta = scio.loadmat(os.path.join(data_dir, "meta.mat"))
    intrinsic = meta["intrinsic_matrix"]
    factor_depth = float(np.squeeze(meta["factor_depth"]))

    mask_path = os.path.join(data_dir, "workspace_mask.png")
    workspace_mask = None
    if os.path.exists(mask_path):
        workspace_mask = np.array(Image.open(mask_path)).astype(bool)
    else:
        # 没有掩码也能跑, 但输出会稀疏很多, 且容易把桌面/背景当成可抓
        print("  ! 没有 workspace_mask.png, 抓取输出会稀疏且含背景")

    return Frame(color, depth, intrinsic, factor_depth, workspace_mask)


def capture_from_realsense(serial=None, warmup=30):
    """从 RealSense 实时取一帧。需要 pyrealsense2, 不在本项目依赖里。

    这里单独列出来是因为实时取图有两个容易错的点:
    深度要对齐到彩色(align), 以及内参要取对齐之后的那个 profile。
    """
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    try:
        align = rs.align(rs.stream.color)
        # 前几帧自动曝光没稳定, 丢掉
        for _ in range(warmup):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color = np.asanyarray(color_frame.get_data())[:, :, ::-1]  # BGR->RGB
        color = color.astype(np.float32) / 255.0
        depth = np.asanyarray(depth_frame.get_data())

        # 内参必须取对齐后的 color profile, 取 depth 的会错位
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1.0]])
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        factor_depth = 1.0 / scale  # get_depth_scale 是米/单位, 这里要单位/米
        return Frame(color, depth, K, factor_depth)
    finally:
        pipeline.stop()


# ------------------------------------------------------------- (2) 模型推理


class GraspNetPredictor(object):
    """包一层 graspnet-baseline 的推理。

    baseline 不是 pip 包, 得用 sys.path 挂进来, 且它的三个子目录
    (models/dataset/utils)都要挂, 内部是按扁平模块名互相 import 的。
    """

    def __init__(self, baseline_root, checkpoint_path, num_point=20000,
                 num_view=300, device=None):
        self.baseline_root = os.path.abspath(os.path.expanduser(baseline_root))
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.num_point = num_point
        self.num_view = num_view
        self._net = None
        self._device = device

    def _add_paths(self):
        if not os.path.isdir(self.baseline_root):
            raise RuntimeError("找不到 graspnet-baseline: %s" % self.baseline_root)
        for sub in ("", "models", "dataset", "utils"):
            path = os.path.join(self.baseline_root, sub) if sub else self.baseline_root
            if path not in sys.path:
                sys.path.append(path)

    def load(self):
        """建网 + 加载权重。权重是必须的, 随机初始化的网络输出没有意义。"""
        import torch

        self._add_paths()
        from graspnet import GraspNet  # baseline 的 models/graspnet.py

        if self._device is None:
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if not str(self._device).startswith("cuda"):
            # pointnet2 是 CUDAExtension, 到这一步其实已经 import 失败了。
            # 留这句是为了万一有人改了算子实现, 至少知道自己在跑 CPU。
            print("  ! 没有 CUDA, pointnet2 算子基本不可用")

        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12,
                       num_depth=4, cylinder_radius=0.05, hmin=-0.02,
                       hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
        net.to(torch.device(self._device))

        checkpoint = torch.load(self.checkpoint_path, map_location=self._device)
        net.load_state_dict(checkpoint["model_state_dict"])
        print("  已加载权重 %s (epoch %d)" % (self.checkpoint_path,
                                              checkpoint.get("epoch", -1)))
        net.eval()
        self._net = net
        return self

    def build_input(self, frame):
        """RGB-D -> 采样后的点云, 以及用于碰撞检测的完整点云。"""
        import torch

        self._add_paths()
        from data_utils import CameraInfo, create_point_cloud_from_depth_image

        h, w = frame.shape
        K = frame.intrinsic
        camera = CameraInfo(float(w), float(h), K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                            frame.factor_depth)
        cloud = create_point_cloud_from_depth_image(frame.depth, camera, organized=True)

        mask = frame.depth > 0
        if frame.workspace_mask is not None:
            mask = frame.workspace_mask & mask
        cloud_masked = cloud[mask]
        color_masked = frame.color[mask]
        if len(cloud_masked) == 0:
            raise RuntimeError("掩码内没有有效深度点, 检查 workspace_mask 和深度图")

        # 网络输入点数固定, 多了随机采样, 少了有放回补齐
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked),
                                     self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        end_points = {
            "point_clouds": torch.from_numpy(
                cloud_sampled[np.newaxis].astype(np.float32)).to(self._device),
            "cloud_colors": color_masked[idxs],
        }
        return end_points, cloud_masked

    def predict(self, frame):
        """推理, 返回相机坐标系下的 GraspGroup 和场景点云。"""
        import torch
        from graspnetAPI import GraspGroup
        from graspnet import pred_decode

        if self._net is None:
            self.load()
        end_points, scene_cloud = self.build_input(frame)
        with torch.no_grad():
            end_points = self._net(end_points)
            grasp_preds = pred_decode(end_points)
        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        return gg, scene_cloud


# --------------------------------------------------------------- (3) 后处理


def filter_collisions(gg, scene_cloud, baseline_root, collision_thresh=0.01,
                      voxel_size=0.01, approach_dist=0.05):
    """用观测点云做碰撞检测, 滤掉会撞的抓取。

    注意这是 model-free 的, 只看得见的点。遮挡后面的几何它不知道,
    所以过了检测不等于真的不碰。
    """
    for sub in ("utils",):
        path = os.path.join(os.path.abspath(os.path.expanduser(baseline_root)), sub)
        if path not in sys.path:
            sys.path.append(path)
    from collision_detector import ModelFreeCollisionDetector

    detector = ModelFreeCollisionDetector(scene_cloud, voxel_size=voxel_size)
    collision_mask = detector.detect(gg, approach_dist=approach_dist,
                                    collision_thresh=collision_thresh)
    return gg[~collision_mask]


def postprocess(gg, top_k=50, gripper_max_width=MODEL_MAX_WIDTH):
    """裁剪宽度 -> NMS -> 按分数排序 -> 取前 top_k。

    宽度必须裁: 模型输出上限是 0.1m, 你的夹爪行程通常更小, 发一个超程的
    宽度过去要么报错要么夹不到位。
    """
    if len(gg) == 0:
        return gg

    widths = gg.widths.copy()
    n_over = int(np.sum(widths > gripper_max_width))
    gg.widths = np.clip(widths, 0.0, gripper_max_width)
    if n_over:
        print("  %d 个抓取宽度超过夹爪上限 %.3fm, 已裁剪" % (n_over, gripper_max_width))

    gg.nms()
    gg.sort_by_score()
    return gg[:top_k] if len(gg) > top_k else gg


# ----------------------------------------------------- (4) 位姿 -> 机器人路点


def transform_grasps_to_base(gg, T_base_cam):
    """相机系 -> 机器人基座系。

    T_base_cam 是手眼标定的结果(4x4)。这一步的精度基本决定成败,
    标定误差会一比一地变成抓取偏移。
    """
    T = np.asarray(T_base_cam, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_base_cam 必须是 4x4, 收到 %s" % (T.shape,))
    # GraspGroup.transform 是原地改的(没有 .copy 方法), 深拷一份避免改掉调用方
    out = copy.deepcopy(gg)
    out.transform(T)
    return out


def grasp_to_tcp_pose(rotation, translation):
    """抓取位姿 -> TCP 位姿(4x4)。

    抓取系是 x 朝外, 机器人 TCP 通常 z 朝外, 用 GRASP_TO_TCP 补上这个差。
    """
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation) @ GRASP_TO_TCP
    T[:3, 3] = np.asarray(translation)
    return T


def grasp_to_waypoints(grasp, standoff=DEFAULT_STANDOFF, lift=DEFAULT_LIFT):
    """把一个抓取展开成一串路点。

    返回 [(名字, 4x4 TCP 位姿, 夹爪宽度), ...]。夹爪宽度为 None 表示这一步
    不动夹爪。

    关键点是 approach 那一段必须走直线: 从预抓取点沿接近方向插到抓取点。
    交给普通的关节空间规划器会走一条弧线, 侧面撞进物体。
    """
    R = np.asarray(grasp.rotation_matrix, dtype=np.float64)
    center = np.asarray(grasp.translation, dtype=np.float64)
    approach = R[:, 0]  # 接近方向, 见模块开头的约定

    # 预抓取点: 沿接近方向往后退 standoff
    pre_center = center - approach * standoff
    # 抬起点: 抓到之后竖直往上, 不是沿接近方向退
    lift_center = center + np.array([0.0, 0.0, lift])

    width = float(grasp.width)
    # 张开量留点余量, 正好等于物体宽度会夹不进去
    open_width = min(width + 0.01, MODEL_MAX_WIDTH)

    return [
        ("pre_grasp", grasp_to_tcp_pose(R, pre_center), open_width),
        ("approach", grasp_to_tcp_pose(R, center), None),
        ("close", grasp_to_tcp_pose(R, center), width),
        ("lift", grasp_to_tcp_pose(R, lift_center), None),
    ]


class RobotBase(object):
    """机械臂接口。换机器人只需要实现这四个方法。

    move_to        关节空间或笛卡尔空间随意, 用于自由移动
    move_linear    必须是笛卡尔直线插补, approach 那一步依赖它
    set_gripper    夹爪开合到指定宽度(米)
    """

    def move_to(self, pose):
        raise NotImplementedError

    def move_linear(self, pose):
        raise NotImplementedError

    def set_gripper(self, width):
        raise NotImplementedError

    def home(self):
        raise NotImplementedError


class DryRunRobot(RobotBase):
    """只打印不动作, 用来在没有硬件的情况下验证路点。"""

    def __init__(self):
        self.log = []

    def _fmt(self, pose):
        t = pose[:3, 3]
        approach = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])  # TCP 系 z 朝外
        return "xyz=(%.3f, %.3f, %.3f) 接近=(%.2f, %.2f, %.2f)" % (
            t[0], t[1], t[2], approach[0], approach[1], approach[2])

    def move_to(self, pose):
        self.log.append(("move_to", pose))
        print("    move_to     %s" % self._fmt(pose))

    def move_linear(self, pose):
        self.log.append(("move_linear", pose))
        print("    move_linear %s" % self._fmt(pose))

    def set_gripper(self, width):
        self.log.append(("set_gripper", width))
        print("    set_gripper width=%.3f" % width)

    def home(self):
        self.log.append(("home", None))
        print("    home")


def execute_grasp(robot, grasp, standoff=DEFAULT_STANDOFF, lift=DEFAULT_LIFT):
    """按路点执行一次抓取。"""
    waypoints = grasp_to_waypoints(grasp, standoff=standoff, lift=lift)
    for name, pose, width in waypoints:
        print("  [%s]" % name)
        if name == "close":
            robot.set_gripper(width)
            continue
        if width is not None:
            robot.set_gripper(width)
        # 只有 approach 必须直线, 其余可以自由规划
        if name == "approach":
            robot.move_linear(pose)
        else:
            robot.move_to(pose)
    return waypoints


# ------------------------------------------------------------------- 自检


def verify_grasp_frame():
    """用夹爪网格数值验证接近轴/闭合轴的约定, 确认本文件的假设成立。

    做法: 给一个已知旋转, 看夹爪顶点沿各轴的投影范围。
    指尖应在接近轴正端(= depth), 尾部在负端; 闭合轴上应对称张开约 width/2。
    """
    from graspnetAPI.utils.utils import plot_gripper_pro_max

    th = np.pi / 3
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    width, depth = 0.08, 0.04
    mesh = plot_gripper_pro_max(np.zeros(3), R, width=width, depth=depth)
    v = np.asarray(mesh.vertices)

    proj_app = v @ R[:, 0]
    proj_close = v @ R[:, 1]
    # 指尖正好在 depth 处
    assert abs(proj_app.max() - depth) < 1e-6, proj_app.max()
    # 尾部在负侧(tail_length + finger_width + depth_base = 0.064)
    assert proj_app.min() < -0.06, proj_app.min()
    # 闭合轴对称, 半宽 = width/2 + finger_width
    assert abs(proj_close.max() - (width / 2 + 0.004)) < 1e-6, proj_close.max()
    assert abs(proj_close.min() + (width / 2 + 0.004)) < 1e-6, proj_close.min()
    print("  OK  接近轴 = R[:,0], 闭合轴 = R[:,1]")


def verify_waypoints():
    """验证路点几何: 预抓取点在接近方向后方, 且 approach 是直线。"""
    from graspnetAPI import Grasp

    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    center = np.array([0.4, 0.1, 0.25])
    g = Grasp(np.concatenate([[0.9, 0.06, 0.02, 0.03], R.reshape(-1), center, [0]]))

    standoff, lift = 0.1, 0.15
    wps = grasp_to_waypoints(g, standoff=standoff, lift=lift)
    names = [n for n, _, _ in wps]
    assert names == ["pre_grasp", "approach", "close", "lift"], names

    pre = wps[0][1][:3, 3]
    grasp_pt = wps[1][1][:3, 3]
    approach = R[:, 0]

    # 预抓取点应在抓取点沿接近方向的正后方, 距离 = standoff
    assert abs(np.linalg.norm(grasp_pt - pre) - standoff) < 1e-9
    assert np.allclose(grasp_pt - pre, approach * standoff)
    # 插入方向与接近方向同向
    assert np.dot(grasp_pt - pre, approach) > 0
    # 抬起是竖直向上
    assert np.allclose(wps[3][1][:3, 3] - grasp_pt, [0, 0, lift])
    # close 这一步不动位置, 只动夹爪, 宽度等于 grasp.width
    assert np.allclose(wps[2][1][:3, 3], grasp_pt)
    assert abs(wps[2][2] - g.width) < 1e-9
    # 预抓取时夹爪要比目标宽度张得更开
    assert wps[0][2] > g.width

    # TCP 系的 z 应该对上抓取系的接近方向
    tcp_z = wps[1][1][:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(tcp_z, approach, atol=1e-9), (tcp_z, approach)
    print("  OK  路点几何: 预抓取 -> 直线插入 -> 闭合 -> 抬起")


def verify_transform_and_clip():
    """验证坐标变换不改调用方对象, 以及宽度裁剪生效。"""
    from graspnetAPI import GraspGroup

    gg = GraspGroup(np.zeros((3, 17), dtype=np.float32))
    gg.rotation_matrices = np.tile(np.eye(3), (3, 1, 1))
    gg.translations = np.array([[0.0, 0, 0.3], [0.1, 0, 0.3], [0, 0.1, 0.3]])
    gg.scores = np.array([0.5, 0.9, 0.1])
    gg.widths = np.array([0.04, 0.15, 0.08])  # 0.15 超过上限

    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    before = gg.translations.copy()
    out = transform_grasps_to_base(gg, T)
    # 原对象不能被改
    assert np.allclose(gg.translations, before), "transform 改掉了调用方对象"
    assert np.allclose(out.translations[0], [1.0, 2.0, 3.3]), out.translations[0]

    kept = postprocess(out, top_k=2, gripper_max_width=0.085)
    assert kept.widths.max() <= 0.085 + 1e-9, kept.widths
    assert len(kept) <= 2
    # 排序后第一个应是分数最高的
    assert abs(kept.scores[0] - 0.9) < 1e-6, kept.scores
    print("  OK  坐标变换无副作用, 宽度裁剪与排序正常")


def self_test():
    print("自检 (不需要 CUDA, 不需要数据集):")
    verify_grasp_frame()
    verify_waypoints()
    verify_transform_and_clip()

    print("\n路点演示:")
    from graspnetAPI import Grasp
    R = np.eye(3)
    g = Grasp(np.concatenate([[0.9, 0.05, 0.02, 0.03], R.reshape(-1),
                              [0.45, 0.0, 0.2], [0]]))
    execute_grasp(DryRunRobot(), g)
    print("\n全部通过")
    return 0


# ------------------------------------------------------------------- 主流程


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true",
                   help="只跑几何自检, 不需要 CUDA/数据/硬件")
    p.add_argument("--data-dir", help="RGB-D 目录(color.png/depth.png/meta.mat)")
    p.add_argument("--realsense", action="store_true", help="从 RealSense 实时取一帧")
    p.add_argument("--grasp-npy", help="跳过推理, 直接读已存好的 GraspGroup npy")
    p.add_argument("--baseline-root", default=os.path.expanduser("~/graspnet-baseline"),
                   help="graspnet-baseline 仓库路径")
    p.add_argument("--checkpoint", help="预训练权重, 如 checkpoint-rs.tar")
    p.add_argument("--hand-eye", help="手眼标定 4x4 矩阵的 npy; 不给则假设相机系=基座系")
    p.add_argument("--collision-thresh", type=float, default=0.01,
                   help="碰撞阈值, 设 -1 跳过碰撞检测")
    p.add_argument("--top-k", type=int, default=50, help="保留前 k 个抓取")
    p.add_argument("--gripper-max-width", type=float, default=MODEL_MAX_WIDTH,
                   help="你的夹爪最大行程(米), 超出会被裁剪")
    p.add_argument("--standoff", type=float, default=DEFAULT_STANDOFF,
                   help="预抓取点沿接近方向的后退距离(米)")
    p.add_argument("--lift", type=float, default=DEFAULT_LIFT, help="抬起高度(米)")
    p.add_argument("--dry-run", action="store_true", help="只打印动作, 不接机器人")
    p.add_argument("--vis", action="store_true", help="用 open3d 显示抓取")
    p.add_argument("--save-npy", help="把抓取结果存成 npy")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.self_test:
        return self_test()

    # ---- (1) 取图 / 或直接读现成的抓取
    gg = None
    scene_cloud = None
    if args.grasp_npy:
        from graspnetAPI import GraspGroup
        print("(1) 跳过取图, 读 %s" % args.grasp_npy)
        gg = GraspGroup().from_npy(args.grasp_npy)
        print("    %d 个抓取" % len(gg))
    else:
        print("(1) 取图")
        if args.realsense:
            frame = capture_from_realsense()
        elif args.data_dir:
            frame = load_frame_from_dir(args.data_dir)
        else:
            print("需要 --data-dir 或 --realsense 或 --grasp-npy, "
                  "或用 --self-test 只验证几何")
            return 2
        print("    %dx%d, factor_depth=%.0f" % (frame.shape[1], frame.shape[0],
                                                frame.factor_depth))

        # ---- (2) 推理
        print("(2) 推理")
        if not args.checkpoint:
            print("需要 --checkpoint。没有预训练权重, 网络参数是随机的, "
                  "输出没有意义")
            return 2
        predictor = GraspNetPredictor(args.baseline_root, args.checkpoint)
        gg, scene_cloud = predictor.predict(frame)
        print("    模型输出 %d 个抓取" % len(gg))

    # ---- (3) 后处理
    print("(3) 后处理")
    if scene_cloud is not None and args.collision_thresh > 0:
        n_before = len(gg)
        gg = filter_collisions(gg, scene_cloud, args.baseline_root,
                               collision_thresh=args.collision_thresh)
        print("    碰撞检测滤掉 %d 个, 剩 %d 个" % (n_before - len(gg), len(gg)))
    gg = postprocess(gg, top_k=args.top_k,
                     gripper_max_width=args.gripper_max_width)
    print("    NMS + 排序后剩 %d 个" % len(gg))
    if len(gg) == 0:
        print("没有可用抓取。检查 workspace_mask 和深度图质量")
        return 1

    if args.save_npy:
        gg.save_npy(args.save_npy)
        print("    已存 %s" % args.save_npy)
    if args.vis and scene_cloud is not None:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene_cloud)
        o3d.visualization.draw_geometries([pcd] + gg.to_open3d_geometry_list())

    # ---- (4) 转到基座系并执行
    print("(4) 执行")
    if args.hand_eye:
        T_base_cam = np.load(args.hand_eye)
        print("    手眼标定 %s" % args.hand_eye)
    else:
        T_base_cam = np.eye(4)
        print("    ! 没给 --hand-eye, 当作相机系就是基座系。真机上必须标定, "
              "否则位姿是错的")
    gg_base = transform_grasps_to_base(gg, T_base_cam)

    best = gg_base[0]
    print("    最高分抓取: score=%.3f width=%.3f xyz=(%.3f, %.3f, %.3f)" % (
        best.score, best.width, best.translation[0], best.translation[1],
        best.translation[2]))

    if not args.dry_run:
        print("没有接入真实机器人。实现 RobotBase 的四个方法后, 把 DryRunRobot "
              "换成你的实现, 或先用 --dry-run 看路点")
        return 2
    execute_grasp(DryRunRobot(), best, standoff=args.standoff, lift=args.lift)
    return 0


if __name__ == "__main__":
    sys.exit(main())
