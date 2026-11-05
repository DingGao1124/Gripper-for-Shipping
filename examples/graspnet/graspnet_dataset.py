"""GraspNet-1Billion 数据集可视化 demo。

对应官方文档 "Visualization of Dataset" 一节的四个例子，做了两点调整：

1. 用 split='custom' 只加载需要的场景。官方示例写 split='train'，而 GraspNet.__init__
   会遍历该 split 的每个场景读 object_id_list.txt，那样要求 100 个场景全部就位。
2. 跑之前先检查文件，缺什么直接报出来，而不是等到中途抛 FileNotFoundError。

┌──────────────────────┬──────────────────────────────────────────────────┬────────┐
│         demo         │                  required dataset                |  Size  │
├──────────────────────┼──────────────────────────────────────────────────┼────────┤
│ showObjGrasp         │ models + grasp_label + train_1                   │ ~26 GB │
├──────────────────────┼──────────────────────────────────────────────────┼────────┤
│ show6DPose           │ models + train_1                                 │ ~24 GB │
├──────────────────────┼──────────────────────────────────────────────────┼────────┤
│ showSceneGrasp(rect) │ train_1 + rect_labels                            │ ~33 GB │
├──────────────────────┼──────────────────────────────────────────────────┼────────┤
│ showSceneGrasp(6d)   │ models + grasp_label + collision_label + train_1 │ ~27 GB │
└──────────────────────┴──────────────────────────────────────────────────┴────────┘

用法:
    python examples/vis_dataset.py --check          # 只检查数据完整性
    python examples/vis_dataset.py obj              # showObjGrasp
    python examples/vis_dataset.py pose             # show6DPose
    python examples/vis_dataset.py rect             # showSceneGrasp(format='rect')
    python examples/vis_dataset.py scene6d          # showSceneGrasp(format='6d')

数据集根目录通过 GRASPNET_ROOT 环境变量指定，或改下面的默认值。
"""

import argparse
import os
import sys

# 数据集根目录。目录结构见 https://graspnet.net/datasets.html
# 默认放在仓库的 data/graspnet 下, 已在 .gitignore 里忽略
_DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "graspnet")
GRASPNET_ROOT = os.environ.get("GRASPNET_ROOT", _DEFAULT_ROOT)

# 四个 demo 各自用到的场景/物体，同时也是检查清单的依据
OBJ_DEMO_ID = 0
POSE_DEMO_SCENE = 0
RECT_DEMO_SCENE, RECT_DEMO_CAMERA, RECT_DEMO_ANN = 0, "realsense", 0
SCENE6D_SCENE, SCENE6D_CAMERA, SCENE6D_ANN = 4, "kinect", 2


def scene_dir(scene_id, camera=None):
    parts = [GRASPNET_ROOT, "scenes", "scene_%04d" % scene_id]
    if camera:
        parts.append(camera)
    return os.path.join(*parts)


def scene_objects(scene_id):
    """读场景的 object_id_list.txt，拿到该场景包含哪些物体。"""
    path = os.path.join(scene_dir(scene_id), "object_id_list.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def required_files(demo):
    """列出某个 demo 需要的文件。返回 (路径, 来自哪个下载包) 的列表。"""
    need = []
    if demo == "obj":
        # visObjGrasp: 物体点云 + 抓取标签 + 一个用来定视角的相机外参
        need.append((os.path.join(GRASPNET_ROOT, "models", "%03d" % OBJ_DEMO_ID,
                                  "nontextured.ply"), "models.zip"))
        need.append((os.path.join(GRASPNET_ROOT, "grasp_label",
                                  "%03d_labels.npz" % OBJ_DEMO_ID), "grasp_label.zip"))
        need.append((os.path.join(scene_dir(0, "kinect"), "cam0_wrt_table.npy"),
                     "train_1.zip"))

    elif demo == "pose":
        s, cam = POSE_DEMO_SCENE, "kinect"
        # 点云由 rgb + depth + camK.npy 重建; meta/*.mat 这里用不到
        for rel, pkg in [("cam0_wrt_table.npy", "train_1.zip"),
                         ("camera_poses.npy", "train_1.zip"),
                         ("camK.npy", "train_1.zip"),
                         ("annotations/0000.xml", "train_1.zip"),
                         ("rgb/0000.png", "train_1.zip"),
                         ("depth/0000.png", "train_1.zip")]:
            need.append((os.path.join(scene_dir(s, cam), rel), pkg))
        need.append((os.path.join(scene_dir(s), "object_id_list.txt"), "train_1.zip"))
        for obj_id in scene_objects(s) or []:
            need.append((os.path.join(GRASPNET_ROOT, "models", "%03d" % obj_id,
                                      "nontextured.ply"), "models.zip"))

    elif demo == "rect":
        s, cam, ann = RECT_DEMO_SCENE, RECT_DEMO_CAMERA, RECT_DEMO_ANN
        need.append((os.path.join(scene_dir(s, cam), "rgb", "%04d.png" % ann),
                     "train_1.zip"))
        # 矩形标签是单独的可选下载，解压后还要跑 copy_rect_labels.py 分发到各场景
        need.append((os.path.join(scene_dir(s, cam), "rect", "%04d.npy" % ann),
                     "rect_labels.zip (+ copy_rect_labels.py)"))
        need.append((os.path.join(scene_dir(s), "object_id_list.txt"), "train_1.zip"))

    elif demo == "scene6d":
        s, cam, ann = SCENE6D_SCENE, SCENE6D_CAMERA, SCENE6D_ANN
        for rel, pkg in [("camera_poses.npy", "train_1.zip"),
                         ("cam0_wrt_table.npy", "train_1.zip"),
                         ("camK.npy", "train_1.zip"),
                         ("annotations/%04d.xml" % ann, "train_1.zip"),
                         ("rgb/%04d.png" % ann, "train_1.zip"),
                         ("depth/%04d.png" % ann, "train_1.zip")]:
            need.append((os.path.join(scene_dir(s, cam), rel), pkg))
        need.append((os.path.join(scene_dir(s), "object_id_list.txt"), "train_1.zip"))
        need.append((os.path.join(GRASPNET_ROOT, "collision_label",
                                  "scene_%04d" % s, "collision_labels.npz"),
                     "collision_label.zip"))
        for obj_id in scene_objects(s) or []:
            need.append((os.path.join(GRASPNET_ROOT, "grasp_label",
                                      "%03d_labels.npz" % obj_id), "grasp_label.zip"))
            need.append((os.path.join(GRASPNET_ROOT, "models", "%03d" % obj_id,
                                      "nontextured.ply"), "models.zip"))
    return need


def check(demo):
    """检查 demo 所需文件，返回缺失列表并打印结果。"""
    need = required_files(demo)
    missing = [(p, pkg) for p, pkg in need if not os.path.exists(p)]

    print("[%s] 需要 %d 个文件, 缺 %d 个" % (demo, len(need), len(missing)))
    if not missing:
        print("  数据齐全")
        return missing

    # 按来源的下载包归类，这样能直接看出该下哪个 zip
    by_pkg = {}
    for path, pkg in missing:
        by_pkg.setdefault(pkg, []).append(path)
    for pkg, paths in sorted(by_pkg.items()):
        print("  缺 %d 个, 来自 %s" % (len(paths), pkg))
        for path in paths[:3]:
            print("    %s" % os.path.relpath(path, GRASPNET_ROOT))
        if len(paths) > 3:
            print("    ... 另外 %d 个" % (len(paths) - 3))
    return missing


def make_graspnet(scene_ids, camera):
    """用 custom split 建 GraspNet 实例，只加载指定场景。"""
    from graspnetAPI import GraspNet
    return GraspNet(GRASPNET_ROOT, camera=camera, split="custom", sceneIds=scene_ids)


def demo_obj():
    g = make_graspnet([0], "kinect")
    g.showObjGrasp(objIds=OBJ_DEMO_ID, show=True)


def demo_pose():
    g = make_graspnet([POSE_DEMO_SCENE], "kinect")
    g.show6DPose(sceneIds=POSE_DEMO_SCENE, show=True)


def demo_rect():
    g = make_graspnet([RECT_DEMO_SCENE], RECT_DEMO_CAMERA)
    g.showSceneGrasp(sceneId=RECT_DEMO_SCENE, camera=RECT_DEMO_CAMERA,
                     annId=RECT_DEMO_ANN, format="rect", numGrasp=20)


def demo_scene6d():
    # 这个要现场算抓取姿态, 会跑几分钟
    g = make_graspnet([SCENE6D_SCENE], SCENE6D_CAMERA)
    g.showSceneGrasp(sceneId=SCENE6D_SCENE, camera=SCENE6D_CAMERA,
                     annId=SCENE6D_ANN, format="6d")


DEMOS = {"obj": demo_obj, "pose": demo_pose, "rect": demo_rect, "scene6d": demo_scene6d}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demo", nargs="?", choices=sorted(DEMOS), help="要跑哪个 demo")
    parser.add_argument("--check", action="store_true", help="只检查数据, 不跑可视化")
    args = parser.parse_args()

    print("GRASPNET_ROOT = %s" % GRASPNET_ROOT)
    if not os.path.isdir(GRASPNET_ROOT):
        print("根目录不存在。用 GRASPNET_ROOT 环境变量指定, 数据集下载见 "
              "https://graspnet.net/datasets.html")
        return 1

    if args.check or args.demo is None:
        missing = sum(len(check(d)) for d in sorted(DEMOS))
        return 1 if missing else 0

    if check(args.demo):
        print("数据不全, 跳过可视化")
        return 1
    DEMOS[args.demo]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
