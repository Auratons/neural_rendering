##############################################################################################
# This script generates the proper input dataset for the Neural Rerendering in the Wild model
# and for other renderers than pyrender that generate color renders used for training NRIW.
##################<############################################################################
import argparse
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from scipy.io import loadmat
import pymeshlab
import portalocker

WIDTH = 1600
HEIGHT = 1200
FL = 800 / np.tan(np.pi / 6.0)
CX = 800
CY = 600
ZNEAR = 0.05
ZFAR = 100.0
BUILDINGS = ["CSE3", "CSE4", "CSE5", "DUC1", "DUC2"]


# Coordinate changes from InLoc transformations to C++ renders
ROT_ALIGN = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float)


def link(src, dst):
    if not dst.exists():
        os.link(src, dst)


def load_point_cloud(ptx_path, n_max=None):
    ptx = loadmat(ptx_path)
    n_col, n_row, A = ptx["Ncol"].item(), ptx["Nrow"].item(), ptx["A"]
    xyz = np.concatenate(list(A[0, :3]), axis=1)
    rgb = np.concatenate(list(A[0, 4:]), axis=1)
    if n_max is not None:
        n_max = min(n_max, n_col * n_row)
        shuffle = np.random.choice(
            np.arange(n_col * n_row), n_col * n_row, replace=False
        )[:n_max]
        xyz = xyz[shuffle, :]
        rgb = rgb[shuffle, :]
    return xyz, rgb


def load_initial_transform(transform_path):
    """Load the transformation matrix contained in the `xxx_trans_xxx.txt` file."""
    transform = []
    with open(transform_path, "r") as f:
        for line in f.readlines()[7:-1]:
            transform.append([float(x) for x in line.split()])
    return np.array(transform, dtype=np.float)


def get_rotation_matrix(cutout_name):
    """Get the rotation matrix associated with the frame of an InLoc panorama."""
    angles = cutout_name.split(".")[0].split("_")[-2:]
    # Get the angle in degrees and convert them in radians
    theta_1, theta_2 = float(angles[0]), float(angles[1])
    theta_1 *= np.pi / 180.0
    theta_2 *= np.pi / 180.0
    # Get the rotation matrix
    r_1 = R.from_rotvec([0, -theta_1, 0]).as_matrix()
    r_2 = R.from_rotvec([theta_2, 0, 0]).as_matrix()
    r = r_1 @ r_2
    return r


def get_path_name(building):
    if "CSE" in building:
        return "cse"
    if "DUC" in building:
        return "DUC"


def render_scan(args, building, scan):
    os.makedirs(os.path.join(args.output_path, building, scan), exist_ok=True)
    matrices_dict = {"train": {}, "val": {}}

    building_name = get_path_name(building)
    # Load the scan
    transform_path = (
        args.inloc_path
        / f"database/alignments/{building}/transformations"
        / f"{building_name}_trans_{scan}.txt"
    )
    ptx_path = (
        args.inloc_path
        / f"database/scans/{building}"
        / f"{building_name}_scan_{scan}.ptx.mat"
    )
    xyz, rgb = load_point_cloud(str(ptx_path), n_max=args.n_max_per_scan)
    T = load_initial_transform(transform_path)

    # Transform the points in homogeneous coordinates
    xyz = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    xyz = (T @ xyz.T).T
    xyz = xyz[:, :3] / xyz[:, -1:]

    # Estimate normals for Points Splatting
    rgb = np.concatenate([rgb, np.ones((rgb.shape[0], 1))], axis=1)
    rgb = rgb.astype(np.float64) / 255
    mesh = pymeshlab.Mesh(vertex_matrix=xyz, v_color_matrix=rgb)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    ms.compute_normals_for_point_sets(flipflag=True, viewpos=T[:3, 3])
    ms.save_current_mesh(
        str(
            args.output_path / building / scan / f"{building_name}_scan_{scan}.ptx.ply"
        ),
        save_vertex_normal=True,
    )
    del mesh
    del ms

    # Setup the scene
    ratio = float(args.width) / WIDTH
    height = int(HEIGHT * ratio)
    fl = int(FL * ratio)
    cx, cy = int(CX * ratio), int(CY * ratio)

    intristic_camera = np.eye(4)
    intristic_camera[0, 0] = fl
    intristic_camera[1, 1] = fl
    intristic_camera[0, 2] = cx
    intristic_camera[1, 2] = cy

    cutout_path = args.inloc_path / "database/cutouts" / building / scan
    for cutout_name in cutout_path.iterdir():
        if cutout_name.suffix == ".jpg":
            cutout_name = cutout_name.name
            r = get_rotation_matrix(str(cutout_name))
            camera_pose = T.copy()
            camera_pose[:3, :3] = camera_pose[:3, :3] @ ROT_ALIGN @ r

            # Save the images
            base_name = str(cutout_name).split(".")[0]
            error = False
            try:
                link(
                    args.inloc_rendered_by_pyrender
                    / building
                    / scan
                    / f"{base_name}_reference.png",
                    args.output_path / building / scan / f"{base_name}_reference.png",
                )
            except FileNotFoundError as err:
                error = True
                print(f"Unexpected {type(err)}: {err}")
            if not error:
                matrices_dict["train"][
                    str(args.output_path / building / scan / f"{base_name}_color.png")
                ] = {
                    "calibration_mat": [list(i) for i in list(intristic_camera)],
                    "camera_pose": [list(i) for i in camera_pose],
                }

    with open(
        args.output_path / building / scan / "matrices_for_rendering.txt", "w"
    ) as f:
        json.dump(matrices_dict, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inloc_path", type=Path, help="Raw unzipped dataset path.")
    parser.add_argument(
        "--inloc_rendered_by_pyrender",
        type=Path,
        help="We're only linking other files then color renders, serves for source of links.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Output root where similar structure as for inloc_rendered_by_pyrender is generated.",
    )
    # Rendering parameters
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--n_max_per_scan", type=int, default=None)
    args = parser.parse_args()
    return args


def main(args):
    # pool = multiprocessing.Pool(16)
    for building in BUILDINGS:
        # pool.map(render_scan, [(args, building, i.name) for i in (args.inloc_path / "database/cutouts" / building).iterdir()])
        for scan in (args.inloc_path / "database/cutouts" / building).iterdir():
            scan = scan.name
            try:
                lock_path = str(args.output_path / building / f"{scan}.LOCK")
                # with portalocker.Lock(lock_path) as ff:
                # if not (args.output_path / building / scan.name / "matrices_for_rendering.txt").exists():
                # if not (
                #     args.output_path
                #     / building
                #     / scan
                #     / f"{get_path_name(building)}_scan_{scan}_30M.ptx.ply"
                # ).exists():
                print(f"{building}/{scan}")
                render_scan(args, building, scan)
            except portalocker.exceptions.LockException:
                print(f"Lock {lock_path} already locked, skipping")
                continue


if __name__ == "__main__":
    args = parse_args()
    main(args)
