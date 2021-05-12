##############################################################################################
# Pyrender headless rendering for the Inloc query data.
# This script generates the proper input dataset for the Neural Rerendering in the Wild model.
##############################################################################################

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
# When ran with SLURM on a multigpu node, scheduled on other than GPU0, we need
# to set this or we get an egl initialization error.
os.environ["EGL_DEVICE_ID"] = os.environ.get("SLURM_JOB_GPUS", "0").split(",")[0]

import argparse
import multiprocessing
import pyrender
import trimesh
import time
import cv2
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from pyrender.constants import RenderFlags
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from PIL import Image
from tqdm import tqdm

WIDTH = 1600
HEIGHT = 1200
FL = 800 / np.tan(np.pi / 6.0)
CX = 800
CY = 600
ZNEAR = 0.05
ZFAR = 100.0
FLAGS = RenderFlags.FLAT | RenderFlags.RGBA
BUILDINGS = ["CSE3", "CSE4", "CSE5", "DUC1", "DUC2"]


# Coordinate changes from InLoc transformations to pyrender
ROT_ALIGN = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float)


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
    # Load the scan
    transform_path = os.path.join(
        args.inloc_path,
        "database/alignments/{}/transformations/{}_trans_{}.txt".format(
            building,
            get_path_name(building),
            scan,
        ),
    )
    ptx_path = os.path.join(
        args.inloc_path,
        "database/scans/{}/{}_scan_{}.ptx.mat".format(
            building,
            get_path_name(building),
            scan,
        ),
    )
    xyz, rgb = load_point_cloud(ptx_path, n_max=args.n_max_per_scan)
    T = load_initial_transform(transform_path)

    # Transform the points in homogeneous coordinates
    xyz = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    xyz = (T @ xyz.T).T
    xyz = xyz[:, :3] / xyz[:, -1:]

    # Setup the scene
    ratio = float(args.width) / WIDTH
    height = int(HEIGHT * ratio)
    fl = int(FL * ratio)
    cx, cy = int(CX * ratio), int(CY * ratio)
    mesh = pyrender.Mesh.from_points(xyz, rgb)

    thread_times = []
    process_times = []

    # Render from the cutouts views
    cutout_path = os.path.join(
        args.inloc_path,
        "database/cutouts/{}/{}".format(building, scan),
    )
    os.makedirs(os.path.join(args.output_path, building, scan), exist_ok=True)
    for cutout_name in tqdm(os.listdir(cutout_path)):
        if cutout_name[-3:] == "jpg":
            bg_color = (
                [float(i) for i in args.bg_color.split(",")]
                if args.bg_color is not None
                else None
            )

            thread_start = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
            process_start = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)

            scene = pyrender.Scene(bg_color=bg_color)
            scene.add(mesh)
            camera = pyrender.IntrinsicsCamera(fl, fl, cx, cy, znear=ZNEAR, zfar=ZFAR)
            camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
            scene.add_node(camera_node)
            r = get_rotation_matrix(cutout_name)
            camera_pose = T.copy()
            camera_pose[:3, :3] = camera_pose[:3, :3] @ ROT_ALIGN @ r
            scene.set_pose(camera_node, camera_pose)
            renderer = pyrender.OffscreenRenderer(args.width, height, args.point_size)

            # Render the view
            rendering, depth = renderer.render(scene, flags=FLAGS)

            thread_end = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
            process_end = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)

            thread_times.append(thread_end - thread_start)
            process_times.append(process_end - process_start)

            renderer.delete()
            color = np.array(rendering.copy(), dtype=np.uint8)

            # Discard points where depth is too large
            mask = depth > args.max_depth
            depth[mask] = 0.0
            color[mask, :] = 255

            # Load and resize the reference image
            reference_img = Image.open(os.path.join(cutout_path, cutout_name))
            reference_img = reference_img.resize((args.width, height), Image.ANTIALIAS)

            # Save the images
            base_name = cutout_name.split(".")[0]
            rendered_img = Image.fromarray(color)
            reference_img.save(
                os.path.join(
                    args.output_path,
                    building,
                    scan,
                    "{}_reference.png".format(base_name),
                )
            )
            rendered_img.save(
                os.path.join(
                    args.output_path, building, scan, "{}_color.png".format(base_name)
                )
            )
            # cv2.imwrite saves depth map as a single channel img and as-is meaning
            # if max depth is x, then max of the saved img values will be x as well.
            # skimage.io.imsave saves the image normalized, so max value will always
            # be 255 or 65k depending on the data type. plt.imsave saves RGBA image
            # with depth remapped to a color depending on a colormap used.
            cv2.imwrite(
                os.path.join(
                    args.output_path, building, scan, "{}_depth.png".format(base_name)
                ),
                depth.astype(np.uint16),
            )
            # For possible further recalculation, npz with raw depth map is also saved.
            np.save(
                os.path.join(
                    args.output_path, building, scan, "{}_depth.npy".format(base_name)
                ),
                depth,
            )
    thread_mean = np.mean(thread_times) * 1000
    thread_std = np.std(thread_times) * 1000
    process_mean = np.mean(process_times) * 1000
    process_std = np.std(process_times) * 1000
    cpus = os.environ.get("SLURM_CPUS_ON_NODE", multiprocessing.cpu_count())
    print()
    print(f"Rendered:\n{building}, {scan}, {args}\n\n")
    print(
        f"CPU thread time per render: ({thread_mean:.2f}+-{thread_std:.2f}) ms, {thread_std*100/thread_mean:.2f} %"
    )
    # Count of processors available to the job on this node. Note the
    # select/linear plugin allocates entire nodes to jobs, so the value
    # indicates the total count of CPUs on the node. For the select/cons_res
    # plugin, this number indicates the number of cores on this node
    # allocated to the job.
    print(
        f"CPU process time per render ({cpus} CPUS): ({process_mean:.2f}+-{process_std:.2f}) ms, {process_std*100/process_mean:.2f} %"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inloc_path", type=str, default="/home/bdechamps/InLoc_dataset/"
    )
    parser.add_argument(
        "--output_path", type=str, default="/home/bdechamps/datasets/InLoc_rendered/"
    )
    # Rendering parameters
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--n_max_per_scan", type=int, default=None)
    parser.add_argument("--point_size", type=float, default=1.0)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument(
        "--bg_color",
        type=str,
        default=None,
        help="Background comma separated color for rendering.",
    )
    args = parser.parse_args()
    return args


def main(args):
    for building in BUILDINGS:
        print("Building dataset for {}".format(building))
        for scan in os.listdir(
            os.path.join(args.inloc_path, "database/cutouts/", building)
        ):
            print("scan {}".format(scan))
            render_scan(args, building, scan)


if __name__ == "__main__":
    args = parse_args()
    main(args)
