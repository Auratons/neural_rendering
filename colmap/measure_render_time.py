#! /home/kremeto1/.conda/envs/pipeline/bin/python3.6
"""
This script is used for timing process of generating render of a pointcloud for given
camera position in the world. This step precedes NRIW and NRIW processing time comprises
of this time together with time for NRIW-produced final render.
"""
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
# When ran with SLURM on a multigpu node, scheduled on other than GPU0, we need
# to set this or we get an egl initialization error.
os.environ["EGL_DEVICE_ID"] = os.environ.get("SLURM_JOB_GPUS", "0").split(",")[0]

import argparse
import multiprocessing
import numpy as np
import pyrender
import random
import socket
from timeit import default_timer as timer
import time

from load_data import load_cameras_colmap, get_colmap_file, load_ply, none_or_float
import read_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--colmap_sparse_dir",
        required=True,
        type=str,
        help="Folder with colmap .bin output files with camera poses.",
    )
    parser.add_argument(
        "-l",
        "--ply_path",
        required=True,
        type=str,
        help="Point cloud or mesh file to render.",
    )
    parser.add_argument(
        "-p",
        "--point_size",
        type=float,
        default=2.0,
        help="Point size for point-splatting rendering.",
    )
    parser.add_argument(
        "-r",
        "--render_size",
        type=int,
        default=512,
        help="Side length of a square image to render.",
    )
    parser.add_argument(
        "-v",
        "--voxel_size",
        type=none_or_float,
        default=None,
        help="Voxel size used for downsampling mesh or pointcloud.",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=100,
        help="The number of renderings used for render time measurement.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    print(f"Running on: {socket.gethostname()}")
    args = parse_args()

    random.seed = 42

    print("Loading cameras")
    K, R, T, H, W, src_img_nms = load_cameras_colmap(
        get_colmap_file(args.colmap_sparse_dir, "images"),
        get_colmap_file(args.colmap_sparse_dir, "cameras"),
    )
    print("Loading ply file")
    mesh = load_ply(args.ply_path, args.voxel_size)

    # If args.count <=0 or > number of camera poses at hand, render all poses.
    count = min((args.count if args.count > 0 else len(K)), len(K))
    flags = pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.RGBA
    # Randomly sample views to render
    indices = list(range(len(K)))
    random.shuffle(indices)
    indices = indices[:count]
    thread_times = [0.0] * len(indices)
    process_times = [0.0] * len(indices)

    for idx, idx_to_render in enumerate(indices):
        # print(f"Rendering image {idx}/{count}")
        i = idx_to_render
        k, r, t, w, h, img_nm = K[i], R[i], T[i], W[i], H[i], src_img_nms[i]

        thread_start = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        process_start = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)

        scene = pyrender.Scene()
        scene.add(mesh)
        camera = pyrender.camera.IntrinsicsCamera(k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = r.T
        camera_pose[:3, -1:] = -r.T @ t
        camera_pose[:, 1:3] *= -1
        scene.add(camera, pose=camera_pose)

        # Offscreen rendering
        r = pyrender.OffscreenRenderer(
            args.render_size, args.render_size, point_size=args.point_size
        )
        _ = r.render(scene, flags=flags)

        thread_end = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        process_end = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)

        thread_times[idx] = thread_end - thread_start
        process_times[idx] = process_end - process_start

    thread_mean = np.mean(thread_times) * 1000
    thread_std = np.std(thread_times) * 1000
    process_mean = np.mean(process_times) * 1000
    process_std = np.std(process_times) * 1000
    cpus = os.environ.get("SLURM_CPUS_ON_NODE", multiprocessing.cpu_count())
    print()
    print(f"Rendered:\n{args}\n\n")
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
