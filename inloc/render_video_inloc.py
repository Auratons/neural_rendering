""" Script to render the InLoc Video Supplementary dataset. """

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import pyrender
import trimesh
import cv2
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from pyrender.constants import RenderFlags
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from PIL import Image
from tqdm import tqdm

from render_inloc_db import load_point_cloud, get_path_name, load_initial_transform

ZNEAR = 0.05
ZFAR = 100.0
FLAGS = RenderFlags.FLAT | RenderFlags.RGBA
ROT_ALIGN_QUERY = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)


def invert_T(T):
    """Invert a 4x4 transformation matrix."""
    T_inv = np.eye(4)
    r, t = T[:3, :3], T[:3, -1]
    T_inv[:3, :3] = r.T
    T_inv[:3, -1] = -r.T @ t
    return T_inv


def parse_inloc_video(mat_path, video_ids=None):
    """Returns a dictionary {scan: [(query, P, camera_model, w, h, camera_params), ...]}."""
    matfile = loadmat(mat_path)["gtlist_video"][0]
    res = dict()
    for query in matfile:
        query_name = query[0][0]
        P = query[1]
        scan = query[3][0]
        building, scan = scan.split(".")[0].split("/")[-2:]
        scan = scan.split("_")[-1]
        scan = "/".join([building, scan])
        camera_info = query[2].item()
        camera_model = camera_info[1].item()
        w, h = camera_info[2].item(), camera_info[3].item()
        camera_params = camera_info[4].flatten()
        if scan not in res:
            res[scan] = [(query_name, P, camera_model, w, h, camera_params)]
        else:
            res[scan].append((query_name, P, camera_model, w, h, camera_params))
    return res


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    print("Grouping the query images per scan...")
    grouped_queries = parse_inloc_video(args.mat_path)
    for scan, queries in grouped_queries.items():
        print("Processing scan {}...".format(scan))
        # Load the pointcloud once
        building, scan = scan.split("/")
        ptx_path = os.path.join(
            args.db_path,
            "scans",
            building,
            "{}_scan_{}.ptx.mat".format(get_path_name(building), scan),
        )
        xyz, rgb = load_point_cloud(ptx_path, n_max=args.n_max_per_scan)
        transform_path = os.path.join(
            args.db_path,
            "alignments/{}_alignment/transformations/{}_trans_{}.txt".format(
                building,
                get_path_name(building),
                scan,
            ),
        )
        T_cutout = load_initial_transform(transform_path)
        xyz = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        xyz = xyz @ T_cutout.T
        xyz = xyz[:, :3] / xyz[:, -1:]

        # Setup the scene
        mesh = pyrender.Mesh.from_points(xyz, rgb)
        scene = pyrender.Scene()
        scene.add(mesh)

        for query in tqdm(queries):
            # Load and resize the reference image
            query_name, T_query, camera_model, w, h, camera_params = query
            T_query = np.concatenate([T_query, np.array([[0, 0, 0, 1]])], axis=0)
            query_img = Image.open(os.path.join(args.query_path, query_name))
            w, h = query_img.size
            ratio = float(args.max_img_size) / max(w, h)
            query_img = query_img.resize(
                (int(ratio * w), int(ratio * h)), Image.ANTIALIAS
            )

            # Setup the camera
            fl, cx, cy, _ = camera_params
            fl, cx, cy = ratio * fl, int(ratio * cx), int(ratio * cy)
            camera = pyrender.IntrinsicsCamera(fl, fl, cx, cy, znear=ZNEAR, zfar=ZFAR)
            camera_pose = invert_T(T_query)
            camera_pose[:3, :3] = camera_pose[:3, :3] @ ROT_ALIGN_QUERY
            camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
            scene.add_node(camera_node)

            # Render the view
            renderer = pyrender.OffscreenRenderer(
                int(ratio * w), int(ratio * h), point_size=args.point_size
            )
            rendering, depth = renderer.render(scene, flags=FLAGS)
            color = np.array(rendering.copy(), dtype=np.uint8)
            renderer.delete()

            # Discard points where depth is too large
            mask = depth > args.max_depth
            depth[mask] = 0.0
            color[mask, :] = 255

            # Save the images
            base_name = query_name[:-4]  # remove the .png
            rendered_img = Image.fromarray(color)
            os.makedirs(
                os.path.join(args.output_path, query_name.split("/")[0]), exist_ok=True
            )
            query_img.save(
                os.path.join(args.output_path, "{}_reference.png".format(base_name))
            )
            rendered_img.save(
                os.path.join(args.output_path, "{}_color.png".format(base_name))
            )
            plt.imsave(
                os.path.join(args.output_path, "{}_depth.png".format(base_name)),
                depth,
                cmap="gray",
            )

            # Remove the camera node
            scene.remove_node(camera_node)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_path", type=str, default="/home/bdechamps/InLoc_dataset/database/"
    )
    parser.add_argument(
        "--query_path", type=str, default="/home/bdechamps/InLoc_video_share/RGB/"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/bdechamps/datasets/inloc_video_rendered_sample/",
    )
    parser.add_argument(
        "--mat_path",
        type=str,
        default="/home/bdechamps/InLoc_video_share/gtlist_video.mat",
    )

    # Rendering parameters
    parser.add_argument("--n_max_per_scan", type=int, default=10000000)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument("--max_img_size", type=int, default=1024)
    parser.add_argument("--nvidia_id", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ["EGL_DEVICE_ID"] = str(args.nvidia_id)
    main(args)
