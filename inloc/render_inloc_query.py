##############################################################################################
# Pyrender headless rendering for the Inloc database data.
# This script generates the proper input dataset for the Neural Rerendering in the Wild model.
##############################################################################################

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
# When ran with SLURM on a multigpu node, scheduled on other than GPU0, we need
# to set this or we get an egl initialization error.
os.environ["EGL_DEVICE_ID"] = os.environ.get("SLURM_JOB_GPUS", "0").split(",")[0]

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import pyrender
from PIL import Image
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import sys

sys.path.append(f"{str(Path(__file__).parent)}/../")
import utils

from render_inloc_db import (
    FLAGS,
    get_path_name,
    load_initial_transform,
    load_point_cloud,
)

# IPhone7 intrinsics
FL = 4032 * 28.0 / 36.0
CX = 2016
CY = 1512
ZNEAR = 0.05
ZFAR = 100.0
ROT_ALIGN_QUERY = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)


def invert_T(T):
    """Invert a 4x4 transformation matrix."""
    T_inv = np.eye(4)
    r, t = T[:3, :3], T[:3, -1]
    T_inv[:3, :3] = r.T
    T_inv[:3, -1] = -r.T @ t
    return T_inv


def load_query_outputs(mat_path):
    """Returns the topK .mat file in a dictionary {query_img: [(cutout1, pose1), ...]}."""
    query_mat = loadmat(mat_path)["ImgList"][0]
    query_outputs = dict()
    for res in query_mat:
        query_img = res[0][0]
        ranking = [arr[0] for arr in res[1][0]]
        pose_matrices = list(res[3][0])
        query_outputs[query_img] = list(zip(ranking, pose_matrices))
    return query_outputs


def group_queries_per_scan(mat_path):
    """Returns a dictionary {scan: {query: [rank, transformation]}} to group the rendering
    process per scan."""
    mat_file = load_query_outputs(mat_path)
    scan2queries = dict()
    for query, ranking in mat_file.items():
        for rank, (cutout, T_query) in enumerate(ranking):
            scan = "/".join(cutout.split("/")[:2])
            if scan not in scan2queries:
                scan2queries[scan] = dict()
            if query not in scan2queries[scan]:
                scan2queries[scan][query] = [(rank, T_query)]
            else:
                scan2queries[scan][query].append((rank, T_query))
    return scan2queries


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    print("Grouping the query images per scan...")
    grouped_queries = group_queries_per_scan(args.mat_path)
    for scan, queries in grouped_queries.items():
        print("Processing scan {}...".format(scan))
        # Load the pointcloud once
        building, scan = scan.split("/")
        ptx_path = (
            args.inloc_path
            / "database"
            / "scans"
            / building
            / f"{get_path_name(building)}_scan_{scan}.ptx.mat"
        )
        xyz, rgb = load_point_cloud(ptx_path, n_max=args.n_max_per_scan)
        transform_path = (
            args.inloc_path
            / "database"
            / "alignments"
            / f"{building}"
            / "transformations"
            / f"{get_path_name(building)}_trans_{scan}.txt"
        )
        T_cutout = load_initial_transform(transform_path)
        xyz = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        xyz = xyz @ T_cutout.T
        xyz = xyz[:, :3] / xyz[:, -1:]

        # Setup the scene
        mesh = pyrender.Mesh.from_points(xyz, rgb)
        scene = pyrender.Scene()
        scene.add(mesh)
        # camera = pyrender.IntrinsicsCamera(FL, FL, CX, CY, znear=ZNEAR, zfar=ZFAR)
        # camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
        # scene.add_node(camera_node)

        for query, poses in tqdm(queries.items()):
            # Load and resize the reference image
            query_img = Image.open(args.query_path / query)
            w, h = query_img.size
            ratio = float(args.max_img_size) / max(w, h)
            query_img = query_img.resize(
                (int(ratio * w), int(ratio * h)), Image.ANTIALIAS
            )
            if args.squarify:
                query_img = utils.squarify(np.array(query_img), args.max_img_size)
                query_img = Image.fromarray(query_img)
            fl = FL * ratio
            cx, cy = int(w * ratio / 2.0), int(h * ratio / 2.0)
            camera = pyrender.IntrinsicsCamera(fl, fl, cx, cy, znear=ZNEAR, zfar=ZFAR)
            # camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
            # scene.add_node(camera_node)
            renderer = pyrender.OffscreenRenderer(
                int(ratio * w), int(ratio * h), point_size=args.point_size
            )

            for rank, T_query in poses:
                T_query = np.concatenate([T_query, np.array([[0, 0, 0, 1]])], axis=0)
                T_query_inv = invert_T(T_query)
                camera_pose = T_query_inv.copy()
                camera_pose[:3, :3] = camera_pose[:3, :3] @ ROT_ALIGN_QUERY

                # # Change the camera
                # scene.remove_node(camera_node)
                try:
                    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
                    scene.add_node(camera_node)
                    # scene.set_pose(camera_node, camera_pose)

                    # Render the view
                    rendering, depth = renderer.render(scene, flags=FLAGS)
                    color = np.array(rendering.copy(), dtype=np.uint8)

                    # Discard points where depth is too large
                    mask = depth > args.max_depth
                    depth[mask] = 0.0
                    color[mask, :] = 255
                    if args.squarify:
                        rendered_img = Image.fromarray(
                            utils.squarify(color, args.max_img_size)
                        )
                        depth = utils.squarify(depth, args.max_img_size)
                    else:
                        rendered_img = Image.fromarray(color)

                    # Save the images
                    os.makedirs(args.output_path / query, exist_ok=True)
                    query_img.save(
                        args.output_path / query / f"{rank:04n}_reference.png"
                    )
                    rendered_img.save(
                        args.output_path / query / f"{rank:04n}_color.png"
                    )
                    # cv2.imwrite saves depth map as a single channel img and as-is meaning
                    # if max depth is x, then max of the saved img values will be x as well.
                    # skimage.io.imsave saves the image normalized, so max value will always
                    # be 255 or 65k depending on the data type. plt.imsave saves RGBA image
                    # with depth remapped to a color depending on a colormap used.
                    cv2.imwrite(
                        str(args.output_path / query / f"{rank:04n}_depth.png"),
                        depth.astype(np.uint16),
                    )
                    # For possible further recalculation, npz with raw depth map is also saved.
                    np.save(
                        args.output_path / query / f"{rank:04n}_depth.npy",
                        depth,
                    )

                    scene.remove_node(camera_node)

                except np.linalg.LinAlgError:
                    warnings.warn(
                        "numpy.linalg.LinAlgError raised : skipping pose {} for image {}".format(
                            rank, query
                        )
                    )

            renderer.delete()


def none_or_int(value):
    """Simplifies unification in SLURM script's parameter handling."""
    if value == "None":
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inloc_path", type=Path)
    parser.add_argument("--query_path", type=Path)
    parser.add_argument(
        "--output_path",
        type=Path,
        default="/home/bdechamps/datasets/query_4032_borders/",
    )
    parser.add_argument(
        "--mat_path",
        type=Path,
        default="/home/bdechamps/InLoc_demo/outputs/densePE_top100_shortlist.mat",
    )

    # Rendering parameters
    parser.add_argument("--n_max_per_scan", type=none_or_int, default=None)
    parser.add_argument("--point_size", type=float, default=5.0)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument("--max_img_size", type=int, default=4032)
    parser.add_argument("--nvidia_id", type=int, default=0)
    parser.add_argument("--squarify", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
