import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
# When ran with SLURM on a multigpu node, scheduled on other than GPU0, we need
# to set this or we get an egl initialization error.
batch_mode = os.environ.get("SLURM_JOB_GPUS", "").split(",")[0]
interactive_mode = os.environ.get("SLURM_STEP_GPUS", "").split(",")[0]
os.environ["EGL_DEVICE_ID"] = "".join([batch_mode, interactive_mode])
if os.environ["EGL_DEVICE_ID"] == "":
    os.environ["EGL_DEVICE_ID"] = "0"

import argparse
import distutils.util
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pyrender
import scipy.io as sio
import trimesh
from PIL import Image
from pyrender.constants import RenderFlags

import read_model


def squarify_image(image: np.array, square_size: int) -> np.array:
    assert square_size >= image.shape[0] and square_size >= image.shape[1]
    shape = list(image.shape)
    shape[0] = shape[1] = square_size
    square = np.zeros(shape, dtype=image.dtype)
    h, w = image.shape[:2]
    offset_h = (square_size - h) // 2
    offset_w = (square_size - w) // 2
    square[offset_h : offset_h + h, offset_w : offset_w + w, ...] = image
    return square


# Load camera matrices and names of corresponding src images from
# colmap images.bin and cameras.bin files from colmap sparse reconstruction
def load_cameras_colmap(images_fp, cameras_fp):

    if images_fp.endswith(".bin"):
        images = read_model.read_images_binary(images_fp)
    else:  # .txt
        images = read_model.read_images_text(images_fp)

    if cameras_fp.endswith(".bin"):
        cameras = read_model.read_cameras_binary(cameras_fp)
    else:  # .txt
        cameras = read_model.read_cameras_text(cameras_fp)

    src_img_nms = []
    K = []
    T = []
    R = []
    w = []
    h = []

    for i in images.keys():
        R.append(read_model.qvec2rotmat(images[i].qvec))
        T.append((images[i].tvec)[..., None])
        k = np.eye(3)
        camera = cameras[images[i].camera_id]
        if camera.model in ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]:
            k[0, 0] = cameras[images[i].camera_id].params[0]
            k[1, 1] = cameras[images[i].camera_id].params[0]
            k[0, 2] = cameras[images[i].camera_id].params[1]
            k[1, 2] = cameras[images[i].camera_id].params[2]
        elif camera.model in ["RADIAL", "PINHOLE"]:
            k[0, 0] = cameras[images[i].camera_id].params[0]
            k[1, 1] = cameras[images[i].camera_id].params[1]
            k[0, 2] = cameras[images[i].camera_id].params[2]
            k[1, 2] = cameras[images[i].camera_id].params[3]
        # TODO : Take other camera models into account + factorize
        else:
            raise NotImplementedError("Camera models not supported yet!")

        K.append(k)
        w.append(cameras[images[i].camera_id].width)
        h.append(cameras[images[i].camera_id].height)
        src_img_nms.append(images[i].name)

    return K, R, T, h, w, src_img_nms


def get_colmap_file(colmap_path, file_stem):
    colmap_path = Path(colmap_path)
    fp = colmap_path / f"{file_stem}.bin"
    if not fp.exists():
        fp = colmap_path / f"{file_stem}.txt"
    return str(fp)


def o3d_to_pyrenderer(mesh_or_pt):
    if isinstance(mesh_or_pt, o3d.geometry.PointCloud):
        points = np.asarray(mesh_or_pt.points).copy()
        colors = np.asarray(mesh_or_pt.colors).copy()
        mesh = pyrender.Mesh.from_points(points, colors)
    elif isinstance(mesh_or_pt, o3d.geometry.TriangleMesh):
        mesh = trimesh.Trimesh(
            np.asarray(mesh_or_pt.vertices),
            np.asarray(mesh_or_pt.triangles),
            vertex_colors=np.asarray(mesh_or_pt.vertex_colors),
        )
        mesh = pyrender.Mesh.from_trimesh(mesh)
    else:
        raise NotImplementedError()
    return mesh


def load_ply(ply_path, voxel_size):
    # Loading the mesh / pointcloud
    m = trimesh.load(ply_path)
    if isinstance(m, trimesh.PointCloud):
        if voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(m.vertices))
            pcd.colors = o3d.utility.Vector3dVector(
                np.asarray(m.colors, dtype=np.float64)[:, :3] / 255
            )
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            mesh = o3d_to_pyrenderer(pcd)
        else:
            points = m.vertices.copy()
            colors = m.colors.copy()
            mesh = pyrender.Mesh.from_points(points, colors)
    elif isinstance(m, trimesh.Trimesh):
        if voxel_size is not None:
            m2 = m.as_open3d
            m2.vertex_colors = o3d.utility.Vector3dVector(
                np.asarray(m.visual.vertex_colors, dtype=np.float64)[:, :3] / 255
            )
            m2 = m2.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average,
            )
            mesh = o3d_to_pyrenderer(m2)
        else:
            mesh = pyrender.Mesh.from_trimesh(m)
    else:
        raise NotImplementedError(
            "Unsupported 3D object. Supported format is a `.ply` pointcloud or mesh."
        )
    return mesh


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_colmap", required=True, type=str, help="path to colmap .bin output files"
    )
    parser.add_argument(
        "--ply_path",
        required=True,
        type=str,
        help="path to point cloud or mesh .ply file",
    )
    parser.add_argument(
        "--src_output", required=True, type=str, help="path to write output images"
    )
    parser.add_argument(
        "--point_size", type=float, default=2.0, help="Rendering point size"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=512,
        help="Minimum height and width (for cropping) images",
    )
    parser.add_argument(
        "--voxel_size",
        type=none_or_float,
        default=None,
        help="Voxel size for voxel-based subsampling used on mesh or pointcloud (None means skipping subsampling)",
    )
    parser.add_argument(
        "--bg_color",
        type=str,
        default=None,
        help="Background comma separated color for rendering.",
    )
    parser.add_argument(
        "--squarify",
        type=lambda x: bool(distutils.util.strtobool(x)),
        help="Should all images that fit be placed onto black canvas of size min_size x min_size?",
    )
    parser.add_argument(
        "--input_poses",
        required=True,
        type=Path,
        help="Path to densePE_top100_shortlist matfile from InLoc run.",
    )
    parser.add_argument(
        "--just_jsons",
        action="store_true",
        default=False,
        help="Whether to render or just prepare info for other renderers.",
    )
    args = parser.parse_args()

    return args


def none_or_float(value):
    """Simplifies unification in SLURM script's parameter handling."""
    if value == "None":
        return None
    return float(value)


if __name__ == "__main__":
    args = parse_args()

    rnd = random.Random(42)
    # Create output folders
    os.makedirs(args.src_output, exist_ok=True)
    # Loading camera pose estimates
    K, R, T, H, W, src_img_nms = load_cameras_colmap(
        get_colmap_file(args.src_colmap, "images"),
        get_colmap_file(args.src_colmap, "cameras"),
    )
    mat = sio.loadmat(args.input_poses)

    if not args.just_jsons:
        flags = RenderFlags.FLAT | RenderFlags.RGBA
        # Loading the mesh / pointcloud
        mesh = load_ply(args.ply_path, args.voxel_size)
        times = [0.0] * len(H)

    indices = list(range(len(H)))
    rnd.shuffle(indices)
    test_size = len(mat["ImgList"][0])
    matrices_dict = {"train": {}, "val": {}}

    for index in range(test_size):
        print("Processing image {}/{}".format(index + 1, test_size), flush=True)
        i = indices[index]

        # Camera intrisics and intrisics
        k, r, t, w, h, img_nm = K[i], R[i], T[i], W[i], H[i], src_img_nms[i]
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = k

        query_path = Path(str(mat["ImgList"][0][index][0][0]))

        if not args.just_jsons:
            # Render query for reference.
            scene = pyrender.Scene(
                bg_color=[float(i) for i in args.bg_color.split(",")]
            )
            scene.add(mesh)
            camera = pyrender.camera.IntrinsicsCamera(
                k[0, 0], k[1, 1], k[0, 2], k[1, 2]
            )
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = r.T
            camera_pose[:3, -1:] = -r.T @ t
            camera_pose[:, 1:3] *= -1
            cam_node = scene.add(camera, pose=camera_pose)
            r = pyrender.OffscreenRenderer(w, h, point_size=args.point_size)
            rgb_rendering, depth_rendering = r.render(scene, flags=flags)
            scene.remove_node(cam_node)

            if args.squarify:
                rgb_rendering = squarify_image(rgb_rendering, args.min_size)
            img_rendering = Image.fromarray(rgb_rendering)
            img_rendering.save(str(Path(args.src_output) / query_path.name))

        candidate_paths = [Path(str(i[0])) for i in mat["ImgList"][0][index][1][0]]
        # candidate_scores = mat['ImgList'][0][index][2][0]
        candidate_projections = [i for i in mat["ImgList"][0][index][3][0]]
        # candidate_k = mat['ImgList'][0][index][4][0]
        candidate_r = mat["ImgList"][0][index][5][0]
        candidate_t = mat["ImgList"][0][index][6][0]

        if all([np.isnan(candidate_projections[tidx]).any() for tidx in range(len(candidate_projections))]):
            continue

        os.makedirs(Path(args.src_output) / query_path.stem, exist_ok=True)

        for tidx in range(len(candidate_projections)):
            # camera_pose = np.eye(4)
            # camera_pose[:3, :3] = candidate_r[tidx]  # r.T
            # camera_pose[:3, -1:] = candidate_t[tidx] #  -r.T @ t
            # camera_pose[:, 1:3] *= -1
            # camera_pose[:3, :3] = camera_pose[:3, :3] @ np.array([[0,1,0], [-1,0,0],[0,0,1]])
            camera_pose = np.concatenate(
                [candidate_projections[tidx], np.array([[0, 0, 0, 1]])], axis=0
            )
            camera_pose[:3, :3] = camera_pose[:3, :3].T
            camera_pose[:3, -1:] = -camera_pose[:3, :3] @ camera_pose[:3, -1:]
            camera_pose[:3, :3] = camera_pose[:3, :3] @ np.array(
                [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float
            )
            if np.isnan(camera_pose).any():
                continue

            # rendered image
            rendered_image_prefix = str(
                Path(args.src_output) / query_path.stem / candidate_paths[tidx].stem
            )
            rendered_image_params = {
                "calibration_mat": [list(l) for l in list(camera_matrix)],
                "camera_pose": [list(l) for l in list(camera_pose)],
                "source_reference": str(query_path),
                "retrieved_database_reference": str(candidate_paths[tidx]),
                "source_scan": query_path.parent.parent.name,
                "source_scan_ply_path": args.ply_path,
                "localized_scan": query_path.parent.parent.name,  # There's only one scan
            }

            matrices_dict["train"][
                f"{rendered_image_prefix}_color.png"
            ] = rendered_image_params
            with open(f"{rendered_image_prefix}_params.json", "w") as ff:
                json.dump(rendered_image_params, ff, indent=4)

            if not args.just_jsons:
                start = time.process_time()

                camera = pyrender.camera.IntrinsicsCamera(
                    k[0, 0], k[1, 1], k[0, 2], k[1, 2]
                )
                try:
                    cam_node = scene.add(camera, pose=camera_pose)
                except np.linalg.LinAlgError:
                    print("Eigenvalues did not converge.", flush=True)
                    continue

                # Offscreen rendering
                rgb_rendering, depth_rendering = r.render(scene, flags=flags)
                scene.remove_node(cam_node)

                end = time.process_time()
                times[i] = end - start

                # cv2.imwrite saves depth map as a single channel img and as-is meaning
                # if max depth is x, then max of the saved img values will be x as well.
                # skimage.io.imsave saves the image normalized, so max value will always
                # be 255 or 65k depending on the data type. plt.imsave saves RGBA image
                # with depth remapped to a color depending on a colormap used.
                # For possible further recalculation, npz with raw depth map is also saved.
                np.save(f"{rendered_image_prefix}_depth.npy", depth_rendering)
                if args.squarify:
                    depth_rendering = squarify_image(depth_rendering, args.min_size)
                cv2.imwrite(
                    f"{rendered_image_prefix}_depth.png",
                    depth_rendering.astype(np.uint16),
                )

                # rendered image
                if args.squarify:
                    rgb_rendering = squarify_image(rgb_rendering, args.min_size)
                img_rendering = Image.fromarray(rgb_rendering)
                img_rendering.putalpha(255)
                img_rendering.save(
                    str(
                        Path(args.src_output)
                        / query_path.stem
                        / f"{candidate_paths[tidx].stem}_color.png"
                    )
                )
        if not args.just_jsons:
            r.delete()

    if not args.just_jsons:
        times = np.array(times)
        print(f"Rendering time per image was ({np.mean(times)} +- {np.std(times)}) s.", flush=True)

    with open(Path(args.src_output) / "matrices_for_rendering.json", "w") as ff:
        json.dump(matrices_dict, ff, indent=4)
