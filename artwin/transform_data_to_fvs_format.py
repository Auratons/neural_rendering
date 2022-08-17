import os
from typing import Tuple

os.environ["PYOPENGL_PLATFORM"] = "egl"
# When ran with SLURM on a multigpu node, scheduled on other than GPU0, we need
# to set this or we get an egl initialization error.
os.environ["EGL_DEVICE_ID"] = os.environ.get("SLURM_JOB_GPUS", "0").split(",")[0]

import trimesh
import argparse
import pyrender
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
import sys
import struct
import time
from pathlib import Path
from PIL import Image
from skimage.io import imread
from io import BytesIO
from pyrender.constants import RenderFlags

sys.path.append(f"{str(Path(__file__).parent)}/../colmap")
import read_model

################################################################################
# Load sfm model directly from colmap output files
################################################################################

# Load point cloud with per-point sift descriptors and rgb features from
# colmap database and points3D.bin file from colmap sparse reconstruction
def load_points_colmap(points3D_fp):

    if points3D_fp.endswith(".bin"):
        points3D = read_model.read_points3d_binary(points3D_fp)
    else:  # .txt
        points3D = read_model.read_points3D_text(points3D_fp)

    pcl_xyz = []
    pcl_rgb = []
    for pt3D in points3D.values():
        pcl_xyz.append(pt3D.xyz)
        pcl_rgb.append(pt3D.rgb)

    pcl_xyz = np.vstack(pcl_xyz).astype(np.float32)
    pcl_rgb = np.vstack(pcl_rgb).astype(np.uint8)

    return pcl_xyz, pcl_rgb


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


def render_from_camera(
    mesh, k, r, t, w, h, src_img_nm, out_dir, point_size=3.0, render_depth=True
):
    """Render the point cloud from a given camera pose using trimesh, and save
    it in an output directory.
    Args:
        - mesh : trimesh.points.PointCloud (or.Mesh) object
        - k : camera intrinsic matrix
        - r, t : extrinsic matrix
        - w, h : camera resolution
        - src_img_nm : image name
        - out_dir : save directory
    """
    scene = pyrender.Scene()
    scene.add(mesh)

    # Camera intrisics and intrisics
    camera = pyrender.camera.IntrinsicsCamera(k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = r.T
    camera_pose[:3, -1:] = -r.T @ t
    camera_pose[:, 1:3] *= -1
    scene.add(camera, pose=camera_pose)

    # Off-screen rendering
    flags = RenderFlags.FLAT | RenderFlags.RGBA
    r = pyrender.OffscreenRenderer(w, h, point_size=point_size)
    rgb_rendering, depth_rendering = r.render(  # pylint: disable=unused-variable
        scene, flags=flags
    )
    img_rendering = Image.fromarray(rgb_rendering)
    img_rendering.save(os.path.join(out_dir, "{}_rendered.png".format(src_img_nm[:-4])))


def get_colmap_file(colmap_path, file_stem):
    colmap_path = Path(colmap_path)
    fp = colmap_path / f"{file_stem}.bin"
    if not fp.exists():
        fp = colmap_path / f"{file_stem}.txt"
    return str(fp)


def make_colmap_renderings(colmap_pt, ply_pt, out_dir, render_depth=True):
    """Compute rendered images from a colmap dense reconstruction.
    Args:
        - colmap_pt : path to colmap cameras.bin, images.bin, points3D.bin (from SfM reconstruction)
        - ply_pt : path to .ply dense MVS reconstruction
    """
    os.makedirs(out_dir, exist_ok=True)
    K, R, T, H, W, src_img_nms = load_cameras_colmap(
        get_colmap_file(colmap_pt, "images"), get_colmap_file(colmap_pt, "cameras")
    )
    m = trimesh.load(ply_pt)
    if isinstance(m, trimesh.PointCloud):
        points = m.vertices.copy()
        colors = m.colors.copy()
        mesh = pyrender.Mesh.from_points(points, colors)
    elif isinstance(m, trimesh.Trimesh):
        mesh = pyrender.Mesh.from_trimesh(m)
    else:
        raise NotImplementedError()
    for i in range(len(H)):
        render_from_camera(mesh, K[i], R[i], T[i], W[i], H[i], src_img_nms[i], out_dir)


def load_depth_map(file_path, dtype=np.float16):
    with open(file_path, "rb") as f:
        fbytes = f.read()
    w, h, c = [int(x) for x in str(fbytes[:20])[2:].split("&")[:3]]
    header = "{}&{}&{}&".format(w, h, c)
    body = fbytes[len(header) :]
    img = np.fromstring(body, dtype=dtype).reshape((h, w, c))
    img = np.nan_to_num(img)

    # Clip the resulting map between 0 and 10
    img = np.clip(img, -2, 10)
    img[img < -1] = 10
    return img.astype(np.float32)[:, :, 0]


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


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def resize(im, desired_size):
    side_ratio = lambda tupl: (tupl[0] / tupl[1]) - 1
    old_size = im.shape[:2]  # old_size is in (height, width) format
    print(side_ratio(desired_size))
    print(side_ratio(old_size))
    if side_ratio(desired_size) * side_ratio(old_size) < 0:
        if side_ratio(desired_size) < side_ratio(old_size):
            ratio = float(desired_size[np.argmax(old_size)]) / max(old_size)
        else:
            ratio = float(desired_size[np.argmin(old_size)]) / min(old_size)
    else:
        if side_ratio(desired_size) < side_ratio(old_size):
            ratio = float(desired_size[np.argmin(old_size)]) / min(old_size)
        else:
            ratio = float(desired_size[np.argmax(old_size)]) / max(old_size)
    if side_ratio(old_size) == 0:
        ratio = float(desired_size[np.argmin(desired_size)]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    print(new_size)

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size[1] - new_size[1]
    delta_h = desired_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def get_central_crop(img, crop_height=512, crop_width=512):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    assert len(img.shape) == 3, (
        "input image should be either a 2D or 3D matrix,"
        " but input was of shape %s" % str(img.shape)
    )
    height, width, _ = img.shape
    assert height >= crop_height and width >= crop_width, (
        "input image cannot " "be smaller than the requested crop size"
    )
    st_y = (height - crop_height) // 2
    st_x = (width - crop_width) // 2
    return np.squeeze(img[st_y : st_y + crop_height, st_x : st_x + crop_width, :])


def build_dataset(
    src,
    out,
    ply_path,
    val_ratio=0.2,
    verbose=False,
):
    """Build the input dataset composed of the reference images, the RGBA and depth renderings.
    Args:
        - src : colmap SfM output directory
        - out : output directory
        - ply_path : 3D scene mesh or pointcloud path
        - val_ratio : train / val ratio
    """
    src_reference = "/nfs/projects/artwin/experiments/matlab_60_fov/train"  # str(Path(src) / "images")
    src_colmap = str(Path(src) / "sparse")
    # Create output folders
    os.makedirs(out, exist_ok=True)
    # Loading camera pose estimates
    K, R, T, H, W, src_img_nms = load_cameras_colmap(
        get_colmap_file(src_colmap, "images"), get_colmap_file(src_colmap, "cameras")
    )

    Ks, Rs, ts, mapping = [], [], [], []

    it = 0

    for i in range(len(H)):
        # Camera intrisics and intrisics
        k, r, t, w, h, img_nm = K[i], R[i], T[i], W[i], H[i], src_img_nms[i]

        if verbose:
            print("Processing image {}/{}\t\t{}".format(i + 1, len(H), img_nm))
        try:
            # Reference image
            if not Path(os.path.join(out, "im_{:08n}.png".format(it))).exists():
                img = cv2.imread(
                    os.path.join(src_reference, img_nm), cv2.IMREAD_UNCHANGED
                )
                img = get_central_crop(img, 546, 980)

            # depth image
            if not Path(os.path.join(out, "dm_{:08n}.npy".format(it))).exists():
                src = Path(
                    "/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53-rendered-black_bg/train"
                )
                depth_map = np.load(src / f"{it:04n}_depth.npy")
                depth_map = get_central_crop(depth_map, 546, 980)

            if not Path(os.path.join(out, "im_{:08n}.png".format(it))).exists():
                cv2.imwrite(os.path.join(out, "im_{:08n}.png".format(it)), img)
            if not Path(os.path.join(out, "dm_{:08n}.npy".format(it))).exists():
                np.save(
                    os.path.join(out, "dm_{:08n}.npy".format(it)),
                    depth_map,
                )

            # Building dataset
            if i < (1.0 - val_ratio) * len(H):
                mapping.append(f"{img_nm} -> {it:08n} TRAIN\n")
            else:
                mapping.append(f"{img_nm} -> {it:08n} VALID\n")

            Ks.append(k)
            Rs.append(r)
            ts.append(t)
            it += 1
        except (AttributeError, FileNotFoundError) as e:
            print(f"Fail for {img_nm}: {e}")

    np.save(str(Path(out) / "Ks.npy"), np.array(Ks))
    np.save(str(Path(out) / "Rs.npy"), np.array(Rs))
    np.save(str(Path(out) / "ts.npy"), np.array(ts).squeeze())
    with open(Path(out) / "mapping.txt", "w") as mapping_file:
        mapping_file.writelines(mapping)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src", required=True, type=str, help="path to colmap .bin output files"
    )
    parser.add_argument(
        "--ply_path",
        required=True,
        type=str,
        help="path to point cloud or mesh .ply file",
    )
    parser.add_argument(
        "--out", required=True, type=str, help="path to write output images"
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Train/val ratio")
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity")
    args = parser.parse_args()

    return args


def none_or_float(value):
    """Simplifies unification in SLURM script's parameter handling."""
    if value == "None":
        return None
    return float(value)


if __name__ == "__main__":
    args = parse_args()

    # Build dataset
    build_dataset(
        src=args.src,
        out=args.out,
        ply_path=args.ply_path,
        val_ratio=args.val_ratio,
        verbose=args.verbose,
    )
