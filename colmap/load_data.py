import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import trimesh
import argparse
import pyrender
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.transform import resize
from skimage.io import imread
from io import BytesIO
from pyrender.constants import RenderFlags
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


def build_dataset(
    src_reference,
    ply_path,
    src_colmap,
    out_dir,
    min_size=512,
    val_ratio=0.2,
    src_depth=None,
    point_size=3.0,
    verbose=False,
):
    """Build the input dataset composed of the reference images, the RGBA and depth renderings.
    Args:
        - src_reference : reference images directory
        - ply_path : 3D scene mesh or pointcloud path
        - src_colmap : colmap SfM output directory
        - out_dir : output directory
        - min_size : minimum height and width (for cropping)
        - val_ratio : train / val ratio
        - src_depth : depth maps directory. If None, the depth maps are rendered along with RGB renderings.
        - point_size : Point size for rendering
    """
    # Create output folders
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val"), exist_ok=True)
    # Loading camera pose estimates
    K, R, T, H, W, src_img_nms = load_cameras_colmap(
        get_colmap_file(src_colmap, "images"), get_colmap_file(src_colmap, "cameras")
    )
    flags = RenderFlags.FLAT | RenderFlags.RGBA
    # Loading the mesh / pointcloud
    m = trimesh.load(ply_path)
    if isinstance(m, trimesh.PointCloud):
        points = m.vertices.copy()
        colors = m.colors.copy()
        mesh = pyrender.Mesh.from_points(points, colors)
    elif isinstance(m, trimesh.Trimesh):
        mesh = pyrender.Mesh.from_trimesh(m)
    else:
        raise NotImplementedError(
            "Unsupported 3D object. Supported format is a `.ply` pointcloud or mesh."
        )
    it = 0

    for i in range(len(H)):
        if verbose:
            print("Processing image {}/{}".format(i + 1, len(H)))

        # Camera intrisics and intrisics
        k, r, t, w, h, img_nm = K[i], R[i], T[i], W[i], H[i], src_img_nms[i]

        if min(w, h) > min_size:
            scene = pyrender.Scene()
            scene.add(mesh)
            camera = pyrender.camera.IntrinsicsCamera(
                k[0, 0], k[1, 1], k[0, 2], k[1, 2]
            )
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = r.T
            camera_pose[:3, -1:] = -r.T @ t
            camera_pose[:, 1:3] *= -1
            scene.add(camera, pose=camera_pose)

            # Offscreen rendering
            r = pyrender.OffscreenRenderer(w, h, point_size=point_size)
            rgb_rendering, depth_rendering = r.render(scene, flags=flags)
            img_rendering = Image.fromarray(rgb_rendering)
            # depth_rendering = normalize_depth(depth_rendering)

            # Building dataset
            if i < (1.0 - val_ratio) * len(H):
                output_dir = os.path.join(out_dir, "train")
            else:
                output_dir = os.path.join(out_dir, "val")

            # Reference image
            img = Image.open(os.path.join(src_reference, img_nm))
            img.save(os.path.join(output_dir, "{:04n}_reference.png".format(it)))

            # depth image
            if src_depth is not None:
                raise NotImplementedError(
                    "Use renderings rather than depth maps estimated with COLMAP."
                )
            plt.imsave(
                os.path.join(output_dir, "{:04n}_depth.png".format(it)),
                depth_rendering,
                cmap="gray",
            )

            # rendered image
            img_rendering = Image.fromarray(rgb_rendering)
            img_rendering.save(os.path.join(output_dir, "{:04n}_color.png".format(it)))

            it += 1

        else:
            if verbose:
                print("Skipping this image : too small ")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_reference", required=True, type=str, help="path to reference images"
    )
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
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Train/val ratio")
    parser.add_argument(
        "--point_size", type=float, default=2.0, help="Rendering point size"
    )
    parser.add_argument(
        "--min_size", type=int, default=512, help="Minimum size for images"
    )
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # Build dataset
    build_dataset(
        src_reference=args.src_reference,
        ply_path=args.ply_path,
        src_colmap=args.src_colmap,
        out_dir=args.src_output,
        min_size=args.min_size,
        val_ratio=args.val_ratio,
        src_depth=None,
        point_size=args.point_size,
        verbose=args.verbose,
    )
