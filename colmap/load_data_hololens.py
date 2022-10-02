import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
# When ran with SLURM on a multigpu node, scheduled on other than GPU0, we need
# to set this or we get an egl initialization error.
os.environ["EGL_DEVICE_ID"] = os.environ.get("SLURM_JOB_GPUS", "0").split(",")[0]
sys.path.insert(0, "/home/kremeto1/hololens_mapper/")
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "1000"

import trimesh
import argparse
import pyrender
import numpy as np
import open3d as o3d
import cv2
import time
from pathlib import Path
from PIL import Image
from pyrender.constants import RenderFlags
import read_model
import traceback

from src.utils.UtilsMath import UtilsMath

################################################################################
# Load sfm model directly from colmap output files
################################################################################


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
    voxel_size=100,
    bg_color=None,
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
        - voxel_size : voxel size for voxel-based subsampling used on mesh or pointcloud
                       (None means skipping subsampling)
    """
    # Create output folders
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val"), exist_ok=True)
    # Loading camera pose estimates
    K, R, T, H, W, src_img_nms = load_cameras_colmap(
        get_colmap_file(src_colmap, "images"), get_colmap_file(src_colmap, "cameras")
    )
    flags = RenderFlags.FLAT | RenderFlags.RGBA | RenderFlags.DEPTH_ONLY
    # Loading the mesh / pointcloud
    # mesh = load_ply(ply_path, voxel_size)
    pcd = o3d.io.read_point_cloud(ply_path)
    rgb = np.asarray(pcd.colors).T
    rgb *= 255
    rgb = rgb.astype(np.uint8)
    xyz = np.asarray(pcd.points).T
    it = 0

    import pickle

    # tree = KDTree(xyz.T)
    # nn = tree.query(xyz.T, k=2)
    with open(out_dir + "/radii.pickle", "rb") as f:
        nn = pickle.load(f)
    radii = nn[0][:, 1::2].squeeze().astype(np.uint16)

    times = [0.0] * len(H)
    mapping = []
    utils_math = UtilsMath()

    # For artwin all images have the same dimensions, pre-creating EGL context
    # within renderer speeds up the rendering loop.
    scene = pyrender.Scene(bg_color=bg_color)
    scene.add(o3d_to_pyrenderer(pcd))
    rndr = pyrender.OffscreenRenderer(W[0], H[0], point_size=point_size)

    for i in range(len(H)):
        if verbose:
            print("Processing image {}/{}".format(i + 1, len(H)))

        # Camera intrisics and intrisics
        k, r, t, w, h, img_nm = K[i], R[i], T[i], W[i], H[i], src_img_nms[i]

        if min(w, h) > min_size and it == 22:
            try:
                start = time.process_time()

                camera = pyrender.IntrinsicsCamera(k[0, 0], k[1, 1], k[0, 2], k[1, 2])
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = r.T
                camera_pose[:3, -1:] = -r.T @ t
                camera_pose[:, 1:3] *= -1
                camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
                scene.add_node(camera_node)

                # Offscreen rendering
                rgb_rendering = utils_math.render_colmap_image(
                    {
                        "K": k,
                        "R": r,
                        "C": -r.T @ t,
                        "h": h,
                        "w": w,
                        "rgb": rgb,
                        "xyz": xyz,
                        "radii": radii,
                    },
                    max_radius=18,
                )

                depth_rendering = rndr.render(scene, flags=flags)
                scene.remove_node(camera_node)

                end = time.process_time()
                times[i] = end - start

                # depth_rendering = normalize_depth(depth_rendering)

                # Building dataset
                if i < (1.0 - val_ratio) * len(H):
                    output_dir = os.path.join(out_dir, "train")
                    mapping.append(f"{img_nm} -> {it:08n} TRAIN\n")
                else:
                    output_dir = os.path.join(out_dir, "val")
                    mapping.append(f"{img_nm} -> {it:08n} VALID\n")

                # Reference image
                img = Image.open(os.path.join(src_reference, img_nm))
                print(
                    f'Saving {os.path.join(output_dir, "{:04n}_reference.png".format(it))}'
                )
                img.save(os.path.join(output_dir, "{:04n}_reference.png".format(it)))

                # depth image
                if src_depth is not None:
                    raise NotImplementedError(
                        "Use renderings rather than depth maps estimated with COLMAP."
                    )
                # cv2.imwrite saves depth map as a single channel img and as-is meaning
                # if max depth is x, then max of the saved img values will be x as well.
                # skimage.io.imsave saves the image normalized, so max value will always
                # be 255 or 65k depending on the data type. plt.imsave saves RGBA image
                # with depth remapped to a color depending on a colormap used.
                print(
                    f'Saving {os.path.join(output_dir, "{:04n}_depth.png".format(it))}'
                )
                cv2.imwrite(
                    os.path.join(output_dir, "{:04n}_depth.png".format(it)),
                    depth_rendering.astype(np.uint16),
                )
                print(
                    f'Saving {os.path.join(output_dir, "{:04n}_depth.npy".format(it))}'
                )
                # For possible further recalculation, npz with raw depth map is also saved.
                np.save(
                    os.path.join(output_dir, "{:04n}_depth.npy".format(it)),
                    depth_rendering,
                )

                # rendered image
                img_rendering = Image.fromarray(rgb_rendering)
                print(
                    f'Saving {os.path.join(output_dir, "{:04n}_color.png".format(it))}'
                )
                img_rendering.save(
                    os.path.join(output_dir, "{:04n}_color.png".format(it))
                )
            except:
                traceback.print_exc()

            it += 1

        else:
            if verbose:
                print("Skipping this image : too small ")

        sys.stdout.flush()

    if verbose:
        times = np.array(times)
        print(f"Rendering time per image was ({np.mean(times)} +- {np.std(times)}) s.")

    with open(Path(out_dir) / "mapping.txt", "w") as mapping_file:
        mapping_file.writelines(mapping)


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
    parser.add_argument(
        "--voxel_size",
        type=none_or_float,
        default=None,
        help="Voxel size used for downsampling mesh or pointcloud.",
    )
    parser.add_argument(
        "--bg_color",
        type=str,
        default=None,
        help="Background comma separated color for rendering.",
    )
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
        src_reference=args.src_reference,
        ply_path=args.ply_path,
        src_colmap=args.src_colmap,
        out_dir=args.src_output,
        min_size=args.min_size,
        val_ratio=args.val_ratio,
        src_depth=None,
        point_size=args.point_size,
        verbose=args.verbose,
        voxel_size=args.voxel_size,
        bg_color=[0.0, 0.0, 0.0],  # [float(i) for i in args.bg_color.split(",")],
    )
