"""
python colmap/generate_matrices_for_sphere_pcd_renderer.py --src_colmap /home/kremeto1/neural_rendering/datasets/processed/imc/pantheon_exterior/dense/dense/sparse --src_output /home/kremeto1/neural_rendering/datasets/post_processed/imc/pantheon_exterior_minsz-512_valr-0.2_pts-2.0_down-25_src-fused --src_reference /home/kremeto1/neural_rendering/datasets/processed/imc/pantheon_exterior/dense/dense/images --val_ratio 0.2 --bg_color "0.0,0.0,0.0" --ply_path unused --output_json_path ~/test_matrices_25_pantheon.json --verbose
python colmap/generate_matrices_for_sphere_pcd_renderer.py --src_colmap /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/sparse --src_output /nfs/projects/artwin/experiments/hololens_mapper/2019-09-28_08.31.29-rendered-mesh-black_bg --verbose --src_reference /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/images --val_ratio 0.0 --bg_color "0.0,0.0,0.0" --ply_path /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/2019-09-28_08.31.29.ply --output_json_path ~/test_matrices_29.json --verbose
python colmap/generate_matrices_for_sphere_pcd_renderer.py --src_colmap /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53/sparse --src_output /nfs/projects/artwin/experiments/hololens_mapper/2019-09-28_16.11.53-rendered-mesh-black_bg --verbose --src_reference /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53/images --val_ratio 0.0 --bg_color "0.0,0.0,0.0" --ply_path /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53/2019-09-28_16.11.53.ply --output_json_path ~/test_matrices_53.json --verbose
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import read_model

################################################################################
# Load sfm model directly from colmap output files
################################################################################

# Load point cloud with per-point sift descriptors and rgb features from
# colmap database and points3D.bin file from colmap sparse reconstruction
# def load_points_colmap(points3D_fp):

#     if points3D_fp.endswith(".bin"):
#         points3D = read_model.read_points3d_binary(points3D_fp)
#     else:  # .txt
#         points3D = read_model.read_points3D_text(points3D_fp)

#     pcl_xyz = []
#     pcl_rgb = []
#     for pt3D in points3D.values():
#         pcl_xyz.append(pt3D.xyz)
#         pcl_rgb.append(pt3D.rgb)

#     pcl_xyz = np.vstack(pcl_xyz).astype(np.float32)
#     pcl_rgb = np.vstack(pcl_rgb).astype(np.uint8)

#     return pcl_xyz, pcl_rgb


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
    output_json_path=None,
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
    # Loading camera pose estimates
    K, R, T, H, W, src_img_nms = load_cameras_colmap(
        get_colmap_file(src_colmap, "images"), get_colmap_file(src_colmap, "cameras")
    )
    train = {}
    val = {}

    it = 0
    for i in range(len(H)):
        if verbose:
            print("Processing image {}/{}".format(i + 1, len(H)))

        # Camera intrisics and intrisics
        k, r, t, w, h, img_nm = K[i], R[i], T[i], W[i], H[i], src_img_nms[i]

        if min(w, h) > min_size:
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = r.T
            camera_pose[:3, -1:] = -r.T @ t
            # camera_pose[:3, :3] = r
            # camera_pose[:3, -1:] = t
            camera_pose[:, 1:3] *= -1

            camera_matrix = np.eye(4)
            camera_matrix[:3, :3] = k

            # Building dataset
            if i < (1.0 - val_ratio) * len(H):
                output_dir = os.path.join(out_dir, "train")
                output_dict = train
            else:
                output_dir = os.path.join(out_dir, "val")
                output_dict = val

            # rendered image
            target_path = os.path.join(output_dir, "{:04n}_color.png".format(it))
            output_dict[target_path] = {
                "intrinsic_matrix": [list(l) for l in list(camera_matrix)],
                "extrinsic_matrix": [list(l) for l in list(camera_pose)],
            }

            it += 1

        else:
            if verbose:
                print("Skipping this image : too small ")

    output = {"train": train, "val": val}
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=4)


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
    parser.add_argument(
        "--output_json_path", required=True, type=str, help="where to write matrix file"
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
        bg_color=[float(i) for i in args.bg_color.split(",")],
        output_json_path=args.output_json_path,
    )
