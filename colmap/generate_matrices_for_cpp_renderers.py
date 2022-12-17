"""
python colmap/generate_matrices_for_sphere_pcd_renderer.py --src_colmap /home/kremeto1/neural_rendering/datasets/processed/imc/pantheon_exterior/dense/dense/sparse --src_output /home/kremeto1/neural_rendering/datasets/post_processed/imc/pantheon_exterior_minsz-512_valr-0.2_pts-2.0_down-25_src-fused --src_reference /home/kremeto1/neural_rendering/datasets/processed/imc/pantheon_exterior/dense/dense/images --val_ratio 0.2 --bg_color "0.0,0.0,0.0" --ply_path unused --output_json_path ~/test_matrices_25_pantheon.json --verbose
python colmap/generate_matrices_for_sphere_pcd_renderer.py --src_colmap /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/sparse --src_output /nfs/projects/artwin/experiments/hololens_mapper/2019-09-28_08.31.29-rendered-mesh-black_bg --verbose --src_reference /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/images --val_ratio 0.0 --bg_color "0.0,0.0,0.0" --ply_path /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/2019-09-28_08.31.29.ply --output_json_path ~/test_matrices_29.json --verbose
python colmap/generate_matrices_for_sphere_pcd_renderer.py --src_colmap /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53/sparse --src_output /nfs/projects/artwin/experiments/hololens_mapper/2019-09-28_16.11.53-rendered-mesh-black_bg --verbose --src_reference /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53/images --val_ratio 0.0 --bg_color "0.0,0.0,0.0" --ply_path /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53/2019-09-28_16.11.53.ply --output_json_path ~/test_matrices_53.json --verbose
"""

import argparse
import cv2
import distutils.util
import json
import matplotlib.pyplot as plt
import os
import random
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

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


def build_dataset(
    src_reference,
    src_colmap,
    out_dir,
    min_size=512,
    val_ratio=0.2,
    verbose=False,
    squarify=False,
    test_size=0,
):
    """Build the input dataset composed of the reference images, the RGBA and depth renderings.
    Args:
        - src_reference : reference images directory
        - src_colmap : colmap SfM output directory
        - out_dir : output directory
        - min_size : minimum height and width (for cropping)
        - val_ratio : train / val ratio
    """
    rnd = random.Random(42)
    # Create output folders
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val"), exist_ok=True)
    # Loading camera pose estimates
    K, R, T, H, W, src_img_nms = load_cameras_colmap(
        get_colmap_file(src_colmap, "images"), get_colmap_file(src_colmap, "cameras")
    )
    it = 0

    indices = list(range(len(H)))
    rnd.shuffle(indices)
    split = int(val_ratio * (len(H) - test_size))

    train = {}
    val = {}
    test = {}

    for index in range(len(H)):
        if verbose:
            print("Processing image {}/{}".format(index + 1, len(H)))
        i = indices[index]

        # Camera intrisics and intrisics
        k, r, t, w, h, img_nm = K[i], R[i], T[i], W[i], H[i], src_img_nms[i]

        if squarify or min(w, h) > min_size:
            try:
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = r.T
                camera_pose[:3, -1:] = -r.T @ t
                camera_pose[:, 1:3] *= -1

                camera_matrix = np.eye(4)
                camera_matrix[:3, :3] = k

                # Building dataset
                if it < test_size:
                    output_dir = os.path.join(out_dir, "test")
                    output_dict = test
                elif it < test_size + split:
                    output_dir = os.path.join(out_dir, "val")
                    output_dict = val
                else:
                    output_dir = os.path.join(out_dir, "train")
                    output_dict = train

                # Reference image
                img = plt.imread(os.path.join(src_reference, img_nm))
                if squarify:
                    img = squarify_image(img, min_size)
                img = Image.fromarray(img)
                img.save(os.path.join(output_dir, "{:04n}_reference.png".format(it)))

                depth_p = Path(output_dir) / "{:04n}_depth.png".format(it)
                if depth_p.exists():
                    depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
                    assert len(depth.shape) == 2 or depth.shape[2] == 1, "Weird depth"
                    depth = squarify_image(depth, min_size)
                    cv2.imwrite(str(depth_p), depth)

                color_p = Path(output_dir) / "{:04n}_color.png".format(it)
                if color_p.exists():
                    color = cv2.imread(str(color_p), cv2.IMREAD_UNCHANGED)
                    color = squarify_image(color, min_size)
                    if color.shape[2] == 4:
                        color[:, :, 3] = np.iinfo(color.dtype).max
                    cv2.imwrite(str(color_p), color)

                # rendered image
                target_path = os.path.join(output_dir, "{:04n}_color.png".format(it))
                output_dict[target_path] = {
                    "intrinsic_matrix": [list(l) for l in list(camera_matrix)],
                    "extrinsic_matrix": [list(l) for l in list(camera_pose)],
                }
            except AssertionError:
                print(os.path.join(src_reference, img_nm))
                print(traceback.format_exc())

            it += 1

        else:
            if verbose:
                print("Skipping this image : too small ")

    with open(Path(out_dir) / "test" / "matrices_for_rendering.json", "w") as f:
        f.write(json.dumps({"train": test, "val": {}}, indent=4))

    with open(Path(out_dir) / "val" / "matrices_for_rendering.json", "w") as f:
        f.write(json.dumps({"train": val, "val": {}}, indent=4))

    with open(Path(out_dir) / "train" / "matrices_for_rendering.json", "w") as f:
        f.write(json.dumps({"train": train, "val": {}}, indent=4))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_reference", required=True, type=str, help="path to reference images"
    )
    parser.add_argument(
        "--src_colmap", required=True, type=str, help="path to colmap .bin output files"
    )
    parser.add_argument(
        "--src_output", required=True, type=str, help="path to write output images"
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Train/val ratio")
    parser.add_argument(
        "--min_size", type=int, default=512, help="Minimum size for images"
    )
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity")
    parser.add_argument(
        "--squarify",
        type=lambda x: bool(distutils.util.strtobool(x)),
        help="Should all images that fit be placed onto black canvas of size min_size x min_size?",
    )
    parser.add_argument(
        "--test_size", type=int, default=0, help="Test size for generated dataset"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # Build dataset
    build_dataset(
        src_reference=args.src_reference,
        src_colmap=args.src_colmap,
        out_dir=args.src_output,
        min_size=args.min_size,
        val_ratio=args.val_ratio,
        verbose=args.verbose,
        squarify=args.squarify,
        test_size=args.test_size,
    )
