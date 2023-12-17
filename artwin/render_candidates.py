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
        "--src_output", required=True, type=str, help="path to write output images"
    )
    parser.add_argument(
        "--point_size", type=float, default=2.0, help="Rendering point size"
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
        default="0.0,0.0,0.0",
        help="Background comma separated color for rendering.",
    )
    parser.add_argument(
        "--input_mapping",
        required=True,
        type=Path,
        help="Mapping from n-tuple to source img",
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

    # Build the input dataset composed of the reference images, the RGBA and depth renderings.

    rnd = random.Random(42)
    # Create output folders
    os.makedirs(args.src_output, exist_ok=True)

    def reverse_dict(dictionary):
        return {v: k for k, v in dictionary.items()}

    # Get mapping from reference images in joined dataset for nriw training to
    # reference images in a concrete nriw training dataset from which the joined
    # one was generated.

    with open(args.input_mapping, "r") as file:
        sub_mapping_1 = json.load(file)
    sub_mapping_1 = reverse_dict(sub_mapping_1)

    mat = sio.loadmat(args.input_poses)

    ply29 = "/nfs/projects/artwin/experiments/thesis/artwin_as_inloc/2019-09-28_08.31.29/2019-09-28_08.31.29.ply"
    ply53 = "/nfs/projects/artwin/experiments/thesis/artwin_as_inloc/2019-09-28_16.11.53/2019-09-28_16.11.53.ply"

    if not args.just_jsons:
        flags = RenderFlags.FLAT | RenderFlags.RGBA
        # Loading the mesh / pointcloud
        mesh29 = load_ply(ply29, args.voxel_size)
        mesh53 = load_ply(ply53, args.voxel_size)
        times = [0.0] * len(mat["ImgList"][0])

    test_size = len(mat["ImgList"][0])
    matrices_dict = {"train": {}, "val": {}}
    matrices_dict_29 = {"train": {}, "val": {}}
    matrices_dict_53 = {"train": {}, "val": {}}

    for index in range(test_size):
        print("Processing image {}/{}".format(index + 1, test_size))

        joined_dataset_query_path = Path(str(mat["ImgList"][0][index][0][0]))

        source_photo = Path(sub_mapping_1[str(joined_dataset_query_path)])
        source_photo_prefix = str(
            source_photo.parent / source_photo.stem.strip("_reference")
        )

        try:
            matrices_file = (
                source_photo.parent.parent / "train" / "matrices_for_rendering.json"
            )
            with open(matrices_file, "r") as file:
                params = json.load(file)
            p = params["train"][f"{source_photo_prefix}_color.png"]
            # Backward compatibility with wrong naming
            k = np.array(
                p["calibration_mat"]
                if "calibration_mat" in p
                else p["intrinsic_matrix"]
            )
            camera_pose = np.array(
                p["camera_pose"] if "camera_pose" in p else p["extrinsic_matrix"]
            )
        except:
            params_in_path = f"{source_photo_prefix}_params.json"
            with open(params_in_path, "r") as file:
                params = json.load(file)
            k = np.array(params["calibration_mat"])
            camera_pose = np.array(params["camera_pose"])
            camera_pose[:, 1:3] *= -1

        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = k

        if not args.just_jsons:
            # Render query for reference.
            scene = pyrender.Scene(
                bg_color=[float(i) for i in args.bg_color.split(",")]
            )
            if "53" in str(source_photo.parent.parent):
                mesh = mesh53
            else:
                mesh = mesh29
            scene.add(mesh)
            camera = pyrender.camera.IntrinsicsCamera(
                k[0, 0], k[1, 1], k[0, 2], k[1, 2]
            )
            cam_node = scene.add(camera, pose=camera_pose)
            r = pyrender.OffscreenRenderer(
                k[0, 2] * 2, k[1, 2] * 2, point_size=args.point_size
            )
            rgb_rendering, depth_rendering = r.render(scene, flags=flags)
            scene.remove_node(cam_node)
            img_rendering = Image.fromarray(rgb_rendering)
            img_rendering.save(
                str(Path(args.src_output) / joined_dataset_query_path.name)
            )

        candidate_paths = [Path(str(i[0])) for i in mat["ImgList"][0][index][1][0]]
        # candidate_scores = mat['ImgList'][0][index][2][0]
        candidate_projections = mat["ImgList"][0][index][3][0]
        candidate_r = mat["ImgList"][0][index][4][0]
        candidate_t = mat["ImgList"][0][index][5][0]
        # candidate_k = mat['ImgList'][0][index][6][0]

        if all([np.isnan(candidate_projections[tidx]).any() for tidx in range(len(candidate_projections))]):
            continue

        os.makedirs(
            Path(args.src_output) / joined_dataset_query_path.stem, exist_ok=True
        )

        for tidx in range(len(candidate_projections)):
            camera_pose = np.eye(4)
            camera_pose[:3, :] = candidate_projections[tidx]
            camera_pose[:3, :3] = camera_pose[:3, :3].T
            camera_pose[:3, -1] = -camera_pose[:3, :3] @ camera_pose[:3, -1]
            if np.isnan(camera_pose).any():
                continue

            # From simulated "global" CS back to local CS of pcd.
            localized_scan = "29"
            if camera_pose[2, -1] > 25:
                localized_scan = "53"
                camera_pose[2, -1] -= 50
            camera_pose[:, 1:3] *= -1
            # camera_pose[:3, :3] = camera_pose[:3, :3] @ np.array([[0,1,0], [-1,0,0],[0,0,1]])

            # rendered image
            rendered_image_prefix = str(
                Path(args.src_output)
                / joined_dataset_query_path.stem
                / candidate_paths[tidx].stem
            )
            rendered_image_params = {
                "calibration_mat": [list(l) for l in list(camera_matrix)],
                "camera_pose": [list(l) for l in list(camera_pose)],
                "source_reference": str(source_photo),
                "retrieved_database_reference": str(candidate_paths[tidx]),
                "source_scan": "53"
                if "53" in str(source_photo.parent.parent)
                else "29",
                "source_scan_ply_path": ply53
                if "53" in str(source_photo.parent.parent)
                else ply29,
                "localized_scan": localized_scan,
            }

            matrices_dict["train"][
                f"{rendered_image_prefix}_color.png"
            ] = rendered_image_params
            if "53" in str(source_photo.parent.parent):
                matrices_dict_53["train"][
                    f"{rendered_image_prefix}_color.png"
                ] = rendered_image_params
            else:
                matrices_dict_29["train"][
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
                    print("Eigenvalues did not converge.")
                    continue

                # Offscreen rendering
                rgb_rendering, depth_rendering = r.render(scene, flags=flags)
                scene.remove_node(cam_node)

                end = time.process_time()
                times[index] = end - start

                # cv2.imwrite saves depth map as a single channel img and as-is meaning
                # if max depth is x, then max of the saved img values will be x as well.
                # skimage.io.imsave saves the image normalized, so max value will always
                # be 255 or 65k depending on the data type. plt.imsave saves RGBA image
                # with depth remapped to a color depending on a colormap used.
                # For possible further recalculation, npz with raw depth map is also saved.
                np.save(f"{rendered_image_prefix}_depth.npy", depth_rendering)
                cv2.imwrite(
                    f"{rendered_image_prefix}_depth.png",
                    depth_rendering.astype(np.uint16),
                )

                img_rendering = Image.fromarray(rgb_rendering)
                img_rendering.putalpha(255)
                img_rendering.save(f"{rendered_image_prefix}_color.png")

            # try:
            # except AssertionError:
            #     print(joined_dataset_query_path)
            #     print(traceback.format_exc())
        if not args.just_jsons:
            r.delete()

    if not args.just_jsons:
        times = np.array(times)
        print(f"Rendering time per image was ({np.mean(times)} +- {np.std(times)}) s.")

    with open(Path(args.src_output) / "matrices_for_rendering.json", "w") as ff:
        json.dump(matrices_dict, ff, indent=4)

    with open(Path(args.src_output) / "matrices_for_rendering_29.json", "w") as ff:
        json.dump(matrices_dict_29, ff, indent=4)

    with open(Path(args.src_output) / "matrices_for_rendering_53.json", "w") as ff:
        json.dump(matrices_dict_53, ff, indent=4)
