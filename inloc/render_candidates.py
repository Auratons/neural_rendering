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
        "--src_ply_root",
        type=Path,
        help="Path with rendered scans and scan in global CS themselves.",
        default=Path(
            "/home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_pyrender"
        ),
    )
    parser.add_argument(
        "--src_output",
        type=Path,
        help="path to write output images",
        default=Path(
            "/home/kremeto1/inloc/datasets/pipeline-inloc-lifted-conv5-pyrender_corrected_missing_points/candidate_renders"
        ),
    )
    parser.add_argument(
        "--point_size", type=float, default=3.0, help="Rendering point size"
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
        "--input_poses",
        type=Path,
        help="Path to densePE_top100_shortlist matfile from InLoc run.",
        default="/home/kremeto1/inloc/datasets/pipeline-inloc-lifted-conv5-pyrender_corrected_missing_points/densePE_top100_shortlist.mat",
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


# fmt: off
scans = {
    "cse": {
        "CSE3": ["000", "001", "002", "003", "004", "005", "006"],
        "CSE4": ["009", "010", "011", "012", "013", "015", "016", "017", "018", "019", "020", "021", "022", "023", "024", "025", "026", "027", "028", "029", "030", "031", "032", "033", "034", "036", "037", "038", "039", "040", "041", "042", "043", "044", "045", "046", "047", "048", "049", "050", "051", "052", "053", "054", "055", "129", "130", "131", "132", "133", "134", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159"],
        "CSE5": ["058", "059", "060", "061", "062", "063", "064", "065", "066", "067", "068", "069", "070", "071", "072", "073", "074", "075", "076", "077", "078", "079", "080", "081", "082", "083", "084", "085", "086", "087", "088", "089", "090", "092", "093", "094", "095", "096", "097", "098", "099", "100", "101", "102", "103", "104", "105", "111", "112", "113", "114", "115", "116", "121", "122", "123", "124", "125", "126", "127", "128", "162", "163", "164", "165"]
    },
    "DUC": {
        "DUC1": ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015", "016", "017", "018", "019", "020", "021", "022", "023", "024", "025", "068", "069", "070", "071", "072", "073", "074", "075", "076", "077", "078", "079", "080", "081", "082", "083", "084", "085", "086", "087", "088", "089", "090", "091"],
        "DUC2": ["027", "028", "029", "030", "031", "032", "033", "034", "035", "036", "037", "038", "039", "040", "041", "042", "043", "044", "045", "046", "047", "048", "049", "050", "051", "052", "053", "054", "055", "056", "057", "058", "059", "060", "061", "062", "063", "064", "065", "066", "092", "093", "094", "095", "096", "097", "098", "099", "100", "101", "102", "103", "104", "105", "106", "107", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132"]
    }
}
# fmt: on


if __name__ == "__main__":
    args = parse_args()

    rnd = random.Random(42)
    # Create output folders
    os.makedirs(args.src_output, exist_ok=True)
    mat = sio.loadmat(args.input_poses)

    if not args.just_jsons:
        flags = RenderFlags.FLAT | RenderFlags.RGBA
        # Loading the mesh / pointcloud
        thread_times = [0.0] * len(mat["ImgList"][0])
        process_times = [0.0] * len(mat["ImgList"][0])

    test_size = len(mat["ImgList"][0])
    matrices_dict = {"train": {}, "val": {}}
    per_scan_matrices = {}

    k = np.array(
        [
            [1385.0, 0.0, 800.0, 0.0],
            [0.0, 1385.0, 600.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    k2 = np.array(
        [
            [1385.0, 0.0, 600.0, 0.0],
            [0.0, 1385.0, 800.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    r = pyrender.OffscreenRenderer(k[0, 2] * 2, k[1, 2] * 2, point_size=args.point_size)
    r2= pyrender.OffscreenRenderer(k2[0, 2] * 2, k2[1, 2] * 2, point_size=args.point_size)

    for index in range(test_size):
        print("Processing image {}/{}".format(index + 1, test_size))

        query_path = Path(str(mat["ImgList"][0][index][0][0]))

        candidate_paths = [Path(str(i[0])) for i in mat["ImgList"][0][index][1][0]]
        # candidate_scores = mat['ImgList'][0][index][2][0]
        candidate_projections = [i for i in mat["ImgList"][0][index][3][0]]
        # candidate_k = mat['ImgList'][0][index][4][0]
        # candidate_r = mat["ImgList"][0][index][5][0]
        # candidate_t = mat["ImgList"][0][index][6][0]

        if all([np.isnan(candidate_projections[tidx]).any() for tidx in range(len(candidate_projections))]):
            continue

        os.makedirs(args.src_output / query_path.stem, exist_ok=True)

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
                print("DIVERGED")
                continue

            # if camera_pose[2, -1] > (160 - 20):
            #     localized_scan = "CSE5"
            #     camera_pose[2, -1] -= 160
            # elif camera_pose[2, -1] > (120 - 20):
            #     localized_scan = "CSE4"
            #     camera_pose[2, -1] -= 120
            # elif camera_pose[2, -1] > (80 - 20):
            #     localized_scan = "CSE3"
            #     camera_pose[2, -1] -= 80
            # elif camera_pose[2, -1] > (40 - 20):
            #     localized_scan = "DUC2"
            #     camera_pose[2, -1] -= 40
            # else:
            #     localized_scan = "DUC1"

            db_scan_number = candidate_paths[tidx].name.split("_")[2]
            for building in scans.keys():
                if building in candidate_paths[tidx].name:
                    building_shorcut = building
                    break

            for exact_building in scans[building_shorcut].keys():
                if db_scan_number in scans[building_shorcut][exact_building]:
                    db_exact_building = exact_building
                    break

            ply_path_for_localized_query = (
                args.src_ply_root
                / db_exact_building
                / db_scan_number
                / f"{building_shorcut}_scan_{db_scan_number}_30M.ptx.ply"
            )

            if cv2.imread(f"/home/kremeto1/neural_rendering/datasets/raw/inloc/query/resized/{query_path.name}").shape[:2] == (1600, 1200):
                usek2 = True
            else:
                usek2 = False

            # rendered image
            rendered_image_prefix = str(
                args.src_output / query_path.stem / candidate_paths[tidx].stem
            )
            rendered_image_params = {
                "calibration_mat": [list(l) for l in list(k2)] if usek2 else [list(l) for l in list(k)],
                "camera_pose": [list(l) for l in list(camera_pose)],
                "source_reference": str(query_path),
                "retrieved_database_reference": str(candidate_paths[tidx]),
                "source_scan": "unknown",
                "source_scan_ply_path": "unknown",
                "localized_scan": f"{db_exact_building}/{db_scan_number}",
                "localized_scan_ply_path": str(ply_path_for_localized_query),
            }

            if str(ply_path_for_localized_query) not in per_scan_matrices:
                per_scan_matrices[str(ply_path_for_localized_query)] = {"train": {}, "val": {}}

            per_scan_matrices[str(ply_path_for_localized_query)]["train"][
                f"{rendered_image_prefix}_color.png"
            ] = rendered_image_params
            matrices_dict["train"][
                f"{rendered_image_prefix}_color.png"
            ] = rendered_image_params
            with open(f"{rendered_image_prefix}_params.json", "w") as ff:
                json.dump(rendered_image_params, ff, indent=4)

            if not args.just_jsons:
                mesh = load_ply(ply_path_for_localized_query, args.voxel_size)
                scene = pyrender.Scene(
                    bg_color=[float(i) for i in args.bg_color.split(",")]
                )
                scene.add(mesh)

                # Return the value (in fractional seconds) of the sum of the system
                # and user CPU time of the current thread. It does not include time
                # elapsed during sleep.
                thread_start = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
                # Return the value (in fractional seconds) of the sum of the system
                # and user CPU time of the current process. It does not include time
                # elapsed during sleep. Since these calls utilize more threads, this
                # may significantly overgrowth the "real" sys+cpu time seen by
                # observing the main thread.
                process_start = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)

                if usek2:
                    camera = pyrender.camera.IntrinsicsCamera(
                        k2[0, 0], k2[1, 1], k2[0, 2], k2[1, 2]
                    )
                else:
                    camera = pyrender.camera.IntrinsicsCamera(
                        k[0, 0], k[1, 1], k[0, 2], k[1, 2]
                    )
                try:
                    cam_node = scene.add(camera, pose=camera_pose)
                    # Offscreen rendering
                    rgb_rendering, depth_rendering = r2.render(scene, flags=flags)
                except np.linalg.LinAlgError:
                    print("Eigenvalues did not converge.", flush=True)
                    continue

                thread_end = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
                process_end = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)

                del scene
                del mesh
                thread_times[index] = thread_end - thread_start
                process_times[index] = process_end - process_start

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

                # rendered image
                img_rendering = Image.fromarray(rgb_rendering)
                img_rendering.putalpha(255)
                img_rendering.save(
                    str(
                        args.src_output
                        / query_path.stem
                        / f"{candidate_paths[tidx].stem}_color.png"
                    )
                )
                print(f'Rendered {str(args.src_output / query_path.stem / f"{candidate_paths[tidx].stem}_color.png")}', flush=True)
                print(
                    f"CPU thread time: {(thread_end - thread_start) * 1000:.2f} ms", flush=True
                )
                # Count of processors available to the job on this node. Note the
                # select/linear plugin allocates entire nodes to jobs, so the value
                # indicates the total count of CPUs on the node. For the select/cons_res
                # plugin, this number indicates the number of cores on this node
                # allocated to the job.
                print(
                    f"CPU process time (uses one core): {(process_end - process_start) * 1000:.2f}) ms", flush=True
                )

    if not args.just_jsons:
        thread_times = np.array(thread_times)
        process_times = np.array(process_times)
        print(np.mean(thread_times))
        print(np.std(thread_times))
        print(np.mean(process_times))
        print(np.std(process_times))

        print(
            f"CPU thread time: {np.mean(thread_times) * 1000:.2f} ms +- {np.std(thread_times):.2f} ms", flush=True
        )
        # Count of processors available to the job on this node. Note the
        # select/linear plugin allocates entire nodes to jobs, so the value
        # indicates the total count of CPUs on the node. For the select/cons_res
        # plugin, this number indicates the number of cores on this node
        # allocated to the job.
        print(
            f"CPU process time (uses one core): {np.mean(process_times) * 1000:.2f} ms +- {np.std(process_times):.2f} ms) ms", flush=True
        )

    # with open(args.src_output / "matrices_for_rendering.json", "w") as ff:
    #     json.dump(matrices_dict, ff, indent=4)

    with open(args.src_output / "per_scan_matrices_for_rendering.json", "w") as ff:
        json.dump(per_scan_matrices, ff, indent=4)
    
    for key in per_scan_matrices.keys():
        with open(args.src_output / f"matrices_for_rendering-{str(Path(key).name)}.json", "w") as ff:
            json.dump(per_scan_matrices[key], ff, indent=4)
        with open(args.src_output / f"matrices_for_rendering-{str(Path(key).name)}.txt", "w") as ff:
            print(key, file=ff)
