import os
import json
import argparse
from pathlib import Path


def link(view, part):
    stem = "_".join(view.stem.split("_")[:-1])
    print(f"Linking {stem} to {part.name}")

    def link_(src, dst):
        if src.exists():
            if not dst.exists():
                os.link(src, dst)

    link_(view.parent / f"{stem}_reference.png", part / f"{stem}_reference.png")
    # link_(view.parent / f"{stem}_depth.png", part / f"{stem}_depth.png")
    # link_(view.parent / f"{stem}_depth.npy", part / f"{stem}_depth.npy")
    # link_(view.parent / f"{stem}_color.png", part / f"{stem}_color.png")


def get_matrices(reference_path, dataset_path, matrices_d):
    stem = "_".join(reference_path.stem.split("_")[:-1])
    params_file = reference_path.parent / f"{stem}_params.json"
    with open(params_file, "r") as f:
        params = json.load(f)
    matrices_d["train"][str(dataset_path / f"{stem}_color.png")] = {
        # 3x3 -> 4x4
        "intrinsic_matrix": [
            i + [j] for i, j in zip(params["calibration_mat"], [0, 0, 0])
        ]
        + [[0, 0, 0, 1]],
        "extrinsic_matrix": params["camera_pose"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        help="path experiment root",
        default=Path("/nfs/projects/artwin/experiments/hololens_mapper"),
    )
    parser.add_argument(
        "glob",
        nargs="?",
        type=str,
        help="dataset name glob to process",
        default="2019-*-rendered-mesh-black_bg",
    )
    parser.add_argument(
        "output_suffix",
        nargs="?",
        type=str,
        help="suffix to globbed names to be created",
    )
    args = parser.parse_args()

    prefix = args.root.absolute()

    halls = sorted(list(prefix.glob(args.glob)))
    for hall in halls:
        matrices_dict = {"train": {}, "val": {}}
        output = prefix / f"{hall.name}-{args.output_suffix}"
        os.makedirs(output / "train", exist_ok=True)
        os.makedirs(output / "val", exist_ok=True)
        output = output / "train"
        print(f"Processing {hall.name}")
        reference_photos = list((hall / "images").glob("*_reference.png"))
        for ref in reference_photos:
            get_matrices(ref, output, matrices_dict)
            link(ref, output)

        with open(output / "matrices_for_rendering.json", "w") as f:
            f.write(json.dumps(matrices_dict, indent=4))
