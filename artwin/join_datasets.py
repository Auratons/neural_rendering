"""Join COLMAP-rendered datasets into one prepared for dataset_utils.py."""
import os
import random
import string
import json
import argparse
from pathlib import Path


def link(view, part, index, mapping):
    stem = view.stem.split("_")[0]
    print(f"Linking {stem} to {part.name}")

    def link_(src, dst):
        if not dst.exists():
            os.link(src, dst)

    mapping[str(view.parent / f"{stem}_reference.png")] = str(
        part / f"{index:04n}_reference.png"
    )
    link_(view.parent / f"{stem}_reference.png", part / f"{index:04n}_reference.png")
    link_(view.parent / f"{stem}_depth.png", part / f"{index:04n}_depth.png")
    link_(view.parent / f"{stem}_depth.npy", part / f"{index:04n}_depth.npy")
    link_(view.parent / f"{stem}_color.png", part / f"{index:04n}_color.png")


if __name__ == "__main__":
    rnd = random.Random(42)

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
        "output", nargs="?", type=str, help="name of output dataset", default=""
    )
    args = parser.parse_args()

    if args.output == "":
        args.output = "joined_dataset_" + "".join(
            random.Random(43).choice(string.ascii_uppercase + string.digits)
            for _ in range(10)
        )

    prefix = args.root.absolute()

    train = prefix / args.output / "train"
    val = prefix / args.output / "val"
    test = prefix / args.output / "test"
    os.makedirs(train, exist_ok=True)
    os.makedirs(val, exist_ok=True)
    os.makedirs(test, exist_ok=True)

    test_size = 50
    it = 0
    mapping = {}
    halls = sorted(list(prefix.glob(args.glob)))
    for hall in halls:  # Stratified split across halls
        print(f"Processing {hall.name}")
        reference_photos = list((hall / "train").glob("*_reference.png"))
        rnd.shuffle(reference_photos)
        split = int(0.2 * (len(reference_photos) - test_size))
        for ref in reference_photos[: int(test_size / len(halls))]:
            link(ref, test, it, mapping)
            it += 1
        for ref in reference_photos[int(test_size / len(halls)) : split]:
            link(ref, val, it, mapping)
            it += 1
        for ref in reference_photos[split:]:
            link(ref, train, it, mapping)
            it += 1

    with open(prefix / args.output / "mapping.txt", "w") as mapping_file:
        mapping_file.write(json.dumps(mapping, indent=4))
