"""Join COLMAP-rendered datasets into one prepared for dataset_utils.py."""
import os
import random
from pathlib import Path


def link(view, part):
    stem = view.stem.split("_")[0]
    print(f"Linking {stem} to {part.name}")

    def link_(src, dst):
        if not dst.exists():
            os.link(src, dst)

    link_(view, part / view.name)
    link_(view.parent / f"{stem}_reference.png", part / f"{stem}_reference.png")
    link_(view.parent / f"{stem}_depth.png", part / f"{stem}_depth.png")
    link_(view.parent / f"{stem}_color.png", part / f"{stem}_color.png")


if __name__ == "__main__":
    random.seed(42)

    prefix = Path("/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender")

    train = prefix / "joined_dataset" / "train"
    val = prefix / "joined_dataset" / "val"
    test = prefix / "joined_dataset" / "test"
    os.makedirs(train, exist_ok=True)
    os.makedirs(val, exist_ok=True)
    os.makedirs(test, exist_ok=True)

    for hall in prefix.glob("2019*"):
        views = list((hall / "rendered" / "train").glob("*_depth.npy"))
        random.shuffle(views)
        split = int(0.2 * (len(views) - 15))
        for view in views[:15]:
            link(view, test)
        for view in views[16:split]:
            link(view, val)
        for view in views[split:]:
            link(view, train)
