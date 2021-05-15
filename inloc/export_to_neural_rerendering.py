import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_mode", type=str, default="random")
    parser.add_argument(
        "--val_buildings",
        type=str,
        default="CSE5 DUC2",
        help="If `split_mode` is set to `building`, use the renderings from the corresponding buildings as validation data.",
    )

    args = parser.parse_args()

    train_dir = os.path.join(args.output_path, "train")
    val_dir = os.path.join(args.output_path, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_num = 0
    val_num = 0

    correspondences = dict()

    # Mode "building" : The buildings given as parameters are used as validation data
    if args.split_mode == "building":
        val_buildings = args.val_buildings.split(" ")
        for building in os.listdir(args.input_path):
            print("Processing {} building...".format(building))
            for scan in tqdm(os.listdir(os.path.join(args.input_path, building))):
                temp_path = os.path.join(args.input_path, building, scan)
                temp_names = []
                for img_name in os.listdir(temp_path):
                    cutout = "_".join(img_name.split("_")[:-1])
                    if cutout not in temp_names:
                        if building in val_buildings:
                            for ext in ["reference", "depth", "color"]:
                                os.link(
                                    os.path.join(
                                        temp_path, "{}_{}.png".format(cutout, ext)
                                    ),
                                    os.path.join(
                                        val_dir,
                                        "{}_{}.png".format(str(val_num).zfill(5), ext),
                                    ),
                                )
                            val_num += 1
                            correspondences[
                                "val/{}".format(str(val_num).zfill(5))
                            ] = "∕".join([building, scan, cutout])
                        else:
                            for ext in ["reference", "depth", "color"]:
                                os.link(
                                    os.path.join(
                                        temp_path, "{}_{}.png".format(cutout, ext)
                                    ),
                                    os.path.join(
                                        train_dir,
                                        "{}_{}.png".format(
                                            str(train_num).zfill(5), ext
                                        ),
                                    ),
                                )
                            train_num += 1
                            correspondences[
                                "train/{}".format(str(train_num).zfill(5))
                            ] = "∕".join([building, scan, cutout])

    # Mode "random" : split all the images coming from every building randomly
    if args.split_mode == "random":
        # Gather all cutout names
        all_cutout_paths = []
        for building in os.listdir(args.input_path):
            print("Processing {} building...".format(building))
            for scan in tqdm(os.listdir(os.path.join(args.input_path, building))):
                temp_path = os.path.join(args.input_path, building, scan)
                temp_names = []
                for img_name in os.listdir(temp_path):
                    cutout = "_".join(img_name.split("_")[:-1])
                    if cutout not in temp_names:
                        all_cutout_paths.append(os.path.join(temp_path, cutout))
                        temp_names.append(cutout)
        # Split them randomly
        n = len(all_cutout_paths)
        shuffle = np.random.choice(np.arange(n), n, replace=False)
        ratio = args.val_ratio * n
        for i in range(n):
            cutout_path = all_cutout_paths[shuffle[i]]
            if i < ratio:
                for ext in ["reference", "depth", "color"]:
                    os.link(
                        "{}_{}.png".format(cutout_path, ext),
                        os.path.join(
                            val_dir, "{}_{}.png".format(str(val_num).zfill(5), ext)
                        ),
                    )
                val_num += 1
                correspondences["val/{}".format(str(val_num).zfill(5))] = "∕".join(
                    [building, scan, cutout]
                )
            else:
                for ext in ["reference", "depth", "color"]:
                    os.link(
                        "{}_{}.png".format(cutout_path, ext),
                        os.path.join(
                            train_dir, "{}_{}.png".format(str(train_num).zfill(5), ext)
                        ),
                    )
                train_num += 1
                correspondences["train/{}".format(str(train_num).zfill(5))] = "∕".join(
                    [building, scan, cutout]
                )

    # Mode "balanced" : Split every scan in train and val sets and merge them (every building is present in the same proportion in each dataset)
    if args.split_mode == "balanced":
        val_cutout_paths = []
        train_cutout_paths = []
        for building in os.listdir(args.input_path):
            print("Processing {} building...".format(building))
            for scan in tqdm(os.listdir(os.path.join(args.input_path, building))):
                temp_path = os.path.join(args.input_path, building, scan)
                temp_names = []
                for img_name in os.listdir(temp_path):
                    cutout = "_".join(img_name.split("_")[:-1])
                    cutout = os.path.join(temp_path, cutout)
                    if cutout not in temp_names:
                        temp_names.append(cutout)
                n = len(temp_names)
                shuffle = np.random.choice(np.arange(n), n, replace=False)
                ratio = args.val_ratio * n
                for i in range(n):
                    cutout_path = temp_names[shuffle[i]]
                    if i < ratio:
                        for ext in ["reference", "depth", "color"]:
                            os.link(
                                "{}_{}.png".format(cutout_path, ext),
                                os.path.join(
                                    val_dir,
                                    "{}_{}.png".format(str(val_num).zfill(5), ext),
                                ),
                            )
                        val_num += 1
                        correspondences[
                            "val/{}".format(str(val_num).zfill(5))
                        ] = "∕".join([building, scan, cutout])
                    else:
                        for ext in ["reference", "depth", "color"]:
                            os.link(
                                "{}_{}.png".format(cutout_path, ext),
                                os.path.join(
                                    train_dir,
                                    "{}_{}.png".format(str(train_num).zfill(5), ext),
                                ),
                            )
                        train_num += 1
                        correspondences[
                            "train/{}".format(str(train_num).zfill(5))
                        ] = "∕".join([building, scan, cutout])

    # Save the correspondences
    with open(os.path.join(args.output_path, "correspondences.pkl"), "wb") as f:
        pickle.dump(correspondences, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
