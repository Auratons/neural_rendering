import argparse
import numpy as np
import skimage.metrics
from pathlib import Path
from PIL import Image


def get_central_crop(img, crop_height=512, crop_width=512):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    assert len(img.shape) == 3, (
        "input image should be either a 2D or 3D matrix,"
        " but input was of shape %s" % str(img.shape)
    )
    height, width, ch = img.shape
    # assert height >= crop_height and width >= crop_width, (
    #     "input image cannot " "be smaller than the requested crop size"
    # )
    st_y = abs((height - crop_height) // 2)
    st_x = abs((width - crop_width) // 2)
    if height >= crop_height and width >= crop_width:
        return np.squeeze(img[st_y : st_y + crop_height, st_x : st_x + crop_width, :])
    else:
        new_img = np.zeros((crop_height, crop_height, ch), dtype=np.float32)
        new_img[st_y : st_y + height, st_x : st_x + width, :] = img
        return np.squeeze(new_img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--real_imgs_path",
        type=Path,
        help="Path where real world images reside.",
        default=Path(
            "/home/kremeto1/neural_rendering/datasets/post_processed/imc/grand_place_brussels_thesis_test_squarify-100_src-fused/val"
        ),
    )
    parser.add_argument(
        "--real_imgs_glob",
        type=str,
        help="Path glob for filtering proper images from path",
        default="*_reference.png",
    )
    parser.add_argument(
        "--rendered_imgs_path",
        type=Path,
        help="Path where rendered images reside.",
        default=None,
    )
    parser.add_argument(
        "--rendered_imgs_glob",
        type=str,
        help="Path glob for filtering proper images from path",
        default="*_color.png",
    )
    args = parser.parse_args()

    if args.rendered_imgs_path is None:
        args.rendered_imgs_path = args.real_imgs_path

    set1 = sorted(args.real_imgs_path.glob(args.real_imgs_glob))
    set2 = sorted(args.rendered_imgs_path.glob(args.rendered_imgs_glob))

    loss_l1 = 0
    loss_psnr = 0

    for i, (img1_path, img2_path) in enumerate(zip(set1, set2)):
        img1_in_ar = np.array(Image.open(str(img1_path)))[:, :, :3]
        img2_in_ar = np.array(Image.open(str(img2_path)))[:, :, :3]
        min_sh0 = min(img1_in_ar.shape[0], img2_in_ar.shape[0])
        min_sh1 = min(img1_in_ar.shape[1], img2_in_ar.shape[1])
        img1_in_ar = get_central_crop(img1_in_ar, min_sh0, min_sh1)
        img2_in_ar = get_central_crop(img2_in_ar, min_sh0, min_sh1)

        mask = np.any(img2_in_ar != [0, 0, 0], axis=-1)
        if np.all(mask == False):
            continue

        loss_psnr += skimage.metrics.peak_signal_noise_ratio(img1_in_ar[mask], img2_in_ar[mask])
        loss_l1 += np.mean(np.abs(img1_in_ar[mask].astype(np.float32) - img2_in_ar[mask].astype(np.float32)))

    print("*** mean %s loss for experiment = %f" % ("L1", loss_l1 / len(set1)))
    print("*** mean %s loss for experiment = %f" % ("PSNR", loss_psnr / len(set1)))
