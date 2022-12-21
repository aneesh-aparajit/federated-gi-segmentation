import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import cv2 as cv

from config import Config

cfg = Config()


def rle_decode(mask_rle, shape):
    """
    Extracted from: https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode/script

    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(img):
    """
    Extracted from: https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode/script

    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def get_rgb_mask(
    #     img_path: str,
    rle_sequence: str,
    _class: str,
    height: int,
    width: int,
):
    """
    This function generates the RGB masks with the color mapped to the region in the GI.

    Args:
        - rle_sequence: the Encoded Pixels in str format
        - _class: the region you want to map to in the GI
        - height: height of the input
        - width: width of the input

    Returns:
        rgb mask: np.array
    """
    dec = rle_decode(rle_sequence, (height, width))
    rgb = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            if dec[i, j] == 1:
                rgb[i, j] = cfg.COLORS[_class]

    #     rgb = np.uint8(rgb)
    return rgb


def build_metadata():
    PATHS, HEIGHT, WIDTH, ID = [], [], [], []
    for case in tqdm(os.listdir(cfg.BASE_DIR)):
        if case == ".DS_Store":
            continue
        for day in os.listdir(os.path.join(cfg.BASE_DIR, case)):
            if day == ".DS_Store":
                continue
            DAYS += 1
            for file in os.listdir(os.path.join(cfg.BASE_DIR, case, day, "scans")):
                if file == ".DS_Store":
                    continue
                slices = file.split("_")
                _id = day + "_" + slices[0] + "_" + slices[1]
                ID.append(_id)
                HEIGHT.append(slices[2])
                WIDTH.append(slices[3])
                PATHS.append(os.path.join(cfg.BASE_DIR, case, day, "scans", file))
                TOTAL += 1

            print(
                f'dir: {os.path.join(cfg.BASE_DIR, case, day, "scans")}, # of files: {len(os.path.join(cfg.BASE_DIR, case, day, "scans"))}'
            )
    return pd.DataFrame({"id": ID, "img_path": PATHS, "WIDTH": WIDTH, "HEIGHT": HEIGHT})
