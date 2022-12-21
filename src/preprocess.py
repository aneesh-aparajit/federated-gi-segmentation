from config import Config
from utils import rle_decode, build_metadata, get_rgb_mask
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2 as cv


def main() -> None:
    cfg = Config()
    df1 = pd.read_csv(cfg.TRAIN_CSV_PATH)
    df2 = build_metadata()
    df = df1.merge(df2, on="id")
    df = df.dropna().reset_index(drop=True)

    # Convert Width and Height to integer for safety
    df["WIDTH"] = df["WIDTH"].astype(int)
    df["HEIGHT"] = df["HEIGHT"].astype(int)

    # Save this to the new save path
    df.to_csv(cfg.NEW_TRAIN_CSV_PATH, index=False)

    ids = df["id"].tolist()

    for _id in tqdm(ids):
        subset = df[df["id"] == _id].reset_index(drop=True)
        img = plt.imread(subset.iloc[0]["img_path"])
        final_mask = None
        for ix in range(len(subset)):
            _class = subset.iloc[ix]["class"]
            rle_sequence = subset.iloc[ix]["segmentation"]
            mask = get_rgb_mask(
                rle_sequence=rle_sequence,
                _class=_class,
                height=subset.iloc[ix]["HEIGHT"],
                width=subset.iloc[ix]["WIDTH"],
            )

            if final_mask is None:
                final_mask = mask
            else:
                final_mask = cv.bitwise_or(final_mask, mask)

        plt.imsave(os.path.join(cfg.DATASET_CKPT_DIR, "mask", f"{_id}.png"), final_mask)
        plt.imsave(
            os.path.join(cfg.DATASET_CKPT_DIR, "imgs" f"{_id}.png"), img, cmap="gray"
        )


if __name__ == "__main__":
    pass
