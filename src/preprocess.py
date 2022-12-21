from config import Config
from utils import rle_decode, build_metadata
import pandas as pd


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


if __name__ == "__main__":
    pass
