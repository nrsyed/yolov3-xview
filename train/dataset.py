import concurrent.futures
import os
import pathlib

import cv2
import numpy as np
import pandas as pd
import torch


def compute_stats(fpath):
    img = cv2.imread(str(fpath))
    img = img.reshape(-1, 3).astype(np.float32) / 255.0
    sum_ = np.sum(img, axis=0)
    sum_sq = np.sum(img**2, axis=0)
    n_pixels = img.shape[0]
    return sum_, sum_sq, n_pixels


def dataset_stats(image_dir: pathlib.Path, labels_fpath: pathlib.Path):
    """
    Compute mean and stdev of each channel of an image dataset.

    https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    """
    df = pd.read_parquet(labels_fpath)
    fnames = df["id"].unique()

    sum_ = np.zeros((3,))
    sum_sq = np.zeros((3,))
    n_pixels = 0

    pool = concurrent.futures.ThreadPoolExecutor()
    futures = []

    for fname in fnames:
        img_fpath = image_dir / fname
        future = pool.submit(compute_stats, img_fpath)
        futures.append(future)

    pool.shutdown()

    for future in futures:
        _sum, _sum_sq, _n_pixels = future.result()
        sum_ += _sum
        sum_sq += _sum_sq
        n_pixels += _n_pixels

    mean = sum_ / n_pixels
    var = (sum_sq / n_pixels) - (mean**2)
    std = np.sqrt(var)

    print(mean)
    print(var)
    print(std)


class XViewDataset(torch.utils.data.Dataset):
    def __init__(
        self, image_dir: pathlib.Path, labels_fpath: pathlib.Path
    ):
        self.image_dir = image_dir
        self.labels_fpath = labels_fpath
        self.labels = pd.read_parquet(labels_fpath)
        self.images = os.listdir(image_dir).sort()

    def __getitem__(self, idx: int):
        pass

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    xview_dir = pathlib.Path("~/datasets/xview").expanduser()
    image_dir = xview_dir / "images"
    train_labels_path = xview_dir / "labels" / "xview_train.parquet"
    dataset_stats(image_dir, train_labels_path)
