import concurrent.futures
import os
import pathlib
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision


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
        self, image_dir: pathlib.Path, labels_fpath: pathlib.Path,
        transforms=None
    ):
        classes = [
            "barge", "bus", "car", "container ship", "ferry", "fishing vessel",
            "helicopter", "industrial vehicle", "maritime vessel", "motorboat",
            "oil tanker", "pickup", "plane", "railcar", "sailboat", "truck",
            "tugboat", "yacht"
        ]
        class_to_idx = dict()
        idx_to_class = dict()
        for i, class_ in enumerate(classes):
            class_to_idx[class_] = i
            idx_to_class[i] = class_
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

        self.labels_fpath = labels_fpath
        self.labels = pd.read_parquet(labels_fpath)

        # Convert class name strings to integers.
        self.labels["class"] = self.labels["class"].apply(
            lambda class_: class_to_idx[class_]
        )

        # Convert bboxes from str to list of ints, and set any
        # negative bbox coordinates to zero.
        self.labels["bbox"] = self.labels["bbox"].apply(
            lambda s: [max(0, int(val)) for val in s.split(",")]
        )

        self.image_dir = image_dir

        self.images = list(self.labels["id"].unique())
        self.images.sort(key=lambda fname: int(os.path.splitext(fname)[0]))

        self.transforms = transforms

    def __getitem__(self, idx: int):
        fname = self.images[idx]
        img_fpath = self.image_dir / fname
        img = cv2.cvtColor(cv2.imread(str(img_fpath)), cv2.COLOR_BGR2RGB)

        df = self.labels[self.labels["id"] == fname]

        target = {
            "bbox": df["bbox"],
            "class": df["class"],
            #"image_id": idx,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)
