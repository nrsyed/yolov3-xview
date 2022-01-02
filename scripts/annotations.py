import json
import os
import pathlib
import random
from typing import List

import cv2
import pandas as pd


def to_dataframe(anns: dict, class_map: dict):
    image_ids = []
    class_ids = []
    bboxes = []

    for ann in anns["features"]:
        class_id = ann["properties"]["type_id"]
        image_id = ann["properties"]["image_id"]

        if (class_id in class_map) and (image_id != "1395.tif"):
            class_ = class_map[class_id]
            bbox = ann["properties"]["bounds_imcoords"]
            
            image_ids.append(image_id)
            class_ids.append(class_)
            bboxes.append(bbox)

    df = pd.DataFrame(
        {
            "id": image_ids,
            "class": class_ids,
            "bbox": bboxes,
        }
    )
    return df


def load_class_map(fpath: pathlib.Path):
    class_map = dict()
    with open(fpath, "r") as f:
        for line in f:
            class_id, class_name = line.strip().split(":")

            class_id = int(class_id)
            class_name = class_name.lower()

            class_map[class_id] = class_name
    return class_map


def postprocess_annotations(df: pd.DataFrame):
    # Note: some class names on xview website
    # (https://challenge.xviewdataset.org/data-format) are wrong.
    exclude_classes = [
        "Building", "Hut/Tent", "Shed", "Aircraft Hangar",
        "Damaged Building", "Facility",
        "Tower crane", "Container Crane", "Helipad", "Pylon",
        "Shipping Container", "Shipping Container lot", "Storage Tank",
        "Vehicle Lot", "Construction Site", "Tower",
    ]
    exclude_classes = set([s.lower() for s in exclude_classes])

    # Remap specified "subclasses" (each element in the value/list) to a
    # different class name (each key).
    class_to_subclasses = {
        "railcar": [
            "passenger car", "cargo/container car", "flat car", "tank car",
            "locomotive", "railway vehicle",
        ],
        "industrial vehicle": [
            "engineering vehicle", "excavator", "haul truck",
            "cement mixer", "crane truck", "ground grader", "scraper/tractor",
            "reach stacker", "straddle carrier", "front loader/bulldozer",
            "mobile crane",
        ],
        "truck": [
            "truck tractor w/ box trailer", "truck tractor", "trailer",
            "truck tractor w/ flatbed trailer", "truck tractor w/ liquid tank",
            "cargo truck", "utility truck", "dump truck",
        ],
        "car": ["small car", "passenger vehicle"],
        "pickup": ["pickup truck"],
        "plane": [
            "fixed-wing aircraft", "passenger/cargo plane", "small aircraft",
        ],
    }

    df = df[~df["class"].isin(exclude_classes)]

    subclass_to_class = dict()
    for class_, subclasses in class_to_subclasses.items():
        for subclass in subclasses:
            subclass_to_class[subclass] = class_

    for _, row in df.iterrows():
        if row["class"] in subclass_to_class:
            row["class"] = subclass_to_class[row["class"]]

    return df


def show_image(fpath: pathlib.Path, df: pd.DataFrame):
    img = cv2.imread(str(fpath))

    fname = fpath.name
    df = df[df["id"] == fname]

    if df.empty:
        return

    for _, ann in df.iterrows():
        bbox = [int(val) for val in ann["bbox"].split(",")]
        cv2.rectangle(img, bbox[:2], bbox[2:], (0, 255, 0), thickness=1)
        cv2.putText(
            img, ann["class"], (bbox[0], bbox[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)
        )
    
    #img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow(fname, img)
    cv2.waitKey(0)
    cv2.destroyWindow(fname)


def convert_to_parquet(
    json_fpath: pathlib.Path, class_labels_fpath: pathlib.Path,
    dst_fpath: pathlib.Path
):
    class_map = load_class_map(class_labels_fpath)
    anns = json.load(open(json_fpath, "r"))
    df = to_dataframe(anns, class_map)
    df = postprocess_annotations(df)
    df = df.reset_index(drop=True)
    df.to_parquet(dst_fpath)
    print(f"Wrote annotations to {dst_fpath}")


def make_splits(
    df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
):
    """
    TODO
    """
    #unused_fnames = set(df["id"].unique())
    used_fnames = set()

    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)

    test_ratio = 1 - train_ratio - val_ratio

    # Order classes from fewest to most instances.
    # DataFrame.value_counts returns an object containing classes in order
    # of most to least frequent.
    classes = []
    for class_, count in df["class"].value_counts(ascending=True).items():
        classes.append((class_, count))

    for class_, count in classes:
        # Number of instances of the current class to be placed in each of
        # train, test, and val.
        n_train = int(round(train_ratio * count))
        n_val = int(round(val_ratio * count))
        n_test = max(0, count - n_train - n_val)

        # Subtract the number of instances already present in the respective
        # dataset splits.
        n_train = max(0, n_train - len(train_df[train_df["class"] == class_]))
        n_val = max(0, n_val - len(val_df[val_df["class"] == class_]))
        n_test = max(0, n_test - len(test_df[test_df["class"] == class_]))

        # Count the number of instances of the current class in each file,
        # using only files that have not yet been assigned to a split.
        # We create a list in descending order of count, greedily assigning
        # first to the train set, then val, then finally test.
        _df = df[df["class"] == class_]

        fnames = []
        for fname, fname_count in _df["id"].value_counts().items():
            if fname not in used_fnames:
                fnames.append((fname, fname_count))

        while n_train > 0 and fnames:
            fname, fname_count = fnames.pop(0)
            train_df = train_df.append(df[df["id"] == fname])
            used_fnames.add(fname)
            n_train -= fname_count

        while n_val > 0 and fnames:
            fname, fname_count = fnames.pop(0)
            val_df = val_df.append(df[df["id"] == fname])
            used_fnames.add(fname)
            n_val -= fname_count

        while n_test > 0 and fnames:
            fname, fname_count = fnames.pop(0)
            test_df = test_df.append(df[df["id"] == fname])
            used_fnames.add(fname)
            n_test -= fname_count

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_counts = dict(train_df["class"].value_counts())
    val_counts = dict(val_df["class"].value_counts())
    test_counts = dict(test_df["class"].value_counts())

    for class_ in df["class"].unique():
        n_train = train_counts[class_]
        n_val = val_counts[class_]
        n_test = test_counts[class_]
        n_tot = n_train + n_val + n_test
        print(
            f"{class_}: {n_train}, {n_val}, {n_test} "
            f"({n_train / n_tot:.2f}, {n_val / n_tot:.2f}, {n_test / n_tot:.2f})"
        )

    return train_df, val_df, test_df


if __name__ == "__main__":
    xview_dir = pathlib.Path("~/xview").expanduser()
    json_fpath = xview_dir / "xView_train.geojson"
    class_labels_fpath = xview_dir / "classes.txt"

    anns_fpath = xview_dir / "xview.parquet"
    train_fpath = xview_dir / "xview_train.parquet"
    val_fpath = xview_dir / "xview_val.parquet"
    test_fpath = xview_dir / "xview_test.parquet"

    img_dir = xview_dir / "train_images"

    #convert_to_parquet(json_fpath, class_labels_fpath, anns_fpath)
    df = pd.read_parquet(anns_fpath)
    train_df, val_df, test_df = make_splits(df)

    train_df.to_parquet(train_fpath)
    val_df.to_parquet(val_fpath)
    test_df.to_parquet(test_fpath)

    #df_ = df[df["class"] == "barge"]
    #fnames = list(df_["id"].unique())
    #for fname in fnames:
    #    fpath = img_dir / fname
    #    show_image(fpath, df)
