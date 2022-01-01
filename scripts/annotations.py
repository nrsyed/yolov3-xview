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

            #bounds = ann["properties"]["bounds_imcoords"]
            #bbox = [int(coord) for coord in bounds.split(",")]
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
        "railway vehicle": [
            "passenger car", "cargo car", "flat car", "tank car",
            "locomotive"
        ],
        "construction vehicle": ["engineering vehicle"],
        "truck": [
            "truck tractor w/ box trailer", "truck tractor", "trailer",
            "truck tractor w/ flatbed trailer", "truck tractor w/ liquid tank",
            "cargo truck", "utility truck",
        ],
        "car": ["small car", "passenger vehicle"],
        "pickup": ["pickup truck"],
        "plane": [
            "fixed-wing aircraft", "passenger/cargo plane", "small aircraft",
        ],
        "bulldozer": ["front loader/bulldozer"],
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


def init_parquet():
    fpath = "xView_train.geojson"
    class_labels_fpath = "xView1_baseline/xview_class_labels.txt"
    
    class_map = load_class_map(class_labels_fpath)
    anns = json.load(open(fpath, "r"))
    df = to_dataframe(anns, class_map)
    df = postprocess_annotations(df)
    df.to_parquet("xview_train.parquet")

if __name__ == "__main__":
    #init_parquet()

    anns_fpath = "xview_train.parquet"
    img_dir = pathlib.Path("train_images")

    df = pd.read_parquet(anns_fpath)

    fnames = list(df["id"].unique())
    for fname in fnames:
        fpath = img_dir / fname
        show_image(fpath, df)
