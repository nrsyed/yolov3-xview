import pathlib

import cv2
import numpy as np
import torch
import torchvision

import yolov3

from dataset import XViewDataset
import transforms


def freeze_darknet53_backbone(model: yolov3.Darknet):
    # Freeze layers 0-74, which comprise Darknet-53.
    for i in range(75):
        model.modules_[i].requires_grad_(False)


if __name__ == "__main__":
    config_path = "yolov3_xview.cfg"
    weights_path = "yolov3_xview.pth"
    device = "cuda"

    xview_dir = pathlib.Path("~/datasets/xview").expanduser()
    image_dir = xview_dir / "images"
    train_labels_path = xview_dir / "labels" / "xview_train.parquet"
    val_labels_path = xview_dir / "labels" / "xview_val.parquet"

    state_dict = torch.load(weights_path)
    net = yolov3.Darknet(config_path, device=device)
    net.load_state_dict(state_dict)
    freeze_darknet53_backbone(net)

    #dataset_stats(image_dir, train_labels_path)

    inp_h = net.net_info["height"]
    inp_w = net.net_info["width"]
    inp_mean = net.net_info["mean"]
    inp_std = net.net_info["std"]

    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(inp_mean, inp_std),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.Repeat(transforms.RandomCrop((inp_h, inp_w)), 8),
    ])
    dataset = XViewDataset(
        image_dir, train_labels_path, transforms=transforms_
    )

    exit()
    import random
    for _ in range(100):
        i = random.randint(0, len(dataset) - 1)
        chips, targets = dataset[i]
        
        for chip, target in zip(chips, targets):
            img = chip
            img = img.detach().numpy().transpose([1, 2, 0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            target = target.detach().numpy()
            for (x1, y1, x2, y2, cls_idx) in target:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
                class_ = dataset.idx_to_class[cls_idx]
                cv2.putText(
                    img, class_, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0)
                )

            cv2.imshow("img", img)
            cv2.waitKey(0)
