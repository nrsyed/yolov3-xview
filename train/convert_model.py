import collections
import pathlib

import numpy as np
import torch

import yolov3


def load_network(
    config_path: pathlib.Path,
    weights_path: pathlib.Path = None,
    device: str = "cuda",
) -> yolov3.Darknet:
    model = yolov3.Darknet(config_path, device=device)

    if weights_path is not None:
        model.load_weights(weights_path)
    model.cuda(device=device)
    return model


def get_test_input(model: yolov3.Darknet, device: str = "cuda"):
    img_h, img_w = model.net_info["height"], model.net_info["width"]
    img = np.random.randint(0, 255, (3, img_h, img_w))
    inp = img[np.newaxis, ...].astype(np.float32) / 255.0
    inp = torch.tensor(inp, device=device)
    return inp


def rectify_state_dicts(
    dict1: collections.OrderedDict, dict2: collections.OrderedDict
) -> collections.OrderedDict:
    rectified = collections.OrderedDict()
    for k in dict1:
        if k in dict2 and dict1[k].shape == dict2[k].shape:
            rectified[k] = dict1[k]
    return rectified


if __name__ == "__main__":
    orig_config_path = "../../pytorch-yolov3/models/yolov3.cfg"
    new_config_path = "yolov3_xview.cfg"
    weights_path = "../../pytorch-yolov3/models/yolov3.weights"

    net1 = load_network(orig_config_path, weights_path)
    net2 = load_network(new_config_path)

    state_dict1 = net1.state_dict()
    state_dict2 = net2.state_dict()
    rectified_state_dict = rectify_state_dicts(state_dict1, state_dict2)
    net2.load_state_dict(rectified_state_dict, strict=False)

    # Save the new state dict.
    torch.save(net2.state_dict(), "yolov3_xview.pth")
