# TODO: use last two indices of shape for h/w in case shape changes?
import random
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode

import yolov3


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, target: dict):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class Normalize:
    def __init__(self, mean: float, std: float, inplace: bool = False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, image, target):
        image = torchvision.transforms.functional.normalize(
            image, self.mean, self.std, self.inplace
        )
        return image, target

class RandomSelect:
    def __init__(self, transforms: list, p: float = 0.5):
        """
        Apply a single transform from ``transforms`` with probability `p`.
        In other words, on each call, there is a `1 - p` probability that
        no transform will be applied. If a transform is applied, only one
        from ``transforms`` will be randomly selected with probability
        ``1 / len(transforms)``.
        """
        self.transforms = transforms
        self.p = p

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        if self.p < torch.rand(1):
            return image, target
        transform = random.choice(self.transforms)
        image, target = transform(image, target)
        return image, target


class RandomCrop:
    def __init__(self, size: Union[int, Sequence[int]]):
        """
        Args:
            size: Either an int (for a `size` x `size` crop) or a 2-tuple
                of ints representing (w, h).
        """
        if isinstance(size, int):
            size = (size, size)
        elif len(size) != 2:
            raise TypeError("Size must be an int or a sequence of two ints")
        self.size = size

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        # TODO: pad image if smaller than crop?
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(image, self.size)
        image = torchvision.transforms.functional.crop(image, i, j, h, w)

        # Shift all bboxes and remove any that now fall outside the bounds of
        # the crop.
        target[:, [0, 2]] -= j
        target[:, [1, 3]] -= i

        x_out_of_bounds = torch.logical_or(
            torch.logical_and(target[:, 0] < 0, target[:, 2] < 0),
            torch.logical_and(target[:, 0] >= w, target[:, 2] >= w)
        )
        y_out_of_bounds = torch.logical_or(
            torch.logical_and(target[:, 1] < 0, target[:, 3] < 0),
            torch.logical_and(target[:, 1] >= h, target[:, 3] >= h)
        )
        out_of_bounds = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
        target = target[~out_of_bounds, :]

        # Set negative top left coordinates to zero.
        target[:, 0][target[:, 0] < 0] = 0
        target[:, 2][target[:, 2] < 0] = 0

        return image, target


class Repeat:
    def __init__(self, transform, k: int):
        """
        Repeat ``transform`` `k` times and concatenate the result along
        the batch dimension.
        """
        self.transform = transform
        self.k = k

    def __call__(self, image, target):
        images = []
        targets = []

        for _ in range(self.k):
            image_, target_ = self.transform(image, target.clone())
            images.append(image_)
            targets.append(target_)

        images = torch.stack(images)
        return images, targets


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        if self.p < torch.rand(1):
            return image, target
        image = torchvision.transforms.functional.hflip(image)
        h, w = image.shape[1:]
        target[:, [0, 2]] = w - target[:, [2, 0]]
        return image, target


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        if self.p < torch.rand(1):
            return image, target
        image = torchvision.transforms.functional.vflip(image)
        h, w = image.shape[1:]
        target[:, [1, 3]] = h - target[:, [3, 1]]
        return image, target


class Resize:
    def __init__(
        self,
        size: Sequence[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[bool] = None
    ):
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size must be an int or a sequence of two ints")
        elif isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError(
                "If size is a sequence, it must have 1 or two values"
            )
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        old_h, old_w = image.shape[1:]

        image = torchvision.transforms.functional.resize(
            image, self.size, self.interpolation, self.max_size, self.antialias
        )

        new_h, new_w = image.shape[1:]
        scale_x = old_w / new_w
        scale_y = old_h / new_h

        new_X = torch.div(target[:, [0, 2]], scale_x, rounding_mode="trunc")
        new_Y = torch.div(target[:, [1, 3]], scale_y, rounding_mode="trunc")

        target[:, [0, 2]] = new_X.type(target.dtype)
        target[:, [1, 3]] = new_Y.type(target.dtype)

        return image, target


class ToTensor:
    def __call__(self, image: np.ndarray, target: dict):
        """
        Convert target dict to an Mx6 int tensor of M detections, where column
        indices 0-3 are the bbox (x1, y1, x2, y2) coordinates, 4 is the class
        id
        """
        image = torchvision.transforms.functional.to_tensor(image)

        bbox = torch.tensor(target["bbox"].values.tolist())
        class_ = torch.tensor(target["class"].values)[:, None]
        #image_id = torch.tensor(target["image_id"].values)[:, None]
        target = torch.hstack((bbox, class_))

        return image, target
