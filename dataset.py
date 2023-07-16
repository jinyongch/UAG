import random

import cv2
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RSOC(Dataset):
    def __init__(self, root, train=True, downsample=True):
        self.train = train
        if self.train:
            root = root * 4
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.downsample = downsample

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"

        img_path = self.lines[index]

        img, target = self.load_data(img_path)

        img = self.transform(img)

        return img, target

    def load_data(self, img_path):
        img = Image.open(img_path).convert("RGB")

        gt_path = (
            img_path.replace(".jpg", ".h5")
            .replace(".png", ".h5")
            .replace("images", "ground_truth")
        )

        with h5py.File(gt_path, "r") as f:
            target = np.asarray(f["density_map"])
            target = target.squeeze()

        if (
            ("small-vehicle" in gt_path)
            or ("large-vehicle" in gt_path)
            or ("ship" in gt_path)
        ):
            target = (
                cv2.resize(target, (1024, 768), interpolation=cv2.INTER_CUBIC)
                * target.shape[0]
                / 768
                * target.shape[1]
                / 1024
            )

        if self.train:
            crop_size = (img.size[0] // 2, img.size[1] // 2)
            if random.randint(0, 9) <= 4.5:
                dx = int(random.randint(0, 1) * img.size[0] * 1.0 / 2)
                dy = int(random.randint(0, 1) * img.size[1] * 1.0 / 2)
            else:
                dx = int(random.random() * img.size[0] * 1.0 / 2)
                dy = int(random.random() * img.size[1] * 1.0 / 2)

            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            target = target[dy : crop_size[1] + dy, dx : crop_size[0] + dx]

            if random.random() > 0.5:
                target = np.fliplr(target)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.downsample:
            target = (
                cv2.resize(
                    target,
                    (target.shape[1] // 8, target.shape[0] // 8),
                    interpolation=cv2.INTER_CUBIC,
                )
                * 64
            )
        else:
            target = cv2.resize(
                target,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        return img, target
