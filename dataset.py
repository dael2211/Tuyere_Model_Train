import os
import cv2
from torch.utils.data import Dataset as BaseDataset
import numpy as np


class CombustionChamberDataset(BaseDataset):
    def __init__(
        self,
        images_dir,
        masks_dir=None,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.ids = os.listdir(images_dir)
        if masks_dir:
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        else:
            self.masks_fps = []

        # convert str names to class values on masks
        if classes:
            self.class_values = [i for i, cls in enumerate(classes)]
        else:
            self.class_values = []

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        if self.masks_fps:
            mask = cv2.imread(self.masks_fps[i], 0)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype("float")
            return image, mask
        else:
            return image, np.zeros((256, 256), dtype=np.float32)

    def __len__(self):
        return len(self.ids)


