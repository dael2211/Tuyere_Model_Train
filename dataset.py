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
        """
        Initialize the dataset with directory paths, class information, normalization parameters,
        and optional augmentation.
        :param images_dir: Directory containing image files.
        :param masks_dir: Directory containing corresponding mask files.
        :param classes: List of classes in the dataset.
        :param mean: Mean values for each channel [R, G, B].
        :param std: Standard deviation values for each channel [R, G, B].
        :param augmentation: Optional augmentation to be applied to the samples.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.classes = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        """
        retrieve an item by index.
        :param idx: index of the item to retrieve.
        :return: tuple (image, mask) where both are numpy arrays.
        """
        if i >= len(self.ids) or i < 0:
            raise IndexError(f"Index {i} out of range for dataset with size {len(self.ids)}")

        # get image path
        image_path = os.path.join(self.images_dir, self.ids[i])

        # find mask path with different file type
        mask_stem = os.path.splitext(self.ids[i])[0]
        mask_path = os.path.normpath(os.path.join(self.masks_dir, f"{mask_stem}.png")) if self.masks_dir is not None else None

        # check if the mask exists
        if mask_path is not None and not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found for: {mask_stem}")

        #read image and mask
        image = cv2.imread(image_path)
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
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            return image, mask
        else:
            return image, np.zeros(1) # Return a dummy mask with a small size

    def __len__(self):
        """
        returns the number of items in the dataset.
        :return: integer count of items.
        """
        return len(self.ids)

    def get_max_class_value(self):
        """
        returns the maximum class value found in the dataset.
        this value is cached to avoid recalculating.
        :return: integer maximum class value.
        """
        max_class_value = 0
        for idx in self.ids:
            # replace the image extension with .png for mask path
            mask_stem = os.path.splitext(idx)[0]
            mask_path = os.path.join(self.masks_dir, f"{mask_stem}.png")  # masks are .pngs

            mask = cv2.imread(mask_path, 0)
            if mask is not None:
                max_class_value = max(max_class_value, mask.max())
        max_class_id = max_class_value
        print(f"max class value in mask {max_class_value}")
        return max_class_id
