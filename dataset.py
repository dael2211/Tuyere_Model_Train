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

        # the image^mask pair id, beware the id is the image file and the fitting mask might have a different file type
        mask_stems = {os.path.splitext(mask_id)[0] for mask_id in os.listdir(masks_dir)} if masks_dir is not None else set()
        self.ids = [img_id for img_id in os.listdir(images_dir)
                    if os.path.splitext(img_id)[0] in mask_stems]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] if classes is not None else None

        self.max_class_id = self.get_max_class_value() # cached value of the maximum class
        print(f"Images loaded from {self.images_dir}")
        if masks_dir is not None:
            print(f"Greyscale masks loaded from {self.masks_dir}")
        print(f"Dataset initialized with {len(self.ids)} items.")
        #print(f"IDs in dataset: {self.ids}")



    def __getitem__(self, idx):
        """
        retrieve an item by index.
        :param idx: index of the item to retrieve.
        :return: tuple (image, mask) where both are numpy arrays.
        """
        if idx >= len(self.ids) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset with size {len(self.ids)}")

        # get image path
        image_path = os.path.join(self.images_dir, self.ids[idx])

        # find mask path with different file type
        mask_stem = os.path.splitext(self.ids[idx])[0]
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

        if self.masks_dir:
            mask = cv2.imread(mask_path, 0)
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
            return image, np.zeros_like(image) # Return a dummy mask

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
