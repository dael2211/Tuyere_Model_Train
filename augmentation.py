import albumentations as A

def get_training_augmentation(mean=None, std=None):
    """
    Creates a training augmentation pipeline.
    Args:
        mean (list or array): Mean values for each channel [R, G, B].
        std (list or array): Standard deviation values for each channel [R, G, B].
    Returns:
        An Albumentations Compose object with training augmentations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.1, p=0.3),
        A.RandomCrop(height=480, width=480, p=0.25),
        A.PadIfNeeded(min_height=608, min_width=800, always_apply=True),
        A.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.Perspective(p=0.1),
        A.OneOf([A.CLAHE(p=1), A.RandomGamma(p=1)], p=0.3),
        A.Normalize(mean=mean, std=std, always_apply=True)
    ])

def get_validation_augmentation(mean=None, std=None):
    """
    Creates a validation augmentation pipeline.
    Args:
        mean (list or array): Mean values for each channel [R, G, B].
        std (list or array): Standard deviation values for each channel [R, G, B].
    Returns:
        An Albumentations Compose object with validation augmentations.
    """
    return A.Compose([
        A.PadIfNeeded(min_height=608, min_width=800, p=1),
        A.Normalize(mean=mean, std=std, p=1)
    ])