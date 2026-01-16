import numpy as np
import matplotlib.pyplot as plt
import random
import os
from utils import reverse_normalize, denormalize
from matplotlib.colors import ListedColormap


# define reusable color map for correct mask printing
CLASS_COLORS = [
    (0, 0, 0),       # class 0: black
    (255, 0, 0),     # class 1: red
    (0, 255, 0),     # class 2: green
    (0, 0, 255),     # class 3: blue
    (255, 255, 0)    # class 4: yellow
]
CLASS_COLORS = np.array(CLASS_COLORS) / 255.0  # normalize to [0, 1] for matplotlib
fixed_cmap = ListedColormap(CLASS_COLORS)  # reusable color map

# visualize functions
def visualize_samples(dataset, num_samples=8):
    indices = random.sample(range(len(dataset)), num_samples)
    for idx in indices:
        image, mask = dataset[idx]
        visualize(image=image, mask=mask)


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name)
        if name == 'image':
            image = reverse_normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='gray')
            unique_values = np.unique(image)
            for value in unique_values:
                y, x = np.where(image == value)
                if len(y) > 0 and len(x) > 0:
                    plt.text(float(x[0]), float(y[0]), str(value),
                             color='yellow', fontsize=14, ha='center', va='center',
                             fontweight='bold')
    plt.pause(0.5)  # display for half a second while epoch
    plt.close()


def visualize_masks(
    exp_dir, visualize_count, image, true_mask, pred_mask, std, mean,
    save_path=None, cmap=fixed_cmap
):
    """
    Visualize image, true mask, and predicted mask with a fixed color map for consistent class representation.

    Parameters:
    - exp_dir: str, experiment directory where visualizations will be saved.
    - visualize_count: int, visualization counter.
    - image: numpy array, original image (CHW format).
    - true_mask: numpy array, ground truth mask (HxW, with discrete class labels).
    - pred_mask: numpy array, predicted mask (HxW, with discrete class labels).
    - std: list or numpy array, standard deviation for image normalization.
    - mean: list or numpy array, mean for image normalization.
    - save_path: str, optional, path to save the visualization.
    - cmap: ListedColormap, color map for masks (default: fixed_cmap).
    """
    # Set default save path
    if save_path is None:
        save_path = os.path.join(exp_dir, 'graphs', f'visualization_epoch_{visualize_count}.png')
        visualize_count += 1

    # Create figure
    plt.figure(figsize=(12, 6))

    # original image
    plt.subplot(1, 3, 1)
    plt.imshow(denormalize(image.transpose(1, 2, 0), mean, std))  # convert CHW to HWC
    plt.title("Image")
    plt.axis("off")

    # ground truth mask (with fixed color map)
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap=cmap, vmin=0, vmax=len(CLASS_COLORS) - 1)
    plt.title("Ground Truth")
    plt.axis("off")

    # predicted mask (with fixed color map)
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap=cmap, vmin=0, vmax=len(CLASS_COLORS) - 1)
    plt.title("Prediction")
    plt.axis("off")

    # Save the visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(save_path)
    plt.pause(0.5)  # display 0.5 sec
    plt.close()

