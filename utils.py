import os
import numpy as np
import cv2
import inspect
import logging
import sys
import torch

def create_experiment_directory(base_dir='../runs'):
    """
    creates an experiment folder for the current training run
    looks for the lowest possible exp_n folder
    includes subfolders for the models,logs,results and graphs
    """
    os.makedirs(base_dir, exist_ok=True)
    exp_dirs = [d for d in os.listdir(base_dir) if d.startswith('exp')]
    if exp_dirs:
        exp_nums = sorted([int(d[3:]) for d in exp_dirs if d[3:].isdigit()])
        next_exp_num = exp_nums[-1] + 1
    else:
        next_exp_num = 1
    exp_dir = os.path.join(base_dir, f'exp{next_exp_num}')
    os.makedirs(exp_dir)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'graphs'), exist_ok=True)
    return exp_dir

def create_test_run_dir(base_dir="test_runs"):
    """create a unique folder for the test run"""
    os.makedirs(base_dir, exist_ok=True)
    test_number = 1
    while os.path.exists(os.path.join(base_dir, f"test_{test_number}")):
        test_number += 1
    test_run_dir = os.path.join(base_dir, f"test_{test_number}")
    os.makedirs(test_run_dir, exist_ok=True)
    return test_run_dir

def reverse_normalize(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = image.transpose(1, 2, 0)  # convert CHW to HWC
    image = (image * std + mean)  # reverse normalization
    image = np.clip(image, 0, 1)  # ensure values are within [0, 1] range
    return image


def calculate_mean_std(images_dir, file_list):
    """
    Calculate the mean and standard deviation for each channel in the dataset.
    :param images_dir: Directory containing image files.
    :param file_list: List of image filenames to include in the calculation.
    :return: Tuple (mean, std) where each is a NumPy array of shape (3,) for RGB channels.
    """
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    num_pixels = 0

    for img_filename in file_list:
        img_path = os.path.join(images_dir, img_filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            pixels = img.shape[0] * img.shape[1]
            num_pixels += pixels
            channel_sum += np.sum(img, axis=(0, 1))
            channel_sq_sum += np.sum(np.square(img), axis=(0, 1))

    mean = channel_sum / num_pixels
    std = np.sqrt(channel_sq_sum / num_pixels - np.square(mean))

    return mean, std

def get_model_hyperparameters(model):
    model_signature = inspect.signature(model.__class__.__init__)  #
    model_params = {
        param: getattr(model, param, "Not Set")  # each attribute from the model object
        for param in model_signature.parameters
        if param != "self"  # skip 'self'
    }
    return model_params




def setup_terminal_logging(log_file_path):
    """
    redirects terminal output (stdout and stderr) to a log file and console.

    args:
        log_file_path (str): the path where the terminal output log file will be saved.

    returns:
        none
    """
    # create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create a file handler to log to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # create a console handler to keep logs visible in the terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # set logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.INFO)


class StreamToLogger:
    """
    file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        if "\n" in self.buffer:
            lines = self.buffer.splitlines(keepends=True)
            for line in lines[:-1]:
                line = line.strip()
                if line:
                    self.logger.log(self.log_level, line)
            self.buffer = lines[-1]

    def flush(self):
        if self.buffer.strip():
            self.logger.log(self.log_level, self.buffer.strip())
            self.buffer = ""

def denormalize(image, mean, std):
    """
    Reverse the normalization for a given image.

    Args:
        image (Tensor): The normalized image.
        mean (list): Mean used during normalization.
        std (list): Standard deviation used during normalization.

    Returns:
        Tensor: Denormalized image.
    """
    mean = np.array(mean)
    std = np.array(std)

    # when  the input image is in (H, W, C), transpose it to (C, H, W)
    if image.shape[-1] == len(mean):
        image = image.transpose(2, 0, 1) #  (H, W, C) to (C, H, W)

    # denormalize
    denormalized_image = image * std[:, None, None] + mean[:, None, None]

    denormalized_image = denormalized_image.transpose(1, 2, 0)
    return denormalized_image

def get_dataset_stats(images_dir):
    """
    Calculate mean and std of a dataset.

    Args:
        images_dir (str): Path to the directory containing images.

    Returns:
        tuple: (mean, std)
    """
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Uninitialized tensors for pixel values
    r_pixels, g_pixels, b_pixels = torch.tensor([]), torch.tensor([]), torch.tensor([])

    for image_path in image_files:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Separate channels and normalize to [0, 1]
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        r_pixels = torch.cat((r_pixels, torch.tensor(r.flatten() / 255.0, dtype=torch.float32)), 0)
        g_pixels = torch.cat((g_pixels, torch.tensor(g.flatten() / 255.0, dtype=torch.float32)), 0)
        b_pixels = torch.cat((b_pixels, torch.tensor(b.flatten() / 255.0, dtype=torch.float32)), 0)

    mean = (r_pixels.mean().item(), g_pixels.mean().item(), b_pixels.mean().item())
    std = (r_pixels.std().item(), g_pixels.std().item(), b_pixels.std().item())

    return mean, std
