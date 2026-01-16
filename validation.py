import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import onnxruntime as ort
import cv2
from utils import create_test_run_dir
from dataset import CombustionChamberDataset
from augmentation import get_validation_augmentation


def main(model_path, dataset_dir):
    # All necessary paths and parameters will be derived or set directly
    images_dir = dataset_dir

    # Set parameters directly
    batch_size = 1  # Set batch size to 1 for validation
    mean = [0.485, 0.456, 0.406]  # Standard ImageNet mean
    std = [0.229, 0.224, 0.225]   # Standard ImageNet std
    num_classes = 2 # Assuming 2 classes for now, can be an argument if needed

    # setup test run directories
    predicted_masks_dir = os.path.join(dataset_dir, "predicted_masks")
    os.makedirs(predicted_masks_dir, exist_ok=True)

    # prepare dataset and dataloader
    validation_dataset = CombustionChamberDataset(
        images_dir=images_dir,
        masks_dir=None,  # No masks for validation
        classes=[str(i) for i in range(num_classes)],
        augmentation=get_validation_augmentation(mean=mean, std=std)
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # load ONNX model and create inference session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name

    # evaluation loop
    with torch.no_grad():
        for idx, (images, _) in enumerate(validation_loader):
            images_np = images.numpy()

            # model inference
            ort_inputs = {input_name: images_np}
            ort_outs = ort_session.run(None, ort_inputs)
            logits = torch.from_numpy(ort_outs[0])
            preds = logits.argmax(dim=1)

            # Save predicted mask as an image
            for i in range(preds.size(0)):
                original_filename = os.path.basename(validation_dataset.ids[idx * batch_size + i])
                mask_save_path = os.path.join(predicted_masks_dir, original_filename)

                pred_mask_image = preds[i].cpu().numpy().astype(np.uint8)
                # Color the mask for better visualization if needed
                # For example, for 2 classes, you can map 0 to black and 1 to white
                # Or use a colormap for multiple classes
                colored_mask = pred_mask_image * 255 # Example for binary mask
                cv2.imwrite(mask_save_path, colored_mask)

    print(f"Validation complete. Predicted masks are saved in: {predicted_masks_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Full path to the ONNX model file."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Full path to the directory with validation images."
    )
    args = parser.parse_args()
    main(args.model_path, args.dataset_dir)
