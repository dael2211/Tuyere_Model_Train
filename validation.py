import os
import torch
import numpy as np
import argparse
import onnxruntime as ort
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main(model_path, dataset_dir):
    # All necessary paths and parameters will be derived or set directly
    images_dir = dataset_dir

    # Set parameters directly
    mean = [0.485, 0.456, 0.406]  # Standard ImageNet mean
    std = [0.229, 0.224, 0.225]   # Standard ImageNet std

    # setup test run directories
    predicted_masks_dir = os.path.join(dataset_dir, "predicted_masks")
    os.makedirs(predicted_masks_dir, exist_ok=True)

    # Augmentation pipeline
    augmentation = A.Compose([
        A.PadIfNeeded(min_height=608, min_width=800, p=1),
        A.Normalize(mean=mean, std=std, p=1),
        ToTensorV2()
    ])

    # load ONNX model and create inference session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # evaluation loop
    with torch.no_grad():
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            augmented = augmentation(image=image)
            image_tensor = augmented['image'].unsqueeze(0) # Add batch dimension

            images_np = image_tensor.numpy()

            # model inference
            ort_inputs = {input_name: images_np}
            ort_outs = ort_session.run(None, ort_inputs)
            logits = torch.from_numpy(ort_outs[0])
            preds = logits.argmax(dim=1)

            # Save predicted mask as an image
            mask_save_path = os.path.join(predicted_masks_dir, image_file)

            pred_mask_image = preds[0].cpu().numpy().astype(np.uint8)
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
