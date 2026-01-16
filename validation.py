import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice, MulticlassF1Score, MulticlassJaccardIndex, ConfusionMatrix
import numpy as np
import argparse
import onnxruntime as ort
import cv2
import yaml
from visualize import visualize_masks
from utils import create_test_run_dir
from dataset import CombustionChamberDataset
from augmentation import get_validation_augmentation


def get_num_classes(masks_dir):
    """
    Determines the number of classes by finding the maximum pixel value in the mask images.
    Assumes masks are grayscale images where pixel values correspond to class indices.
    """
    max_val = 0
    if not os.path.exists(masks_dir):
        print(f"Warning: Mask directory not found at {masks_dir}. Assuming 1 class and no metrics for masks.")
        return 1

    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not mask_files:
        print(f"Warning: No masks found in {masks_dir}. Assuming 1 class and no metrics for masks.")
        return 1

    for mask_file in mask_files:
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            max_val = max(max_val, mask.max())
    # Number of classes is max_val + 1 (e.g., if max is 0, there's 1 class; if 1, 2 classes)
    return max_val + 1


def main(model_path, dataset_dir):
    # No longer using a config file for ONNX test
    # All necessary paths and parameters will be derived or set directly

    test_images_dir = os.path.join(dataset_dir, "test_images")
    test_masks_dir = os.path.join(dataset_dir, "test_masks")

    # Dynamically determine the number of classes
    num_classes = get_num_classes(test_masks_dir)
    classes = [str(i) for i in range(num_classes)]  # Define classes based on num_classes

    # Set parameters directly
    batch_size = 1  # Set batch size to 1 for testing, or can be a parameter
    mean = [0.485, 0.456, 0.406]  # Standard ImageNet mean
    std = [0.229, 0.224, 0.225]   # Standard ImageNet std

    # setup test run directories
    test_run_dir = create_test_run_dir()
    results_dir = os.path.join(test_run_dir, "results")
    predicted_masks_dir = os.path.join(test_run_dir, "predicted_masks")  # Define predicted_masks_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predicted_masks_dir, exist_ok=True)  # Create the directory

    # prepare dataset and dataloader
    test_dataset = CombustionChamberDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        classes=classes,
        augmentation=get_validation_augmentation(mean=mean, std=std)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load ONNX model and create inference session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name

    # initialize metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dice_metric = Dice(num_classes=num_classes, average="none").to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="none").to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, average="none").to(device)
    confusion_matrix = ConfusionMatrix(num_classes=num_classes).to(device)

    # evaluation loop
    all_true_masks, all_pred_masks = [], []

    with torch.no_grad():
        for idx, (images, true_masks) in enumerate(test_loader):
            images_np = images.numpy()
            true_masks = true_masks.to(device).long()

            # model inference
            ort_inputs = {input_name: images_np}
            ort_outs = ort_session.run(None, ort_inputs)
            logits = torch.from_numpy(ort_outs[0]).to(device)
            preds = logits.argmax(dim=1)

            # update metrics
            dice_metric.update(preds, true_masks)
            f1_metric.update(preds, true_masks)
            iou_metric.update(preds, true_masks)
            confusion_matrix.update(preds, true_masks)

            # Save predicted mask as an image
            for i in range(preds.size(0)):
                # Get original filename to save the mask with the same name
                original_filename = os.path.basename(test_dataset.ids[idx * batch_size + i])
                mask_save_path = os.path.join(predicted_masks_dir, original_filename)

                pred_mask_image = preds[i].cpu().numpy().astype(np.uint8)
                cv2.imwrite(mask_save_path, pred_mask_image)

            # store masks for metrics
            all_true_masks.append(true_masks.cpu().numpy())
            all_pred_masks.append(preds.cpu().numpy())

            # visualize first 5 batches
            if idx < 5:
                for i in range(images.size(0)):
                    img = images[i].cpu().numpy()
                    true_mask = true_masks[i].cpu().numpy()
                    pred_mask = preds[i].cpu().numpy()
                    save_path = os.path.join(results_dir, f"result_{idx}_{i}.png")
                    visualize_masks(
                        exp_dir=results_dir,
                        visualize_count=idx * batch_size + i,
                        image=img,
                        true_mask=true_mask,
                        pred_mask=pred_mask,
                        std=std,
                        mean=mean,
                        save_path=save_path,
                    )

    # calculate metrics
    avg_dice = dice_metric.compute().mean().cpu().item()
    avg_f1 = f1_metric.compute().mean().cpu().item()
    avg_iou = iou_metric.compute().mean().cpu().item()

    classwise_dice = dice_metric.compute().cpu().numpy()
    classwise_f1 = f1_metric.compute().cpu().numpy()
    classwise_iou = iou_metric.compute().cpu().numpy()
    conf_matrix = confusion_matrix.compute().cpu().numpy()

    # print metrics
    print(f"Average Metrics:")
    print(f"  Dice Score: {avg_dice:.4f}")
    print(f"  F1 Score: {avg_f1:.4f}")
    print(f"  IoU Score: {avg_iou:.4f}")

    print(f"\nClass-wise Metrics:")
    for cls, class_name in enumerate(classes):
        print(f"Class {class_name} (index {cls}):")
        print(f"  Dice: {classwise_dice[cls]:.4f}")
        print(f"  F1: {classwise_f1[cls]:.4f}")
        print(f"  IoU: {classwise_iou[cls]:.4f}")

    # save metrics and confusion matrix
    metrics_summary = {
        "Average Metrics": {
            "Dice Score": avg_dice,
            "F1 Score": avg_f1,
            "IoU Score": avg_iou,
        },
        "Class-wise Metrics": {
            "Dice Scores": classwise_dice.tolist(),
            "F1 Scores": classwise_f1.tolist(),
            "IoU Scores": classwise_iou.tolist(),
        },
        "Confusion Matrix": conf_matrix.tolist(),
    }
    metrics_path = os.path.join(results_dir, "metrics_summary.yaml")
    with open(metrics_path, "w") as file:
        yaml.dump(metrics_summary, file)

    np.save(os.path.join(results_dir, "confusion_matrix.npy"), conf_matrix)
    np.save(os.path.join(results_dir, "true_masks.npy"), np.concatenate(all_true_masks, axis=0))
    np.save(os.path.join(results_dir, "pred_masks.npy"), np.concatenate(all_pred_masks, axis=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Full path to the ONNX model file."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Full path to the test dataset directory."
    )
    args = parser.parse_args()
    main(args.model_path, args.dataset_dir)
