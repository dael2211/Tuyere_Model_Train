import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice, MulticlassF1Score, MulticlassJaccardIndex, ConfusionMatrix
import numpy as np
import yaml
import argparse
from visualize import visualize_masks
from model import CombustionChamberModel
from utils import create_test_run_dir
from dataset import CombustionChamberDataset
from augmentation import get_validation_augmentation


def main(exp_number):
    # relative paths
    pwd = os.getcwd()
    exp_dir = os.path.join(pwd, "runs", f"exp{exp_number}")
    model_checkpoint_path = os.path.join(exp_dir, "models", "best-checkpoint.ckpt")
    config_path = os.path.join(exp_dir, "config.yaml")

    # load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    test_images_dir = os.path.join(pwd, "data", "Tuyere_ds_v3_test_set", "test_images")
    test_masks_dir = os.path.join(pwd, "data", "Tuyere_ds_v3_test_set", "test_masks")

    # read parameters from config
    batch_size = config["training"]["batch_size"]  # use batch_size from training
    num_classes = len(config["dataset"]["classes"])  # infer num_classes from class definitions
    mean = config["dataset"].get("mean", [0.485, 0.456, 0.406])  # default to ImageNet mean
    std = config["dataset"].get("std", [0.229, 0.224, 0.225])  # default to ImageNet std

    # setup test run directories
    test_run_dir = create_test_run_dir()
    results_dir = os.path.join(test_run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # prepare dataset and dataloader
    test_dataset = CombustionChamberDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        classes=config["dataset"]["classes"],
        augmentation=get_validation_augmentation(mean=mean, std=std)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombustionChamberModel.load_from_checkpoint(model_checkpoint_path, map_location=device).to(device)
    model.eval()

    # initialize metrics
    dice_metric = Dice(num_classes=num_classes, average="none").to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="none").to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, average="none").to(device)
    confusion_matrix = ConfusionMatrix(num_classes=num_classes).to(device)

    # evaluation loop
    all_true_masks, all_pred_masks = [], []

    with torch.no_grad():
        for idx, (images, true_masks) in enumerate(test_loader):
            images, true_masks = images.to(device), true_masks.to(device).long()

            # model inference
            logits = model(images)
            preds = logits.argmax(dim=1)

            # update metrics
            dice_metric.update(preds, true_masks)
            f1_metric.update(preds, true_masks)
            iou_metric.update(preds, true_masks)
            confusion_matrix.update(preds, true_masks)

            # store masks for visualization
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
    for cls, class_name in enumerate(config["dataset"]["classes"]):
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
        "--exp_number", type=int, required=True, help="experiment number to test (e.g., --exp_number 1)"
    )
    args = parser.parse_args()
    main(args.exp_number)
