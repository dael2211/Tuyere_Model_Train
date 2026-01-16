import torch
import yaml
import random
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex
from torchmetrics import Dice
from visualize import visualize_masks, fixed_cmap


# model Class
class CombustionChamberModel(pl.LightningModule):
    def __init__(self, model, num_classes, exp_dir, mean, std, config_path='config.yaml'):
        super(CombustionChamberModel, self).__init__()
        self.model = model
        self.num_classes_loss = len(num_classes)
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass', classes=num_classes)
        self.exp_dir = exp_dir
        self.mean = mean
        self.std = std

        # load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.learning_rate = self.config['training']['learning_rate']
        self.visualize = self.config['logging']['visualize']
        self.visualize_count = 0

        # optimizer parameters
        self.optimizer_name = self.config['optimizer']['name']  # e.g., 'Adam', 'SGD'
        self.optimizer_params = self.config['optimizer'].get('params', {})  # optional params

        # metrics for training and validation
        self.train_dice = Dice(num_classes=self.num_classes_loss, average='macro')
        self.val_dice = Dice(num_classes=self.num_classes_loss, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes_loss, average='macro')
        self.val_iou = MulticlassJaccardIndex(num_classes=self.num_classes_loss)

        # store metrics for custom logging
        self.epoch_logs = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.long()
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        # calculate training dice coefficient
        preds = logits.argmax(dim=1)
        dice_score = self.train_dice(preds, masks)

        # log training loss and metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_dice', dice_score, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # save metrics to epoch logs
        if self.current_epoch not in self.epoch_logs:
            self.epoch_logs[self.current_epoch] = {}
        self.epoch_logs[self.current_epoch]['train_loss'] = loss.item()
        self.epoch_logs[self.current_epoch]['train_dice'] = dice_score.item()

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        masks = masks.long()

        loss = self.loss_fn(logits, masks)


        # calculate validation metrics
        preds = logits.argmax(dim=1)

        dice_score = self.val_dice(preds, masks)
        f1_score = self.val_f1(preds, masks)
        iou_score = self.val_iou(preds, masks)

        # log validation loss and metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_dice', dice_score,on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_f1', f1_score,on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_iou', iou_score,on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # save metrics to custom epoch logs
        if self.current_epoch not in self.epoch_logs:
            self.epoch_logs[self.current_epoch] = {}
        self.epoch_logs[self.current_epoch]['val_loss'] = loss.item()
        self.epoch_logs[self.current_epoch]['val_dice'] = dice_score.item()
        self.epoch_logs[self.current_epoch]['val_f1'] = f1_score.item()
        self.epoch_logs[self.current_epoch]['val_iou'] = iou_score.item()

        # visualize for the first batch
        if self.visualize and batch_idx == 0:
            random_idx = random.randint(0, images.size(0) - 1)
            pred_mask = preds.cpu().numpy()[random_idx]  # predicted mask
            true_mask = masks[random_idx].cpu().numpy()  # ground truth mask
            image = images[random_idx].cpu().numpy()  # original image

            visualize_masks(
                exp_dir=self.exp_dir,
                visualize_count=self.visualize_count,
                image=image,
                true_mask=true_mask,
                pred_mask=pred_mask,
                std=self.std,
                mean=self.mean,
                cmap=fixed_cmap
            )
            self.visualize_count += 1

        return loss

    def configure_optimizers(self):
        # dynamically create optimizer from config
        optimizer_class = getattr(torch.optim, self.optimizer_name, torch.optim.Adam)
        optimizer = optimizer_class(self.parameters(), lr=self.learning_rate, **self.optimizer_params)

        # scheduler configuration
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.config['training']['patience'],
            factor=self.config['training']['factor']
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
