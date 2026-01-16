import yaml
import os
import torch
import csv
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from model import CombustionChamberModel
from dataset import CombustionChamberDataset
from utils import create_experiment_directory
from utils import calculate_mean_std
from augmentation import get_training_augmentation, get_validation_augmentation
from utils import get_model_hyperparameters
from utils import setup_terminal_logging
from training_curves import plot_metrics_individual


def main():
    # load the configuration from YAML file
    with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    # setup the base directory for the script
    base_dir = os.path.dirname(__file__)

    # create a new experiment directory for this run
    experiment_dir = create_experiment_directory(os.path.join(base_dir, 'runs'))

    # dynamic device configuration
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = torch.cuda.device_count()
    else:
        accelerator = 'cpu'
        devices = 1

    # configure logging
    log_dir = os.path.join(experiment_dir, 'logs')  # log subdirectory
    # specify the path for the TEMINAL log file
    terminal_log_path = "terminal_output_log.txt"
    setup_terminal_logging(terminal_log_path)

    # paths should be relative
    images_dir = os.path.join(base_dir, config['dataset']['images_dir'])
    masks_dir = os.path.join(base_dir, config['dataset']['masks_dir'])

    # retrieve and sort file list to ensure consistent ordering
    all_files = sorted(os.listdir(images_dir))
    all_indices = list(range(len(all_files)))

    # split ds indices into training and validation
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=config['dataset']['val_size'],
        random_state=config['dataset']['random_seed']
    )

    # full dataset
    full_dataset = CombustionChamberDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        classes=config['dataset']['classes'],
        augmentation=None  # different augmentation for subsets later
    )

    # split into training and validation subsets
    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, val_indices)

    # use the train_dataset's indices to calculate mean and std
    train_files_idxs = [full_dataset.ids[idx] for idx in train_indices]
    mean, std = calculate_mean_std(images_dir, train_files_idxs)

    # add mean and std to the config object under the "training" section for later model use
    config["training"]["mean"] = mean.tolist()
    config["training"]["std"] = std.tolist()

    # attach stats
    full_dataset.mean = mean.tolist()
    full_dataset.std = std.tolist()

    #  augmentations for train and valid
    train_dataset.dataset.augmentation = get_training_augmentation(mean=full_dataset.mean, std=full_dataset.std)
    valid_dataset.dataset.augmentation = get_validation_augmentation(mean=full_dataset.mean, std=full_dataset.std)

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, persistent_workers=True,
                              num_workers=config['training']['num_workers'], pin_memory=config['training']['pin_memory'])
    valid_loader = DataLoader(valid_dataset, batch_size=config['training']['batch_size'], shuffle=False, persistent_workers=True,
                              num_workers=config['training']['num_workers'], pin_memory=config['training']['pin_memory'])
    """
    # initialize deeplabv3+ model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_depth=5,
        encoder_weights="imagenet",
        encoder_output_stride=16,
        decoder_channels=256,
        decoder_atrous_rates=(12, 24, 36),
        in_channels=3,
        classes=4,
        activation=None,
        upsampling=4
    )
    """

    # initialize UNet model
    model = smp.Unet(
        encoder_name="resnet34",       # Encoder backbone
        encoder_depth=5,              # Number of stages in the encoder
        encoder_weights="imagenet",   # Pre-trained weights for encoder
        decoder_channels=(256, 128, 64, 32, 16),  # Channels in decoder blocks
        in_channels=3,                # Input channels (3 for RGB images)
        classes=4,                    # Number of output classes
        activation=None,              # Activation function (e.g., "softmax" or None for logits)
        aux_params=None               # Auxiliary output for classification (if needed)
    )


    model_parameter = get_model_hyperparameters(model=model)
    combustion_model = CombustionChamberModel(model, list(range(valid_dataset.dataset.max_class_id + 1)) ,
                                              exp_dir=experiment_dir, mean=full_dataset.mean, std=full_dataset.std)

    # callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(experiment_dir, 'models'),
        filename='best-checkpoint',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['callbacks']['early_stopping']['patience'],
        mode='min',
        verbose=True
    )

    logger = TensorBoardLogger(save_dir=log_dir, name="combustion_chamber_model")

    #combine config and model parameter for logging
    log_param ={
        **config,
        "model": model_parameter
    }
    # sanatize
    log_param_san = {k: (v if isinstance(v, (int, float, str, bool, torch.Tensor)) else str(v)) for k, v in
                 log_param.items()}
    logger.experiment.add_hparams(log_param_san, metric_dict={"initial_metric": 0.0}) #dummy metric for initialisation

    # dynamically create the input tensor
    batch_size = config["training"].get("batch_size", 1)
    channels = 3
    input_height = log_param["model"].get("input_height", 224)
    input_width = log_param["model"].get("input_width", 224)

    #create dummy tensor with relevant model input shape
    dummy_tensor = torch.randn(int(batch_size), int(channels), int(input_height), int(input_width))
    logger.experiment.add_graph(model, dummy_tensor)

    # set trainer using configuration
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        precision=config['training']['precision'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches']
    )

    # train the model
    trainer.fit(combustion_model, train_loader, valid_loader)

    # save model
    torch.save(combustion_model, os.path.join(experiment_dir, 'models', 'combustion_model.pt'))


    # write the metrics to CSV
    with open(os.path.join(log_dir,'training_epoch_metrics.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        # header
        writer.writerow(['epoch', 'train_loss', 'train_dice', 'val_loss', 'val_dice', 'val_f1', 'val_iou'])
        # write metrics for each epoch
        for epoch, metrics in combustion_model.epoch_logs.items():
            writer.writerow([
                epoch,
                metrics.get('train_loss', 'N/A'),
                metrics.get('train_dice', 'N/A'),
                metrics.get('val_loss', 'N/A'),
                metrics.get('val_dice', 'N/A'),
                metrics.get('val_f1', 'N/A'),
                metrics.get('val_iou', 'N/A')
            ])

    # write the config to the exp dir
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as file:
        yaml.safe_dump(config, file)


    # plot the metrics
    plot_metrics_individual(log_dir, "training_epoch_metrics.csv")
if __name__ == '__main__':
    main()
