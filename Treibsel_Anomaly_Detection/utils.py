import math
import os
import logging
logging.basicConfig(level=logging.WARNING)

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import torchvision
torchvision.disable_beta_transforms_warning()
from torchmetrics.functional.classification import binary_jaccard_index, binary_recall, binary_f1_score
from dataset import SeegrasDataset
from torch.utils.data import DataLoader

root = "./Treibsel_Anomaly_Detection/data/"
NUM_DATA = len(os.listdir(root + "train_images"))

def save_checkpoint(state, filename=root + "my_checkpoint.pth.tar"):
    logging.info("Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    logging.info("Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = SeegrasDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SeegrasDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def generate_metrics(loader, model, device: str = "cpu") -> dict[str, float]:
    logging.info("Evaluating Model")
    recall = 0
    f1_score = 0
    jaccard = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1).int() # because mask/label has no RGB-channels
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).int()

            recall += binary_recall(preds, y)
            f1_score += binary_f1_score(preds, y)
            jaccard += binary_jaccard_index(preds, y)           
    
    recall = recall/len(loader)
    f1_score = f1_score/len(loader)
    jaccard = jaccard/len(loader)

    metrics = {
        "recall" : recall,
        "f1_score": f1_score,
        "jaccard_index": jaccard
    }
    model.train()

    return metrics

def save_predictions_as_imgs(loader, model, folder: str = "saved_images/", device: str = "cpu"):
    logging.info("Saving Generated Masks")
    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{root}{folder}pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{root}{folder}{idx}.png")
    model.train()


def write_metrics_to_TensorBoard(metrics: dict[str, float], epoch: int, run_name: str = "Test run") -> None:
    """Method for writing calculated model metrics to a Tensorboard after each successful training epoch.
    For every epoch the gathered metrics should be named identical to achieve consitency for depiction.

    Args:
        metrics (dict[str, float]): A dictionary containing the key-value-pairs of evaluation metrics.
        epoch (int): Zero-indexed epoch number.
        run_name (str, optional): Name of run for differentiation of multiple test runs. Defaults to "Test run".
    """
    logging.info("Saving metrics to TensorBoard...")
    writer.add_scalars(run_name, metrics, epoch)
    writer.flush()

def write_loss_to_TensorBoard(loss: float, batch_id: int, batch_size: int, epoch: int) -> None:
    """Method for writing the loss calculated after every training batch of an epoch to a Tensorboard. 

    Args:
        loss (float): Calculated loss for the batch.
        batch_id (int): Zero-indexed batch number.
        batch_size (int): Size of the batch.
        epoch (int): Zero-indexed epoch the batch is used in.
    """
    logging.info("Saving loss of batch to TensorBoard...")
    factor = math.ceil(NUM_DATA / batch_size)
    writer.add_scalar("Loss per Batch", loss, ((epoch * factor) + batch_id))
    writer.flush()