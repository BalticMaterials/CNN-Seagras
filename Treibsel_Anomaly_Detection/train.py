# https://www.youtube.com/watch?v=IHq1t7NxS8k
# Tensorboard start in seperate shell: tensorboard --logdir=runs

import logging
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    generate_metrics,
    save_predictions_as_imgs,
    write_metrics_to_TensorBoard,
    write_loss_to_TensorBoard,
)


# Hyperparameters etc.
RUN_NAME = "BCEWithLogitsLoss, Batch: 2, LR: 1e-2"
LEARNING_RATE = 1e-3
rates = [1e-4, 1e-3, 1e-2]

from loss_functions.dice import DiceLoss
from loss_functions.IoU import IoULoss
from loss_functions.tversky import TverskyLoss, FocalTverskyLoss
loss_functions = [nn.BCEWithLogitsLoss(), DiceLoss(), IoULoss(), TverskyLoss(), FocalTverskyLoss()]
LOSS_FN = loss_functions[0]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
batches = [2, 4, 8, 16, 32]
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 # changing size later in the training process to increase accuracy, then resize with nearest interpolation
IMAGE_WIDTH = 240 
PIN_MEMORY = True
LOAD_MODEL = False
PATH = "Treibsel_Anomaly_Detection/"
TRAIN_IMG_DIR = PATH + "data/train_images/"
TRAIN_MASK_DIR = PATH + "data/train_masks/"
VAL_IMG_DIR = PATH + "data/val_images/"
VAL_MASK_DIR = PATH + "data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    loop = tqdm(loader) # Progressbar
    loss_in_epoch = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        write_loss_to_TensorBoard(loss=loss, batch_id=batch_idx, epoch=epoch, batch_size=BATCH_SIZE)
        loss_in_epoch += loss.item()
    return loss_in_epoch / (batch_idx + 1)


def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ],)

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ],)


    for loss_function in loss_functions:
        for batch_size in batches:
            for lr in rates:
                # TODO: Reset model and make new training
                RUN_NAME = f"{loss_function._get_name()}, Batch: {batch_size}, LR: {lr}"
                LOSS_FN = loss_function
                BATCH_SIZE = batch_size
                LEARNING_RATE = lr


                logging.info(f"Star of run: {RUN_NAME}")
                model = UNET(in_channels=3, out_channels=1).to(DEVICE)    
                loss_fn = LOSS_FN
                
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

                train_loader, val_loader = get_loaders(
                    TRAIN_IMG_DIR,
                    TRAIN_MASK_DIR,
                    VAL_IMG_DIR,
                    VAL_MASK_DIR,
                    BATCH_SIZE,
                    train_transform,
                    val_transform,
                    NUM_WORKERS,
                    PIN_MEMORY
                )

                if LOAD_MODEL:
                    load_checkpoint(torch.load(PATH + "data/my_checkpoint.pth.tar"), model)
                
                scaler = torch.cuda.amp.GradScaler()
                for epoch in range(NUM_EPOCHS):
                    loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

                    # save model
                    if epoch == 49:
                        checkpoint = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(), 
                        }
                        save_checkpoint(checkpoint)
                        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

                    metrics = generate_metrics(val_loader, model, device=DEVICE)
                    metrics.update({"loss": loss})
                    write_metrics_to_TensorBoard(metrics, epoch, run_name=RUN_NAME)                        
                
                logging.info(f"End of run: {RUN_NAME}")

if __name__ == "__main__":
    main()
    