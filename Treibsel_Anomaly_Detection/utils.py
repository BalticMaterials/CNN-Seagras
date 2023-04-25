import math
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import torchvision
torchvision.disable_beta_transforms_warning()
from torchmetrics.functional.classification import binary_jaccard_index, binary_precision, binary_recall, binary_f1_score
from dataset import SeegrasDataset
from torch.utils.data import DataLoader
root = "./Treibsel_Anomaly_Detection/PyTorch_Playground/UNET_example/data/"
NUM_DATA = 82

def save_checkpoint(state, filename=root + "my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
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

def dice_metric(prediction, target, smooth=1):
    prediction = prediction.view(-1)
    target = target.view(-1)
        
    intersection = (prediction * target).sum()                            
    dice = (2.*intersection + smooth)/(prediction.sum() + target.sum() + smooth)
    return dice

def generate_metrics(loader, model, device="cpu"):
    print("=> Evaluating Model")
    precision = 0
    recall = 0
    dice = 0
    f1_score = 0
    jaccard = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1).int() # because mask/label has no RGB-channels
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).int()

            precision += binary_precision(preds, y)
            recall += binary_recall(preds, y)
            dice += dice_metric(preds, y)
            f1_score += binary_f1_score(preds, y)
            jaccard += binary_jaccard_index(preds, y)           
    
    precision = precision/len(loader)
    recall = recall/len(loader)
    f1_score = f1_score/len(loader)
    dice = dice/len(loader)
    jaccard = jaccard/len(loader)

    metrics = {
        "precision" : precision,
        "recall" : recall,
        "f1_score": f1_score,
        "dice_coefficient": dice,
        "jaccard_index": jaccard
    }
    model.train()

    return metrics

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cpu"):
    print("=> Saving Generated Masks")
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

def write_metrics_to_TensorBoard(metrics, epoch):
    print("=> Saving Metrics to TensorBoard")
    writer.add_scalars("Test run", metrics, epoch)
    writer.flush()

def write_loss_to_TensorBoard(loss, batch_id, batch_size, epoch):
    factor = math.ceil(NUM_DATA / batch_size)
    writer.add_scalar("Loss per Batch", loss, ((epoch) * factor) + (batch_id))
    writer.flush()