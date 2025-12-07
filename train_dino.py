import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from lightly.data import LightlyDataset, DINOCollateFunction
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
import timm
import os
import glob

#Configuration
IMAGE_FOLDER = "./data/images/24h_day1"
BATCH_SIZE = 32
EPOCHS = 100
SAVE_EVERY = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "dino_checkpoints"

#Using DINOv2 as the SSL
BACKBONE_NAME = "vit_small_patch14_dinov2.lvd142m"


def train_dino():
    print(f"Starting DINO training on {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Loading backbone: {BACKBONE_NAME}")
    try:
        backbone = timm.create_model(BACKBONE_NAME, pretrained=True)
    except Exception as e:
        print(f"Error loading DINOv2: {e}")
        print("Falling back to ViT-Small (patch16, 224).")
        backbone = timm.create_model("vit_small_patch16_224", pretrained=True)

    backbone.head = nn.Identity()

    input_dim = 384
    head = DINOProjectionHead(input_dim, 2048, 256, 2048, batch_norm=False)

    student_backbone = backbone
    student_head = head

    try:
        teacher_backbone = timm.create_model(BACKBONE_NAME, pretrained=True)
    except Exception:
        teacher_backbone = timm.create_model("vit_small_patch16_224", pretrained=True)

    teacher_backbone.head = nn.Identity()
    teacher_head = DINOProjectionHead(input_dim, 2048, 256, 2048, batch_norm=False)

    deactivate_requires_grad(teacher_backbone)
    deactivate_requires_grad(teacher_head)

    student_backbone = student_backbone.to(DEVICE)
    student_head = student_head.to(DEVICE)
    teacher_backbone = teacher_backbone.to(DEVICE)
    teacher_head = teacher_head.to(DEVICE)

    collate_fn = DINOCollateFunction(global_crop_size=224,local_crop_size=96)
    dataset = LightlyDataset(input_dir=IMAGE_FOLDER)
    dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,collate_fn=collate_fn,shuffle=True,drop_last=True,num_workers=4)
    criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)
    optimizer = torch.optim.AdamW(
        list(student_backbone.parameters()) + list(student_head.parameters()),
        lr=0.0005,
        weight_decay=1e-4,
    )

    start_epoch = 0
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/dino_epoch_*.pth"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Resuming from checkpoint: {latest}")
        checkpoint = torch.load(latest)
        student_backbone.load_state_dict(checkpoint["student_backbone"])
        student_head.load_state_dict(checkpoint["student_head"])
        teacher_backbone.load_state_dict(checkpoint["teacher_backbone"])
        teacher_head.load_state_dict(checkpoint["teacher_head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    print(
        f"Training on {len(dataset)} images for {EPOCHS - start_epoch} additional epochs..."
    )

    #Training
    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0
        momentum_val = cosine_schedule(epoch, EPOCHS, 0.996, 1)

        for (views, _, filenames) in dataloader:
            global_views = [view.to(DEVICE) for view in views[:2]]
            local_views = [view.to(DEVICE) for view in views[2:]]

            update_momentum(student_backbone, teacher_backbone, m=momentum_val)
            update_momentum(student_head, teacher_head, m=momentum_val)

            student_features = [
                student_head(student_backbone(view))
                for view in global_views + local_views
            ]
            teacher_features = [
                teacher_head(teacher_backbone(view)) for view in global_views
            ]

            loss = criterion(student_features, teacher_features, epoch=epoch)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

        #Saving checkpoints
        if (epoch + 1) % SAVE_EVERY == 0:
            save_path = f"{CHECKPOINT_DIR}/dino_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "student_backbone": student_backbone.state_dict(),
                    "student_head": student_head.state_dict(),
                    "teacher_backbone": teacher_backbone.state_dict(),
                    "teacher_head": teacher_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            print(f"Saved checkpoint: {save_path}")

    
    torch.save(student_backbone.state_dict(), "final_dino_model.pth")
    print("Training complete. Saved 'final_dino_model.pth'.")

if __name__ == "__main__":
    train_dino()
