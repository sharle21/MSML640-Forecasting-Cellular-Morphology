import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from lightly.data import LightlyDataset, DINOCollateFunction
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
import timm
import os
import glob
import copy

#Configuration
IMAGE_FOLDERS = [
    "/scratch/zt1/project/msml640/user/sharle/data/images/24h_day1",
    "/scratch/zt1/project/msml640/user/sharle/data/images/72h_day4",
    "/scratch/zt1/project/msml640/user/sharle/data/images/2Weeks",
    "/scratch/zt1/project/msml640/user/sharle/data/images/4Weeks",
]

BATCH_SIZE = 32
EPOCHS = 100
SAVE_EVERY = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "dino_multistep_checkpoints"


def train_dino():
    print(f"Starting multi-timepoint DINO training on {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    backbone_name = "vit_small_patch14_dinov2.lvd142m"
    print(f"Loading backbone: {backbone_name}")

    try:
        backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            img_size=224,
            dynamic_img_size=True,
        )
    except Exception:
        print("DINOv2 load failed, falling back to ViT-Small")
        backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            dynamic_img_size=True,
        )

    backbone.head = nn.Identity()

    input_dim = 384
    head = DINOProjectionHead(input_dim, 2048, 256, 2048, batch_norm=False)

    student_backbone = backbone
    student_head = head

    teacher_backbone = copy.deepcopy(backbone)
    teacher_head = DINOProjectionHead(input_dim, 2048, 256, 2048, batch_norm=False)

    deactivate_requires_grad(teacher_backbone)
    deactivate_requires_grad(teacher_head)

    student_backbone = student_backbone.to(DEVICE)
    student_head = student_head.to(DEVICE)
    teacher_backbone = teacher_backbone.to(DEVICE)
    teacher_head = teacher_head.to(DEVICE)
    collate_fn = DINOCollateFunction(global_crop_size=224, local_crop_size=98)

    datasets = []
    print("Scanning datasets...")
    for folder in IMAGE_FOLDERS:
        if os.path.exists(folder):
            print(f"  Found folder: {folder}")
            ds = LightlyDataset(input_dir=folder)
            print(f"    Loaded {len(ds)} images.")
            datasets.append(ds)
        else:
            print(f"  Warning: folder not found, skipping: {folder}")

    if not datasets:
        print("No data found. Check image paths.")
        return

    combined_dataset = ConcatDataset(datasets)
    print(f"Total training images: {len(combined_dataset)}")

    dataloader = DataLoader(combined_dataset,batch_size=BATCH_SIZE,collate_fn=collate_fn,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)
    criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.AdamW(list(student_backbone.parameters()) + list(student_head.parameters()),lr=0.0005,weight_decay=1e-4)

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

    print(f"Starting loop for {EPOCHS - start_epoch} epochs...")

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0
        momentum_val = cosine_schedule(epoch, EPOCHS, 0.996, 1)

        for i, (views, _, _) in enumerate(dataloader):
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

            if i % 50 == 0:
                print(
                    f"  Epoch {epoch + 1} | "
                    f"Batch {i}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

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

    torch.save(student_backbone.state_dict(), "final_multistep_dino.pth")
    print("Training complete.")

if __name__ == "__main__":
    train_dino()
