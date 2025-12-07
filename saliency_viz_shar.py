import os
import torch
import torch.nn as nn
import tifffile as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from torchvision import transforms
from timm.models import create_model
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from train_trajectory import TrajectoryPredictor

# --- CONFIG ---
CSV_FILE = 'paired_dataset.csv'
MODEL_PTH = "trajectory_model.pth"
DATA_ROOT = '/scratch/zt1/project/msml640/user/sharle/data/images'
OP_WEIGHTS = '/home/sharle/.cache/huggingface/hub/models--recursionpharma--OpenPhenom/snapshots/645dc0eb8a947ebc5963d3dd076cbb73c2fc9931/model.safetensors'

HIDDEN_DIM = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Standard ImageNet transform
tfm = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(256),
])

def get_backbone():
    print(f"Loading backbone from {OP_WEIGHTS}...")
    m = create_model('vit_small_patch16_224', pretrained=False, num_classes=0, in_chans=6, img_size=256)
    
    if not os.path.exists(OP_WEIGHTS):
        print(f"Warning: Weights not found at {OP_WEIGHTS}")
        return m.to(device).eval()

    sd = torch.load(OP_WEIGHTS, map_location='cpu')
    sd = sd.get('state_dict', sd)
    
    # Fix key names
    new_sd = {k.replace("encoder.vit_backbone.", "").replace("fc_norm", "norm"): v 
              for k, v in sd.items() if "encoder.vit_backbone." in k}
    
    # Expand input channels 1->6
    if 'patch_embed.proj.weight' in new_sd:
        w = new_sd['patch_embed.proj.weight']
        if w.shape[1] == 1:
            new_sd['patch_embed.proj.weight'] = w.repeat(1, 6, 1, 1)

    # Fix pos_embed size (257 tokens -> 256 size)
    if 'pos_embed' in new_sd:
        pe = new_sd['pos_embed']
        if pe.shape[1] != m.pos_embed.shape[1]:
            cls, patch = pe[:, :1], pe[:, 1:]
            patch = torch.nn.functional.interpolate(patch.transpose(1, 2), size=256, mode='linear')
            new_sd['pos_embed'] = torch.cat((cls, patch.transpose(1, 2)), dim=1)

    m.load_state_dict(new_sd, strict=False)
    return m.to(device).eval()

def get_local_path(s3_url):
    if not s3_url: return None
    if '2020_11_18' in s3_url: batch = '24h_day1'
    elif '2020_11_19' in s3_url: batch = '72h_day4'
    else: return None

    parts = s3_url.split('/')
    plate = parts[-3] if parts[-2] == 'Images' else parts[-2]
    return os.path.join(DATA_ROOT, batch, plate, parts[-1])

def process_single_row(row):
    try:
        urls = ast.literal_eval(row['image_paths_24h'])
    except:
        return None, None

    stack = []
    raw_img = None

    for i in range(6):
        # Default to empty channel if missing
        t = torch.zeros((1, 256, 256))
        
        if i < len(urls):
            path = get_local_path(urls[i])
            if path and os.path.exists(path):
                img = tf.imread(path).astype(np.float32)
                if i == 0: raw_img = img # Keep DNA channel for viz
                
                # Normalize & Transform
                t = torch.from_numpy(img / (img.max() + 1e-6)).unsqueeze(0)
                t = tfm(t)
        
        stack.append(t)

    if raw_img is None: return None, None
    
    return torch.cat(stack, dim=0).unsqueeze(0).to(device), raw_img

def main():
    df = pd.read_csv(CSV_FILE)
    
    # Init models
    backbone = get_backbone()
    
    print(f"Loading trajectory head: {MODEL_PTH}")
    head = TrajectoryPredictor(hidden_dim=HIDDEN_DIM)
    head.load_state_dict(torch.load(MODEL_PTH, map_location=device))
    head.to(device).eval()

    # Find first valid sample
    tensor_in, raw_img = None, None
    for _, row in df.iterrows():
        tensor_in, raw_img = process_single_row(row)
        if tensor_in is not None:
            print(f"Visualizing perturbation: {row['perturbation_id']}")
            break
    
    if tensor_in is None:
        raise RuntimeError("No local images found to visualize.")

    # Gradient pass
    tensor_in.requires_grad_()
    
    emb = backbone(tensor_in)
    pred = head(emb)
    
    # We want to know which pixels caused the *magnitude* of the movement
    score = torch.norm(pred)
    
    backbone.zero_grad()
    head.zero_grad()
    score.backward()
    
    # Saliency calculation
    grads = tensor_in.grad.data.cpu().squeeze().numpy()
    # Max over channels to find "important" pixels regardless of stain
    saliency = np.max(np.abs(grads), axis=0)
    
    # Smooth for better visualization
    saliency = gaussian_filter(saliency, sigma=2)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Raw DNA
    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title("Input (DNA Channel)")
    axes[0].axis('off')
    
    # 2. Saliency Heatmap
    axes[1].imshow(saliency, cmap='inferno')
    axes[1].set_title("Saliency (Gradient Norm)")
    axes[1].axis('off')

    # 3. Overlay
    # resize raw to match saliency shape (256x256) after transform
    raw_resized = resize(raw_img, saliency.shape)
    axes[2].imshow(raw_resized, cmap='gray')
    axes[2].imshow(saliency, cmap='inferno', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('saliency_map.png', dpi=300)
    print("Saved saliency_map.png")

if __name__ == "__main__":
    main()
