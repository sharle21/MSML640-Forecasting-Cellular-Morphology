import os
import torch
import pandas as pd
import tifffile as tf
import numpy as np
import ast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.models import create_model
from tqdm import tqdm

# Config
CSV_FILE = 'paired_dataset.csv'
OUT_FILE = 'openphenom_embeddings.pt'
DATA_ROOT = '/scratch/zt1/project/msml640/user/sharle/data/images'
MODEL_PATH = '/home/sharle/.cache/huggingface/hub/models--recursionpharma--OpenPhenom/snapshots/645dc0eb8a947ebc5963d3dd076cbb73c2fc9931/model.safetensors'
BATCH_SIZE = 32
NUM_WORKERS = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PairedCellDataset(Dataset):
    def __init__(self, csv_path, root_dir):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(256),
        ])

    def __len__(self):
        return len(self.df)

    def _get_local_path(self, s3_url):
        if not s3_url: return None
        # Quick map for batch folders
        if '2020_11_18' in s3_url: batch = '24h_day1'
        elif '2020_11_19' in s3_url: batch = '72h_day4'
        else: return None
        
        parts = s3_url.split('/')
        plate = parts[-3] if parts[-2] == 'Images' else parts[-2]
        return os.path.join(self.root, batch, plate, parts[-1])

    def _load_stack(self, url_list_str):
        # fast parse "['s3://...', ...]"
        try:
            urls = ast.literal_eval(str(url_list_str))
        except:
            urls = []
            
        stack = []
        # Expecting 6 channels (indices 0-5)
        for i in range(6):
            path = self._get_local_path(urls[i]) if i < len(urls) else None
            
            if path and os.path.exists(path):
                img = tf.imread(path).astype(np.float32)
                # Normalize 0-1
                img = img / (img.max() + 1e-6)
                stack.append(torch.from_numpy(img).unsqueeze(0))
            else:
                # Padding for missing channels
                stack.append(torch.zeros((1, 256, 256), dtype=torch.float32))
        
        # Stack channels: (6, 256, 256)
        return self.transform(torch.cat(stack, dim=0))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img24 = self._load_stack(row['image_paths_24h'])
        img72 = self._load_stack(row['image_paths_72h'])
        return row['perturbation_id'], img24, img72

def get_model():
    # Force load OpenPhenom weights into timm ViT
    print(f"Loading weights from {MODEL_PATH}...")
    m = create_model('vit_small_patch16_224', pretrained=False, num_classes=0, in_chans=6, img_size=256)
    
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    
    # Remap keys from OpenPhenom specific names to timm standard
    new_sd = {k.replace("encoder.vit_backbone.", "").replace("fc_norm", "norm"): v 
              for k, v in sd.items() if "encoder.vit_backbone." in k}
    
    # Expand 1-channel weights to 6-channel
    if 'patch_embed.proj.weight' in new_sd:
        w = new_sd['patch_embed.proj.weight']
        if w.shape[1] == 1:
            new_sd['patch_embed.proj.weight'] = w.repeat(1, 6, 1, 1)

    # Interpolate Pos Embeds (224 -> 256)
    if 'pos_embed' in new_sd:
        pe = new_sd['pos_embed']
	print(pe.shape[1])
	print(m.pos_embed.shape[1])
        if pe.shape[1] != m.pos_embed.shape[1]:
            # (1, 197, 384) -> separate cls token -> grid -> resize -> flatten -> recombine
            cls_token = pe[:, :1]
            grid = pe[:, 1:].transpose(1, 2)
            # 14x14 grid to 16x16 grid
            grid = torch.nn.functional.interpolate(grid, size=256, mode='linear') #fltetn
            new_sd['pos_embed'] = torch.cat((cls_token, grid.transpose(1, 2)), dim=1)

    m.load_state_dict(new_sd, strict=False)
    return m.to(device).eval()

if __name__ == '__main__':
    dataset = PairedCellDataset(CSV_FILE, DATA_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = get_model()
    results = []

    print(f"Processing {len(dataset)} items in batches of {BATCH_SIZE}...")
    
    with torch.no_grad():
        for pids, imgs24, imgs72 in tqdm(loader):
            imgs24 = imgs24.to(device)
            imgs72 = imgs72.to(device)
            
            # Batch inference
            emb24 = model(imgs24)
            emb72 = model(imgs72)
            
            # Move back to CPU to save RAM
            emb24 = emb24.cpu()
            emb72 = emb72.cpu()
            
            for i in range(len(pids)):
                results.append({
                    'perturbation_id': pids[i],
                    'embedding_24h': emb24[i],
                    'embedding_72h': emb72[i]
                })

    torch.save(results, OUT_FILE)
    print("Done.")
