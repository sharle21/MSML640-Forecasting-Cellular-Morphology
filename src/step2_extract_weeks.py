import os
import pandas as pd
import torch
import tifffile as tf
import numpy as np
import ast
from torchvision import transforms
from timm.models import create_model
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


CSV_FILE = 'paired_dataset_weeks.csv'
OUT_FILE = 'openphenom_embeddings_weeks.pt'
DATA_DIR = '/scratch/zt1/project/msml640/user/sharle/data/images'
MODEL_PATH = '/home/sharle/.cache/huggingface/hub/models--recursionpharma--OpenPhenom/snapshots/645dc0eb8a947ebc5963d3dd076cbb73c2fc9931/model.safetensors'

BATCH_SIZE = 64
NUM_WORKERS = 8 
SAVE = 50 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Standard ImageNet normalization/resize
tfm = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(256),
])

def get_model():
    # Helper to load OpenPhenom weights into standard Timm ViT
    print(f"Loading weights from {MODEL_PATH}...")
    m = create_model('vit_small_patch16_224', pretrained=False, num_classes=0, in_chans=6, img_size=256)
    
    sd = torch.load(MODEL_PATH, map_location='cpu')
    sd = sd.get('state_dict', sd)
    
    # remap keys
    new_sd = {}
    for k, v in sd.items():
        if "encoder.vit_backbone." in k:
            new_k = k.replace("encoder.vit_backbone.", "").replace("fc_norm", "norm")
            new_sd[new_k] = v
    
    # Fix input channels frm 1 to 6
    if 'patch_embed.proj.weight' in new_sd:
        w = new_sd['patch_embed.proj.weight']
        if w.shape[1] == 1:
            new_sd['patch_embed.proj.weight'] = w.repeat(1, 6, 1, 1)

    # Fix Pos Embeds reshaped to 256
    if 'pos_embed' in new_sd:
        pe = new_sd['pos_embed']
        if pe.shape[1] != m.pos_embed.shape[1]:
            cls, patch = pe[:, :1], pe[:, 1:]
            patch = torch.nn.functional.interpolate(patch.transpose(1, 2), size=256, mode='linear')
            new_sd['pos_embed'] = torch.cat((cls, patch.transpose(1, 2)), dim=1)

    m.load_state_dict(new_sd, strict=False)
    return m.to(device).eval()

class WeekDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def _get_path(self, url):
        if '2020_12_02' in url: batch = '2Weeks'
        elif '2020_12_07' in url: batch = '4Weeks'
        else: return None
        
        parts = url.split('/')
        plate = parts[-3] if parts[-2] == 'Images' else parts[-2]
        return os.path.join(DATA_DIR, batch, plate, parts[-1])

    def _load_stack(self, url_list):
        try:
            urls = ast.literal_eval(str(url_list))
        except:
            urls = []
            
        stack = []
        # 6 channels needed out of the 10
        for i in range(6):
            path = self._get_path(urls[i]) if i < len(urls) else None
            
            if path and os.path.exists(path):
                try:
                    img = tf.imread(path).astype(np.float32)
                    t = torch.from_numpy(img / (img.max() + 1e-6)).unsqueeze(0)
                    stack.append(tfm(t))
                except:
                    # File corrupted or read error
                    stack.append(torch.zeros((1, 256, 256)))
            else:
                # Missing channel
                stack.append(torch.zeros((1, 256, 256)))
                
        return torch.cat(stack, dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            row['perturbation_id'],
            self._load_stack(row['image_paths_2W']),
            self._load_stack(row['image_paths_4W'])
        )

if __name__ == "__main__":
    assert os.path.exists(CSV_FILE), f"Missing CSV: {CSV_FILE}"
    
    # Resume logic because losing gpu from mid is UGHly
    results = []
    done_ids = set()
    if os.path.exists(OUT_FILE):
        print(f"Resuming from {OUT_FILE}...")
        results = torch.load(OUT_FILE)
        done_ids = {r['perturbation_id'] for r in results}
        
    df = pd.read_csv(CSV_FILE)
    df = df[~df['perturbation_id'].isin(done_ids)]
    
    if len(df) == 0:
        print("All Done.")
        exit()

    print(f"To process: {len(df)}")
    
    ds = WeekDataset(df)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, 
                        shuffle=False, pin_memory=True, prefetch_factor=2)
    
    model = get_model()
    
    with torch.no_grad():
        for i, (pids, b2w, b4w) in enumerate(tqdm(loader)):
            b2w = b2w.to(device, non_blocking=True)
            b4w = b4w.to(device, non_blocking=True)
            
            e2w = model(b2w).cpu()
            e4w = model(b4w).cpu()
            
            for j, pid in enumerate(pids):
                results.append({
                    'perturbation_id': pid,
                    'embedding_2w': e2w[j],
                    'embedding_4w': e4w[j]
                })
            
            if i % SAVE == 0 and i > 0:
                torch.save(results, OUT_FILE)

    # Final save
    torch.save(results, OUT_FILE)
    print("Finished.")
