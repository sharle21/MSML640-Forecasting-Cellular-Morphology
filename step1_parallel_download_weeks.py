import pandas as pd
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import ast

CSV_FILE = 'paired_dataset_weeks.csv'
SCRATCH_BASE = '/scratch/zt1/project/msml640/user/sharle/data/images' 
WORKERS = 4 

def download_row(row):
    # Setup unsigned client per-process
    # Boto3 handles retries internally, no need for a manual loop
    cfg = Config(signature_version=UNSIGNED, connect_timeout=15, retries={'max_attempts': 3})
    s3 = boto3.client('s3', config=cfg)
    
    downloaded = 0
    
    # Grab both columns
    urls = []
    for col in ['image_paths_2W', 'image_paths_4W']:
        if pd.isna(row.get(col)): continue
        try:
            # properly parse the stringified list "[...]"
            items = ast.literal_eval(row[col])
            urls.extend(items[:6]) # Take first 6
        except:
            continue

    for url in urls:
        if not url: continue
        
        # Path logic
        if '2020_12_02' in url: batch = '2Weeks'
        elif '2020_12_07' in url: batch = '4Weeks'
        else: continue

        parts = url.replace("s3://", "").split('/')
        bucket, key = parts[0], "/".join(parts[1:])
        
        # Handle the plate/images folder weirdness
        plate = parts[-3] if parts[-2] == 'Images' else parts[-2]
        local_path = os.path.join(SCRATCH_BASE, batch, plate, parts[-1])

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            s3.download_file(bucket, key, local_path)
            downloaded += 1
        except Exception:
            # If it fails after boto3 retries, just move on.
            pass

    return downloaded

if __name__ == "__main__":
    assert os.path.exists(os.path.dirname(SCRATCH_BASE)), "Base path missing!"
    
    df = pd.read_csv(CSV_FILE)
    print(f"Queueing {len(df)} rows on {WORKERS} cores...")

    # Convert to list of dicts to strip Pandas overhead before pickling
    rows = df.to_dict('records')

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        # chunksize helps when tasks are small
        list(tqdm(pool.map(download_row, rows, chunksize=5), total=len(rows)))
