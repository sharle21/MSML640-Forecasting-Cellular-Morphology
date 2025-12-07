import pandas as pd
import subprocess
import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Project configs
CSV_FILE = 'paired_dataset_weeks.csv'
# scratch path for msml640
OUT_DIR = '/scratch/zt1/project/msml640/user/sharle/data/images' 
WORKERS = 16 

# Sanity check before starting
assert os.path.exists(os.path.dirname(OUT_DIR)), f"Base path doesn't exist: {OUT_DIR}"

df = pd.read_csv(CSV_FILE)

def get_local_path(s3_url):
    """Maps weird S3 dates to clean local folders"""
    if '2020_12_02' in s3_url: batch = '2Weeks'
    elif '2020_12_07' in s3_url: batch = '4Weeks'
    else: return None

    parts = s3_url.split('/')
    # If the parent folder is 'Images', go up one more level for the plate name
    plate = parts[-3] if parts[-2] == 'Images' else parts[-2]
    return os.path.join(OUT_DIR, batch, plate, parts[-1])

def download_row(row):
    # Columns are stringified lists "['s3://...', ...]", clean them up
    cols = ['image_paths_2W', 'image_paths_4W']
    urls = []
    
    for c in cols:
        # Quick and dirty string cleanup
        clean = row[c].replace('[','').replace(']','').replace("'", "")
        items = [x.strip() for x in clean.split(',')]
        # We only need the first 6 channels
        urls.extend(items[:6])

    for url in urls:
        if not url: continue
        
        dest = get_local_path(url)
        if not dest: continue

        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            # using aws cli is faster than boto3 for simple cp
            cmd = f"aws s3 cp {url} {dest} --no-sign-request"
            subprocess.run(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    print(f"Downloading {len(df)} rows to {OUT_DIR} with {WORKERS} threads...")
    
    # iterrows is slow but fine for just 16 threads
    rows = [r for _, r in df.iterrows()]
    
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(tqdm(pool.map(download_row, rows), total=len(rows)))
