import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os

#config
pt_file = 'openphenom_embeddings_all_timepoints.pt'
out_file = 'counterfactual_analysis.png'
control_lbl = 'DMSO' #set to None to use global mean

def main():
    if not os.path.exists(pt_file):
        raise FileNotFoundError(pt_file)

    #load data directly into dataframe for easier grouping
    print(f"loading {pt_file}...")
    raw = torch.load(pt_file)
    
    #stack tensors for speed
    e24 = np.stack([r['e24'].numpy() for r in raw])
    e4w = np.stack([r['e4w'].numpy() for r in raw])
    pids = [r['perturbation_id'] for r in raw]

    #create dataframe to manage labels
    df = pd.DataFrame(e24)
    df['label'] = pids
    
    #calculate mean vectors per drug
    #groupby is much faster than looping
    means_24 = df.groupby('label').mean()
    means_4w = pd.DataFrame(e4w).groupby(df['label']).mean()
    
    #get control baseline
    if control_lbl and control_lbl in means_24.index:
        mu_control = means_24.loc[control_lbl].values
    else:
        #fallback to global mean
        mu_control = e24.mean(axis=0)

    #calculate treatment vectors (delta = result - baseline_control)
    #vectors is a dataframe where index=drug, values=vector
    vectors = means_4w.sub(mu_control, axis=1)
    
    #find strongest drug to visualize
    #calculate L2 norm for every row
    mags = np.linalg.norm(vectors.values, axis=1)
    target_idx = np.argmax(mags)
    target_drug = vectors.index[target_idx]
    target_vec = vectors.iloc[target_idx].values

    print(f"simulating effect of: {target_drug}")

    #generate counterfactuals
    #adding target vector to ALL baseline cells
    e_counterfactual = e24 + target_vec

    #pca projection
    combined = np.vstack([e24, e4w, e_counterfactual])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)

    #split back out
    n = len(e24)
    p_base = proj[:n]
    p_real = proj[n:2*n]
    p_cf = proj[2*n:]

    plt.figure(figsize=(8,8))
    
    #masking for cells that actually took the drug
    is_target = np.array(pids) == target_drug

    #plot background noise (baseline)
    plt.scatter(p_base[:,0], p_base[:,1], c='gray', s=5, alpha=0.2, label='baseline')

    #plot real outcome of target drug
    plt.scatter(p_real[is_target,0], p_real[is_target,1], c='blue', s=30, label=f'true {target_drug}')

    #plot simulation (what if everyone took it?)
    subset = np.random.choice(np.where(~is_target)[0], 500, replace=False)
    plt.scatter(p_cf[subset,0], p_cf[subset,1], c='red', s=5, alpha=0.3, label='simulated')

    #showing how baseline points move to simulated points
    for i in subset[:15]:
        plt.arrow(p_base[i,0], p_base[i,1], 
                  p_cf[i,0]-p_base[i,0], p_cf[i,1]-p_base[i,1],
                  color='red', alpha=0.4, width=0.02)

    plt.legend()
    plt.title(f"counterfactual shift: {target_drug}")
    plt.savefig(out_file)
    print(f"saved {out_file}")

    #sanity check
    sim_t = e_counterfactual[is_target]
    real_t = e4w[is_target]
    
    #calculating similarity
    cos = cosine_similarity(sim_t, real_t)
    print(f"mean cosine similarity on target group: {np.diag(cos).mean():.4f}")

if __name__ == "__main__":
    main()
