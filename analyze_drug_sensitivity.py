import torch
import pandas as pd
import os

# Config
INPUT_PT = 'openphenom_embeddings_all_timepoints.pt'
OUT_CSV = 'drug_sensitivity_ranking.csv'

def main():
    if not os.path.exists(INPUT_PT):
        raise FileNotFoundError(f"Missing {INPUT_PT}")

    print(f"Loading {INPUT_PT}...")
    data = torch.load(INPUT_PT)
    
    #Convert list of dicts to straight Tensors for vectorized calc
    pids = [d['perturbation_id'] for d in data]
    
    #Stack embeddings into matrices (N_samples, Embedding_Dim)
    e24 = torch.stack([d['e24'].float() for d in data])
    e72 = torch.stack([d['e72'].float() for d in data])
    e4w = torch.stack([d['e4w'].float() for d in data])

    #acute Effect: Euclidean distance between 24h and 72h
    # dim=1 calculates norm across the embedding dimension
    acute_mag = torch.norm(e72 - e24, dim=1)

    #long Term Effect: Euclidean distance between 24h and 4Weeks
    long_mag = torch.norm(e4w - e24, dim=1)

    #reversibility Metric
    #Positive = Cell moved away then came back (Acute > Long)
    #Negative = Cell kept drifting away (Long > Acute)
    reversibility = acute_mag - long_mag

    #Build DataFrame
    df = pd.DataFrame({
        'perturbation_id': pids,
        'acute_72h': acute_mag.numpy(),
        'long_term_4w': long_mag.numpy(),
        'reversibility': reversibility.numpy()
    })

    print(f"\nAnalyzed {len(df)} perturbations.\n")

    #Quick Summary stats
    print("--- Top 10 Most Potent (72h Change) ---")
    print(df.nlargest(10, 'acute_72h')[['perturbation_id', 'acute_72h']].to_string(index=False))

    print("\n--- Top 10 High Recovery (Reverted to Normal) ---")
    print(df.nlargest(10, 'reversibility')[['perturbation_id', 'reversibility']].to_string(index=False))

    print("\n--- Top 10 Progressive/Toxic (Worsened over time) ---")
    print(df.nsmallest(10, 'reversibility')[['perturbation_id', 'reversibility']].to_string(index=False))

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")

if __name__ == "__main__":
    main()
