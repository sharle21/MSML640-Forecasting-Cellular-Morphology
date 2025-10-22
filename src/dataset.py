import numpy as np
import pandas as pd

def load_embeddings(path_early, path_late):
    emb_early = np.load(path_early)
    emb_late = np.load(path_late)
    assert emb_early.shape == emb_late.shape, "Embeddings must align by row"
    return emb_early, emb_late
