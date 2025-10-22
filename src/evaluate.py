import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

def evaluate(emb_true, emb_pred):
    mse = mean_squared_error(emb_true, emb_pred)
    cos_sim = np.mean(np.diag(cosine_similarity(emb_true, emb_pred)))
    return {"MSE": mse, "CosineSimilarity": cos_sim}
