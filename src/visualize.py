import umap
import matplotlib.pyplot as plt

def plot_umap(true_emb, pred_emb, title="Predicted vs True Trajectories"):
    reducer = umap.UMAP(n_components=2)
    proj_true = reducer.fit_transform(true_emb)
    proj_pred = reducer.transform(pred_emb)
    plt.scatter(proj_true[:, 0], proj_true[:, 1], s=5, alpha=0.6, label="True")
    plt.scatter(proj_pred[:, 0], proj_pred[:, 1], s=5, alpha=0.6, label="Predicted")
    plt.legend()
    plt.title(title)
    plt.show()
