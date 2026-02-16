# MSML640-Forecasting-Cellular-Morphology
Forecasting Cellular Morphological Trajectories via Latent Dynamics


This repository contains the complete implementation of a computer vision pipeline designed to forecast the long-term phenotypic trajectory (4 weeks) of cell populations based on early-stage (24h) microscopy images.

Project Overview

Drug discovery is bottlenecked by long experimental timelines. We treat cellular morphology as a deterministic trajectory in a latent semantic space. By leveraging OpenPhenom (Vision Transformer) for feature extraction and a custom Multi-Horizon MLP for forecasting, we achieve >99% accuracy in predicting future cell states from a single snapshot.

Setup & Requirements

1. Environment

All scripts are designed to run in a Python 3.10+ environment with PyTorch.

pip install torch torchvision timm pandas numpy scikit-learn matplotlib scipy tqdm lightly


2. File Structure

Ensure your directory looks like this before running scripts:

.
├── data/                   # Raw images (downloaded)
├── step1_*.py              # Download scripts
├── step2_*.py              # Feature extraction scripts
├── train_*.py              # Training scripts
├── visualize_*.py          # Plotting scripts
└── README.md


Execution Pipeline (Run in Order)

Phase 1: Data Engineering (ETL)

These steps download raw JUMP-CP data (TB scale) and convert them into lightweight embeddings.

1. Download Raw Images
Fetches 6-channel microscopy TIFFs from AWS S3.

python step1_parallel_download.py        # Downloads 24h & 72h data
python step1_parallel_download_weeks.py  # Downloads 2 Week & 4 Week data


2. Extract Features (OpenPhenom)
Processes raw images into 384-dimensional embeddings using a pre-trained Vision Transformer.

python step2_extract.py        # Process 24h -> 72h
python step2_extract_weeks.py  # Process 2w -> 4w


3. Merge Datasets
Aligns the timepoints by Perturbation ID to create full trajectories.

python merge_embeddings.py
# Output: openphenom_embeddings_all_timepoints.pt (The Master Dataset)


Phase 2: Model Training

Train the Multi-Horizon Forecaster
Trains the MLP to predict 72h, 2w, and 4w states simultaneously.

python train_trajectory_multistep.py
# Output: trajectory_model_multistep.pth


(Optional) Train Self-Supervised Backbone (DINOv2)
To validate feature learning from scratch:

python train_dino_multistep.py


Phase 3: Evaluation & Visualization

1. Quantitative Evaluation (The Scorecard)
Calculates MSE and Cosine Similarity vs. Static Baseline.

python evaluate_metrics.py


2. Trajectory Visualization (The "Spaghetti Plot")
Generates PCA plots showing the predicted evolution of cell populations.

python visualize_multistep.py
# Output: trajectory_multistep.png


3. Interpretability (Saliency Maps)
Generates heatmaps showing model attention on biological structures.

python saliency_vis.py
# Output: saliency_map.png


Phase 4: Advanced Analysis (Bonus Tasks)

Generative Counterfactuals
Simulates "What-If" scenarios using Latent Vector Arithmetic.

python counterfactual_analysis.py
# Output: counterfactual_analysis.png


Drug Sensitivity Ranking
Identifies the most potent and toxic gene knockouts (e.g., GRIN2A).

python analyze_drug_sensitivity.py
# Output: drug_sensitivity_ranking.csv


Robustness Stress Test ("Data-in-the-wild")
Tests model accuracy under simulated sensor noise.

python robustness_test.py
# Output: robustness_analysis.png


Edge Deployment Optimization
Quantizes the model to INT8 for deployment on Raspberry Pi/Microscopes.

python optimize_model.py
# Output: trajectory_model_quantized.pth


The .pth files and .csv are not added due their size


References

Dataset: JUMP-CP (Cell Painting) Source 4.

Backbone: OpenPhenom (ViT-Small).

Method: Latent Dynamics Modeling via Multi-Step Regression.
