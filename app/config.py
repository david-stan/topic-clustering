import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "../data/dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/bert/")
RESULTS_PATH = os.path.join(BASE_DIR, "../results/clusters.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "../results/embeddings.npy")
CLUSTER_MODEL_PATH = os.path.join(BASE_DIR, "../results/hdbscan_model.pkl")

# Clustering parameters
N_CLUSTERS = 5
REDUCE_DIMENSIONS = 2
