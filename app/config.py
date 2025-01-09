import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "../data/dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/bert/")
RESULTS_PATH = os.path.join(BASE_DIR, "../results/clusters.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "../results/embeddings.npy")

# Clustering parameters
N_CLUSTERS = 5
REDUCE_DIMENSIONS = 2
