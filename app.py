from app.preprocess import load_data, save_embeddings, load_saved_embeddings
from app.hyperparam_opt import bayesian_optimization
from app.model import generate_embeddings
from app.utils import generate_hierarchical_labels, save_hdbscan_model
from app.config import N_CLUSTERS, REDUCE_DIMENSIONS, EMBEDDINGS_PATH, CLUSTER_MODEL_PATH

import pandas as pd

def main():
    # Step 1: Load Data
    print("Loading dataset...")
    sentences = load_data()
    
    # Step 2: Generate Embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(sentences)
    save_embeddings(embeddings, EMBEDDINGS_PATH)
    # embeddings = load_saved_embeddings(EMBEDDINGS_PATH)
    
    # Step 3: Perform Clustering with Hyperparameter Optimization
    print("Clustering data...")
    best_params_use, best_clusters_use, trials_use = bayesian_optimization(embeddings)

    save_hdbscan_model(best_clusters_use, CLUSTER_MODEL_PATH)

    for index, cluster in enumerate(best_clusters_use.labels_):
        if cluster == 2:
            print(f"Cluster {cluster}: {sentences[index]}")

    # Step 5: Save Results
    print("Saving results...")
    # Generate cluster labels
    clustered_data = pd.DataFrame(data = list(zip(best_clusters_use.labels_, sentences)), columns = ['cluster', 'text'])
    cluster_labels = generate_hierarchical_labels(clustered_data)

    # Display cluster labels
    for main_category, topics in cluster_labels.items():
        print(f"Cluster {main_category}: {topics}")
    
    print("Clustering completed. Results saved.")

if __name__ == "__main__":
    main()
