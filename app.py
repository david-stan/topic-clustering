from app.preprocess import load_data, save_embeddings, load_saved_embeddings
from app.hyperparam_opt import bayesian_optimization, hspace
from app.model import generate_embeddings
from app.utils import label_clusters_with_topics
from app.config import N_CLUSTERS, REDUCE_DIMENSIONS, EMBEDDINGS_PATH

import pandas as pd

def main():
    # Step 1: Load Data
    print("Loading dataset...")
    sentences = load_data()
    
    # Step 2: Generate Embeddings
    print("Generating embeddings...")
    # embeddings = generate_embeddings(sentences)
    # save_embeddings(embeddings, EMBEDDINGS_PATH)
    embeddings = load_saved_embeddings(EMBEDDINGS_PATH)
    
    # Step 3: Perform Clustering
    print("Clustering data...")
    label_lower = 10
    label_upper = 100
    max_evals = 25
    best_params_use, best_clusters_use, trials_use = bayesian_optimization(embeddings, hspace, label_lower, label_upper, max_evals)

    for index, cluster in enumerate(best_clusters_use.labels_):
        if cluster == 2:
            print(f"Cluster {cluster}: {sentences[index]}")

    # Step 5: Save Results
    print("Saving results...")
    # Generate cluster labels
    clustered_data = pd.DataFrame(data = list(zip(best_clusters_use.labels_, sentences)), columns = ['cluster', 'text'])
    cluster_labels = label_clusters_with_topics(clustered_data)

    # Display cluster labels
    for cluster_id, label in cluster_labels.items():
        print(f"Cluster {cluster_id}: {label}")
    
    print("Clustering completed. Results saved.")

if __name__ == "__main__":
    main()
