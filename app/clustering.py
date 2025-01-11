import numpy as np
import hdbscan
import umap.umap_ as umap



def generate_umap(message_embeddings, n_components, n_neighbors, random_state=None):
    """Generate clusters from embeddings.
    Args:
        message_embeddings (np.ndarray): Array of message embeddings.
        n_neighbors (int): Number of neighbors for UMAP algorithm.
        n_components (int): Number of dimensions for UMAP algorithm.
    """
    umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=random_state).fit_transform(message_embeddings)
    return umap_embeddings

def generate_clusters(embeddings, min_cluster_size, min_samples=None):
    """Generate clusters from embeddings.
    
    Args:
        embeddings (np.ndarray): Array of umap embeddings.
        min_cluster_size (int): Minimum number of samples in a cluster for HDBSCAN algorithm.
        min_samples (int, optional): Minimum number of samples for HDBSCAN algorithm.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        hdbscan.HDBSCAN: Clustering model.
    """
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', min_samples=min_samples, gen_min_span_tree=True).fit(embeddings)
    return cluster

def score_clusters(cluster, prob_threshold=0.05):
    """Score the clusters based on their labels and probabilities.

    Args:
        clusters (hdbscan.HDBSCAN): Clustering model.
        prob_threshold (float, optional): Probability threshold for considering a sample as an outlier.
            Default is 0.05.

    Returns:
        int: Number of unique cluster labels.
        float: Cost score representing the fraction of samples with probabilities below the threshold.
"""
    cluster_labels = cluster.labels_
    
    label_count = len(np.unique(cluster_labels))
    total_num = len(cluster_labels)

    cost = np.count_nonzero(cluster.probabilities_ < prob_threshold) / total_num

    return label_count, cost