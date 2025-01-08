from sklearn.cluster import SpectralClustering
import umap
import numpy as np

def reduce_dimensions(embeddings, n_components=2):
    """Apply UMAP for dimensionality reduction."""
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit_transform(embeddings)

def perform_clustering(reduced_data, n_clusters=5):
    """Cluster reduced data using Spectral Clustering."""
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
    return clustering.fit_predict(reduced_data)
