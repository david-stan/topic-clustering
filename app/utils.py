import json
from app.config import RESULTS_PATH

def save_results(clusters, reduced_data):
    """Save clustering results."""
    data = {
        "clusters": clusters.tolist(),
        "reduced_data": reduced_data.tolist()
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f)

def load_results():
    """Load clustering results."""
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# TF-IDF and LDA for topic modeling
def extract_topics(cluster_texts, n_topics=1, n_top_words=5):
    """
    Extract topics from cluster texts using TF-IDF and LDA.
    """
    # Vectorize the text using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_texts)

    # Apply Latent Dirichlet Allocation (LDA)
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(tfidf_matrix)

    # Get feature names and extract top words for each topic
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(", ".join(top_words))

    return topics

# Generate cluster labels
def label_clusters_with_topics(clustered_data):
    """
    Generate cluster labels by extracting topics from clustered words.
    """
    cluster_labels = {}

    for cluster_id in clustered_data["cluster"].unique():
        # Get all text from the current cluster
        cluster_texts = clustered_data[clustered_data["cluster"] == cluster_id]["text"].values

        # Extract topics
        topics = extract_topics(cluster_texts, n_topics=1)
        cluster_labels[cluster_id] = topics[0]

    return cluster_labels