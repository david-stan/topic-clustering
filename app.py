from app.preprocess import load_data, generate_embeddings
from app.clustering import reduce_dimensions, perform_clustering
from app.utils import save_results
from app.config import N_CLUSTERS, REDUCE_DIMENSIONS

def main():
    # Step 1: Load Data
    print("Loading dataset...")
    texts = load_data()
    
    # Step 2: Generate Embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(texts)
    
    # Step 3: Dimensionality Reduction
    print("Reducing dimensions...")
    reduced_data = reduce_dimensions(embeddings, n_components=REDUCE_DIMENSIONS)
    
    # Step 4: Perform Clustering
    print("Clustering data...")
    clusters = perform_clustering(reduced_data, n_clusters=N_CLUSTERS)
    
    # Step 5: Save Results
    print("Saving results...")
    save_results(clusters, reduced_data)
    
    print("Clustering completed. Results saved.")

if __name__ == "__main__":
    main()
