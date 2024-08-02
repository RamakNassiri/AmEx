import numpy as np
import networkx as nx
import pandas as pd

def load_attention_matrices(filename):
    # Load the file as a dictionary
    return np.load(filename, allow_pickle=True).item()

def compute_pagerank_for_head(matrix):
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    return nx.pagerank(G, alpha=0.9)

def compute_and_save_pagerank(attention_matrices):
    for serotype, matrices in attention_matrices.items():
        for head_index, matrix in enumerate(matrices):
            if matrix.ndim > 2:
                matrix = matrix.reshape(-1, matrix.shape[-1])  
            pagerank_scores = compute_pagerank_for_head(matrix)
            
            df = pd.DataFrame.from_dict(pagerank_scores, orient='index', columns=['PageRank'])
            df.to_csv(f'pagerank_scores_serotype500_{serotype}_head_{head_index}.csv')
            print(f"PageRank scores for serotype500 {serotype}, head {head_index} saved.")

def main():
    attention_matrices = load_attention_matrices('averaged_attention_matrices1.npy')
    compute_and_save_pagerank(attention_matrices)
    print("All PageRank scores have been calculated and saved.")

if __name__ == "__main__":
    main()
