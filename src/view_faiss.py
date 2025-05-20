import faiss
import os
import numpy as np

# Path where the FAISS index is saved
faiss_index_path = "C:\\Users\\EndUser\\Downloads\\LLM Project\\faiss_index"

# Load the FAISS index from disk
index = faiss.read_index(os.path.join(faiss_index_path, "index.faiss"))

# Number of vectors in the index
print(f"Number of vectors in the index: {index.ntotal}")

# Example: Retrieve the first 5 vectors
num_vectors_to_view = 5
distances, indices = index.search(np.random.random((num_vectors_to_view, index.d)), num_vectors_to_view)

# Print distances and indices of the first few vectors
print(f"Distances: {distances}")
print(f"Indices: {indices}")
