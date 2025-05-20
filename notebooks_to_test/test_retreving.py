import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Importing the updated HuggingFaceEmbeddings class

# Explicitly pass a model_name to the HuggingFaceEmbeddings constructor
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  

# Load the FAISS index
faiss_index_path = "C:\\Users\\EndUser\\Downloads\\LLM Project\\faiss_index"
vector_store = FAISS.load_local(faiss_index_path, hf_embeddings, allow_dangerous_deserialization=True)

# Define a function to search for the top K similar texts using distances (D) and indices (I)
def search(query, top_k=5):
    # Convert the query to embedding using HuggingFace model
    query_embedding = hf_embeddings.embed_documents([query])
    
    # Convert the query embedding to a NumPy array (required by FAISS)
    query_embedding = np.array(query_embedding[0])

    # Perform the search on the FAISS index (this returns distances and indices)
    distances, indices = vector_store.index.search(query_embedding.reshape(1, -1), top_k)  # Normalize the search vector

    # Retrieve the corresponding texts from the docstore using the indices
    similar_texts = [vector_store.docstore[i] for i in indices[0]]
    
    return similar_texts, distances[0], indices[0]

# Example query text
query_text = "A cute cat on a wooden floor"

# Search the FAISS index for the top 5 similar texts
similar_texts, distances, indices = search(query_text, top_k=5)

# Print the results
for i, (text, dist, idx) in enumerate(zip(similar_texts, distances, indices)):
    print(f"Rank {i+1}: Index {idx} - Text: {text} (Distance: {dist})")
