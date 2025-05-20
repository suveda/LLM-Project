import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  


hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  

def load_faiss_index(faiss_index_path):
    '''Loads the faiss index from the given path'''

    return FAISS.load_local(faiss_index_path, hf_embeddings, allow_dangerous_deserialization=True)

def get_vector_store(faiss_index_path):
    '''Returns the initialized vector store from the FAISS index'''

    return load_faiss_index(faiss_index_path)


def search_faiss(query, vector_store, top_k=5):
    '''Searches the FAISS index for the top K similar embeddings'''

    # Convert the query to embedding using HuggingFace model
    query_embedding = hf_embeddings.embed_documents([query])
    
    # Convert the query embedding to a NumPy array
    query_embedding = np.array(query_embedding[0])

    # Perform the search on the FAISS index
    distances, indices = vector_store.index.search(query_embedding.reshape(1, -1), top_k) 

    # Retrieve the corresponding texts from the docstore using the indices
    similar_texts = [vector_store.docstore[i] for i in indices[0]]
    
    return similar_texts, distances[0], indices[0]


if __name__=="__main__":

    faiss_index_path = "C:\\Users\\EndUser\\Downloads\\LLM Project\\faiss_index"

    vector_store = get_vector_store(faiss_index_path)

    query_text = "A cute cat on a wooden floor"

    # Search the FAISS index for the top 5 similar texts
    similar_texts, distances, indices = search_faiss(query_text,vector_store, top_k=5)

    # Print the results
    for i, (text, dist, idx) in enumerate(zip(similar_texts, distances, indices)):
        print(f"Rank {i+1}: Index {idx} - Text: {text} (Distance: {dist})")
