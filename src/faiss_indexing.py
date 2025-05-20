import os
import json
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings 


def build_faiss_index(embedding_path,faiss_index_path):
    """Builds and saves a FAISS index from extracted embeddings (Batch Processing)."""
    
    all_embeddings = []
    texts = []

    # Load the embeddings from the stored .embedding files
    for filename in os.listdir(embedding_path):
        file_path = os.path.join(embedding_path, filename)

        if filename.endswith(".embedding"):
            with open(file_path, "r", encoding="utf-8") as emb_file:
                embedding_data = json.load(emb_file)
                embedding = np.array(embedding_data["embeddings"], dtype=np.float32)
                text = embedding_data["text"]
                all_embeddings.append(embedding)
                texts.append(text)

    # Convert the list of embeddings into a numpy array
    embedding_array = np.array(all_embeddings, dtype=np.float32)

    # Create the FAISS index from the embeddings
    embedding_dim = embedding_array.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)

    # Add the embeddings to the FAISS index
    faiss_index.add(embedding_array)

    # Create a document store and document ID to index mapping
    docstore = {i: text for i, text in enumerate(texts)} 

    # maps faiss index to correct docstore ID
    index_to_docstore_id = {i: i for i in range(len(texts))}  

    vector_store = FAISS(
        embedding_function=HuggingFaceEmbeddings(), 
        index=faiss_index,
        docstore=docstore, 
        index_to_docstore_id=index_to_docstore_id
    )

    # Save the FAISS index using LangChain's save_local method
    vector_store.save_local(faiss_index_path)

    print(f"FAISS index created and saved at {faiss_index_path}!")

if __name__=="__main__":

    embedding_path = "C:\\Users\\EndUser\\Downloads\\LLM-Project\\extracted_embeddings"

    faiss_index_path = "C:\\Users\\EndUser\\Downloads\\LLM-Project\\faiss_index"

    if not os.path.exists(faiss_index_path):
        os.makedirs(faiss_index_path)

    build_faiss_index(embedding_path,faiss_index_path)


