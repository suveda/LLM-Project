import streamlit as st
import os
import json
import numpy as np
from feature_extraction import extract_features
from embedding_generation import generate_embedding
from retreiving import get_vector_store,search_faiss
from genai_integration import generate_caption
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image

faiss_index_path = "C:\\Users\\EndUser\\Downloads\\LLM-Project\\faiss_index"

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = get_vector_store(faiss_index_path)

# Streamlit UI
st.title("Image Captioning")
st.write("Upload an image to generate a detailed caption.")

# Initialize file uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpeg","png","jpg"])

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image locally
    image_path = "uploaded_image.jpg"

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features from the image
    features = extract_features(image_path)

    # Generate embeddings for the features
    embeddings = hf_embeddings.embed_documents([",".join(features)])
    embeddings = np.array(embeddings[0]).reshape(1,-1)

    # Retrieve Similar Images using FAISS (using vector_store)
    similar_texts, distances, indices = search_faiss(", ".join(features), vector_store, top_k=5)

    # Prepare input for caption generation
    retrieved_texts = " ".join(similar_texts)
    full_query = f"Extracted Features: {",".join(features)}. Retrieved Descriptions: {retrieved_texts}"

    st.subheader("Caption Generation Inputs:")
    st.write(full_query)  

    # Generate Caption using Gemini AI
    caption = generate_caption(full_query,faiss_index_path)

    # Display results
    st.subheader("Top 5 Similar Image Descriptions from FAISS:")
    for i, (text, dist, idx) in enumerate(zip(similar_texts, distances, indices)):
        st.write(f"**Rank {i+1}:** Index {idx} - Text: {text} (Distance: {dist:.4f})")

    st.subheader("Generated Caption:")
    st.write(caption)


