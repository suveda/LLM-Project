import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load Hugging Face model & tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Set the model to evaluation mode (no gradient updates needed)
model.eval()

# Path where the feature extraction files are stored
feature_extraction_path = "C:\\Users\\EndUser\\Downloads\\LLM Project\\extracted_features"

# Path to save the generated embedding files
embedding_path = "C:\\Users\\EndUser\\Downloads\\LLM Project\\extracted_embeddings"

if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)

# Process each file in the feature extraction folder
for filename in os.listdir(feature_extraction_path):
    file_path = os.path.join(feature_extraction_path, filename)

    if filename.endswith(".json"):  # Process only JSON files containing feature data
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        features = data.get("features", [])  # Extract features from the file (this is where features are stored)

        if features:
            # Combine all extracted features into a single text string
            combined_labels = ", ".join(features)

            # Tokenize and generate embeddings
            inputs = tokenizer(combined_labels, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                output = model(**inputs)
                embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()

            # Save the generated embedding into a new .embedding file
            embedding_filename = os.path.splitext(filename)[0] + ".embedding"
            embedding_file_path = os.path.join(embedding_path, embedding_filename)

            # Save the text and embedding to a new file
            with open(embedding_file_path, "w", encoding="utf-8") as emb_file:
                json.dump({
                    "text": combined_labels,  # Store the combined labels as text
                    "embedding": embedding.tolist()  # Store the generated embedding as a list
                }, emb_file, indent=4)

            print(f"Embedding saved for {filename}!")

print("All embeddings generated and saved successfully!")
