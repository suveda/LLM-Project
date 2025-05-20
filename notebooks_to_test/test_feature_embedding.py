import google.generativeai as genai

# Configure API key (Get it from Google AI Studio)
genai.configure(api_key="AIzaSyC8syrU5DMiFGzpd7y5UEAB-YImq6F4TmE")

# Choose the embedding model (Gecko is optimized for text)
model = genai.GenerativeModel("models/text-embedding-geck")

# Example: Convert extracted labels into embeddings
labels = ["cat", "dog", "car", "banana"]
response = genai.embed_content(model="models/embedding-001", content=labels)

# Print the embeddings
for label, embedding in zip(labels, response["embedding"]):
    print(f"Label: {label} -> Embedding: {embedding[:5]}...")  # Show first 5 numbers