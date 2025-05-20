from google import genai
from retreiving import search_faiss,get_vector_store

client = genai.Client(api_key="AIzaSyC8syrU5DMiFGzpd7y5UEAB-YImq6F4TmE")

def generate_caption(query,faiss_index_path,top_k=5):
    '''Generate a detailed caption using Gemini API based on similar texts from FAISS'''

    # Retreive the vector store
    vector_store = get_vector_store(faiss_index_path)

    # Retrieve the top similar texts
    similar_texts, distances, _ = search_faiss(query, vector_store, top_k=5)

    # Format the retrieved similar texts as context for the prompt
    
    retrieved_info = "\n".join([f"- {text} (Distance: {dist:.4f})" for text, dist in zip(similar_texts, distances)])
    
    # Prepare the prompt and generate a caption
    prompt = f"""
    You are an expert AI captioning assistant.

    **Extracted Image Features:**
    {query}

    **Similar Descriptions from FAISS (Ranked by Similarity):**
    {retrieved_info}

    **Task:** Generate a highly descriptive, natural-sounding caption that integrates both the extracted features and retrieved descriptions.
    - Describe how the retrieved objects relate to each other.
    - If multiple animals or objects are present, describe their interaction.
    - Describe the scene, background, and possible emotions.
    - Avoid generic captions like "The image depicts a cat."

    **Output:** Provide a rich, expressive caption in 2-3 sentences.
    """
    
    # Create the request to AI
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=prompt
    )
    
    # Extract and return the model's response
    return response.text

if __name__ == "__main__":

    faiss_index_path = "C:\\Users\\EndUser\\Downloads\\LLM Project\\faiss_index"
    query = "What is in the image?"
    response = generate_caption(query,faiss_index_path)
    print(response)
