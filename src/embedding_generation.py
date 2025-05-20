import os
import torch
import json
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Set the model to evaluation mode (no gradient updates needed)
model.eval()

def mean_pooling(model_output,attention_mask):
    '''Define mean pooling and take attention mask into account for correct averaging'''

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embedding(file_path):
    '''Generate embeddings for a single image ussing HuggingFace Transformer'''

    if file_path.endswith(".json"):
        with open(file_path,"r",encoding="utf-8") as file:
            data = json.load(file)

        features = data.get("features",[])

        if features:
            # Combine all extracted features into a single text string
            combined_labels = ", ".join(features)

            # Tokenize and generate embeddings

            input = tokenizer(combined_labels,padding=True,truncation=True,return_tensors='pt')

            with torch.no_grad():
                model_output = model(**input)

            # Compute average token embedding to create a single fixed size vector
            sentence_embeddings = mean_pooling(model_output,input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings,p=2,dim=1).squeeze().numpy()
    
    return sentence_embeddings,combined_labels

def process_embeddings(feature_extraction_path,embedding_path,single_image=None):
    '''Processes either embeddings for a single image or all the extracted features'''

    if single_image:

        file_path = os.path.join(feature_extraction_path,filename)

        embedding,label = generate_embedding(file_path)

        return {"text": label, "embeddings": embedding}

    for filename in os.listdir(feature_extraction_path):

        file_path = os.path.join(feature_extraction_path,filename)

        sentence_embeddings,combined_labels = generate_embedding(file_path)

        embedding_filename = os.path.splitext(filename)[0] + ".embedding"
        embedding_file_path = os.path.join(embedding_path, embedding_filename)

        with open(embedding_file_path,"w",encoding="utf-8") as emb_file:
            json.dump({
                    "text": combined_labels,
                    "embeddings": sentence_embeddings.tolist()
                },emb_file,indent=4)

print("All embeddings generated and saved successfully!")

if __name__=='__main__':

    feature_extraction_path = "C:\\Users\\EndUser\\Downloads\\LLM-Project\\extracted_features"

    embedding_path = "C:\\Users\\EndUser\\Downloads\\LLM-Project\\extracted_embeddings"

    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)

    process_embeddings(feature_extraction_path,embedding_path)
            

            

