import os
from google.cloud import vision
import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\GoogleAPI\\vision-api-key.json"

# Instantiates a client
client = vision.ImageAnnotatorClient()

def extract_features(image_path):
    '''Extract features from a single image using Google Vision API'''

    with open(image_path,"rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)

    labels = [label.description for label in response.label_annotations]

    return labels

def process_image(folder_path,output_folder,single_image=None):
    '''Processes either a single image or all images in the images folder'''

    if single_image:
        image_path = os.path.join(folder_path,single_image)
        labels = extract_features(image_path)
        return {"image_path": single_image, "features": labels}

    for image_name in os.listdir(folder_path):

        # navigate to the image path
        image_path = os.path.join(folder_path,image_name)

        labels = extract_features(image_path)

        # create a json file for each image with its labels in _features folder
        json_filename = os.path.splitext(image_name)[0]+ ".json"

        json_path = os.path.join(output_folder,json_filename)

        image_features = {
            "image_path": image_name,
            "features": labels
        }

        with open(json_path,"w") as json_file:
            json.dump(image_features,json_file,indent=4)

if __name__=="__main__":

    folder_path = "C:\\Users\\EndUser\\Downloads\\LLM Project\\images"
    
    output_folder = "../extracted_features"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_image(folder_path,output_folder)

