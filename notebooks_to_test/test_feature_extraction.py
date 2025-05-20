
# Imports the Google Cloud client library
from google.cloud import vision
import os

def run_quickstart() -> vision.EntityAnnotation:
    """Provides a quick start example for Cloud Vision."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\GoogleAPI\\vision-api-key.json"

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The URI of the image file to annotate
    image_path = "C:\\Users\\EndUser\\Downloads\\LLM Project\\images\\setagaya_small.jpeg"

    with open(image_path,"rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print("Labels:")
    for label in labels:
        print(label.description)

    return labels

run_quickstart()