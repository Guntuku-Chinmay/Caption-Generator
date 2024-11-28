from flask import Flask, request, render_template
import cv2
import numpy as np
from PIL import Image
import os
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Initialize BLIP model for caption generation
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload
        image_file = request.files['image']
        if image_file:
            # Save image temporarily
            image_path = os.path.join("static", image_file.filename)
            image_file.save(image_path)

            # Read the image using OpenCV
            img = cv2.imread(image_path)

            # Extract features (this can be implemented with any feature extraction method, but we're skipping this step here)
            features = extract_features(img)

            # Generate caption using BLIP model
            caption = generate_caption(image_path)

            return render_template('index.html', image=image_file.filename, caption=caption)
    return render_template('index.html')


def extract_features(img):
    # Example feature extraction with OpenCV (optional - you can use CNN models or other methods)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray  # Placeholder for real feature extraction


def generate_caption(image_path):
    # Open the image using PIL (which is better for the model than OpenCV)
    raw_image = Image.open(image_path).convert("RGB")

    # Preprocess the image and generate a caption
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption


if __name__ == '__main__':
    app.run(debug=True)
