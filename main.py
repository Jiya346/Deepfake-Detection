from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your model
model = load_model("deepfake_classifier.h5")

def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))      # change to your model input size
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    filepath = os.path.join("static", "uploads", "uploaded.jpg")
    file.save(filepath)

    img = prepare_image(filepath)
    prediction = model.predict(img)

    prob = float(prediction[0][0])

    if prob > 0.5:
        label = "Deepfake"
        confidence = round(prob * 100, 2)
    else:
        label = "Real"
        confidence = round((1 - prob) * 100, 2)

    return render_template("result.html", label=label, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
