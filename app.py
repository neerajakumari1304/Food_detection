import os
import json
import numpy as np
import tensorflow as tf
import redis
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- LOAD MODELS ----------------
custom_model = load_model("custom_cnn_model.h5")
resnet_model = load_model("resnet50_model.h5")
vgg_model = load_model("vgg16_model.h5")

# ---------------- CLASS NAMES ----------------
DATA_PATH = "data_split/train"
class_names = sorted(os.listdir(DATA_PATH))

# ---------------- REDIS ----------------
r = redis.StrictRedis(host='localhost', port=6379, db=0)

# ---------------- MODEL METRICS ----------------
metrics = {
    "custom": {
        "accuracy": 0.92,
        "precision": 0.90,
        "recall": 0.89,
        "f1_score": 0.89
    },
    "resnet": {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.93
    },
    "vgg": {
        "accuracy": 0.93,
        "precision": 0.91,
        "recall": 0.90,
        "f1_score": 0.90
    }
}

# ---------------- IMAGE PREPROCESS ----------------
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ---------------- HOME ----------------
@app.route("/")
def index():
    return render_template("index.html", classes=class_names)

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    model_type = request.form.get("model")
    selected_class = request.form.get("selected_class")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = prepare_image(filepath)

    if model_type == "custom":
        model = custom_model
    elif model_type == "resnet":
        model = resnet_model
    elif model_type == "vgg":
        model = vgg_model
    else:
        return jsonify({"error": "Invalid model selected"})

    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    predicted_class = class_names[predicted_index]

    # -------- Nutrition from Redis --------
    nutrition_data = {}
    data = r.get("food_items")
    if data:
        food_list = json.loads(data.decode("utf-8"))
        for item in food_list:
            if item["name"] == predicted_class:
                nutrition_data = item
                break

    return jsonify({
        "predicted_class": predicted_class,
        "selected_class": selected_class,
        "confidence": round(confidence * 100, 2),
        "model_used": model_type,
        "nutrition": nutrition_data,
        "image_path": "/" + filepath,
        "accuracy": metrics[model_type]["accuracy"],
        "precision": metrics[model_type]["precision"],
        "recall": metrics[model_type]["recall"],
        "f1_score": metrics[model_type]["f1_score"]
    })


if __name__ == "__main__":
    app.run(debug=True)