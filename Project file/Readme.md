from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("Blood Cell.h5")  # Make sure the model file is in the same folder

classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']
    if file.filename == '':
        return "No image selected for uploading!"

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Preprocess the image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    return render_template("result.html", prediction=predicted_class, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)

