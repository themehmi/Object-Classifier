from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Initialize app
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load trained model
model = load_model('cifar10_model.keras')

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    img_path = None
    top_predictions = None

    if request.method == 'POST':
        file = request.files['file']

        if file:
            # Save uploaded image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img_path = filepath

            # Preprocess image
            img = image.load_img(filepath, target_size=(32, 32))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)

            # Convert logits to probabilities
            probabilities = tf.nn.softmax(predictions[0]).numpy()

            # Get top 3 predictions
            top_indices = probabilities.argsort()[-3:][::-1]

            top_predictions = []
            for i in top_indices:
                top_predictions.append({
                    "label": class_names[i],
                    "confidence": round(float(probabilities[i]) * 100, 2)
                })

    return render_template('index.html',
                           img_path=img_path,
                           top_predictions=top_predictions)


# Run app
if __name__ == '__main__':
    app.run(debug=True)