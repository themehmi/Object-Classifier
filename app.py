from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Correctly point to static for web serving
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your model
model = load_model('cifar10_model.keras')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET', 'POST'])
def index():
    img_path = None
    top_predictions = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != '':
            # Save file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # This path is used in HTML: <img src="{{ img_path }}">
            # We strip 'static/' because Flask serves static files automatically
            img_path = filepath.replace('\\', '/') 

            # Preprocess
            img = image.load_img(filepath, target_size=(32, 32))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            probabilities = tf.nn.softmax(predictions[0]).numpy()

            # Get top 3
            top_indices = probabilities.argsort()[-3:][::-1]
            top_predictions = []
            for i in top_indices:
                top_predictions.append({
                    "label": class_names[i],
                    "confidence": round(float(probabilities[i]) * 100, 2)
                })

    return render_template('index.html', img_path=img_path, top_predictions=top_predictions)

if __name__ == '__main__':
    app.run(debug=True)