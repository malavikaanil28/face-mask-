import os
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model and class indices
model = load_model('model.h5')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}  # Reverse mapping

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/submit_user', methods=['POST'])
def submit_user():
    name = request.form['name']
    email = request.form['email']
    return render_template('input.html', name=name, email=email)

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    email = request.form['email']
    if 'file' not in request.files:
        return redirect(url_for('input'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('input'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess and predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0][0]
        class_id = 1 if prediction > 0.5 else 0
        class_name = class_names[class_id]
        confidence = prediction if class_id == 1 else 1 - prediction
        
        return render_template('output.html', 
                             name=name, 
                             email=email, 
                             prediction=class_name, 
                             confidence=f"{confidence:.2%}", 
                             image_path=file_path)
    return redirect(url_for('input'))

if __name__ == '__main__':
    app.run(debug=True)