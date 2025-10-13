import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64
import logging
from datetime import datetime

# Model-specific preprocessing
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MODEL_PATH = 'models/vgg16_final.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Class names (same as your training)
CLASS_NAMES = [
    "High squamous intra-epithelial lesion",
    "Low squamous intra-epithelial lesion", 
    "Negative for Intraepithelial malignancy",
    "Squamous cell carcinoma"
]

# Global model variable
model = None

def load_model_once():
    global model
    if model is None:
        try:
            model = load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    return model

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for VGG16"""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = vgg_preprocess(img_array)
    return img_array

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Load model
            model = load_model_once()
            
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and preprocess image
            img = Image.open(filepath).convert('RGB')
            img_array = preprocess_image(img)
            
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Prepare response
            result = {
                'prediction': CLASS_NAMES[predicted_class_idx],
                'confidence': confidence,
                'class_index': int(predicted_class_idx),
                'timestamp': datetime.now().isoformat(),
                'all_predictions': {
                    CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))
                }
            }
            
            logger.info(f"Prediction: {result['prediction']} (Confidence: {confidence:.2f})")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Prediction failed: ' + str(e)}), 500
    
    return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff'}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        model_status = "loaded" if model is not None else "not loaded"
        return jsonify({
            'status': 'healthy', 
            'model_loaded': model is not None,
            'model_status': model_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': CLASS_NAMES})

if __name__ == '__main__':
    # Pre-load model on startup
    try:
        load_model_once()
        logger.info("Server starting with pre-loaded model")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {str(e)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)