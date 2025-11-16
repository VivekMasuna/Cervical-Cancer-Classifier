import os
import numpy as np
import json
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

# Model paths
CNN_MODEL_PATH = 'models/cervical_cancer_cnn_results/cnn_cervical_cancer_model.h5'
VGG16_MODEL_PATH = 'models/cervical_cancer_vgg16_results/vgg16_cervical_cancer_model.h5'

# Metrics paths
CNN_METRICS_PATH = 'models/cervical_cancer_cnn_results/cnn_evaluation_metrics.json'
VGG16_METRICS_PATH = 'models/cervical_cancer_vgg16_results/vgg16_evaluation_metrics.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('models/cervical_cancer_cnn_results', exist_ok=True)
os.makedirs('models/cervical_cancer_vgg16_results', exist_ok=True)

# Class names (same as your training)
CLASS_NAMES = [
    "High squamous intra-epithelial lesion",
    "Low squamous intra-epithelial lesion", 
    "Negative for Intraepithelial malignancy",
    "Squamous cell carcinoma"
]

# Global model variables
cnn_model = None
vgg16_model = None
vgg16_is_saved_model = False  # Flag to track if VGG16 is a SavedModel

class SavedModelWrapper:
    """Wrapper class to make SavedModel behave like a Keras model"""
    def __init__(self, saved_model, signature_name='serving_default'):
        self.saved_model = saved_model
        self.signature = saved_model.signatures[signature_name]
        
        # Extract input and output keys from signature
        try:
            # Get input signature (usually in structured_input_signature[1])
            sig_input = self.signature.structured_input_signature
            if isinstance(sig_input, tuple) and len(sig_input) > 1:
                self.input_key = list(sig_input[1].keys())[0]
            else:
                # Fallback: try to get from function args
                self.input_key = list(sig_input.keys())[0] if isinstance(sig_input, dict) else 'input_1'
        except Exception as e:
            logger.warning(f"Could not determine input key from signature: {e}, using default")
            self.input_key = 'input_1'
    
    def predict(self, x, verbose=0):
        """Predict method compatible with Keras model.predict()"""
        try:
            # Convert numpy array to tensor if needed
            if isinstance(x, np.ndarray):
                x = tf.constant(x, dtype=tf.float32)
            
            # Call the signature with the input
            output = self.signature(**{self.input_key: x})
            
            # Get the output (usually the first output or 'output_0')
            if isinstance(output, dict):
                output_key = list(output.keys())[0]
                predictions = output[output_key].numpy()
            else:
                # If output is a single tensor
                predictions = output.numpy()
            
            return predictions
        except Exception as e:
            logger.error(f"Error in SavedModelWrapper.predict: {str(e)}")
            # Try alternative method: call directly without keyword
            try:
                output = self.signature(x)
                if isinstance(output, dict):
                    predictions = list(output.values())[0].numpy()
                else:
                    predictions = output.numpy()
                return predictions
            except Exception as e2:
                logger.error(f"Alternative prediction method also failed: {str(e2)}")
                raise

def load_cnn_model():
    global cnn_model
    if cnn_model is None:
        try:
            # Try loading with compile=False first to avoid compilation issues
            cnn_model = load_model(CNN_MODEL_PATH, compile=False)
            logger.info("CNN model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CNN model: {str(e)}")
            # Try loading with compile=True as fallback
            try:
                cnn_model = load_model(CNN_MODEL_PATH, compile=True)
                logger.info("CNN model loaded successfully (with compile=True)")
            except Exception as e2:
                logger.error(f"Error loading CNN model with compile=True: {str(e2)}")
                raise
    return cnn_model

def load_vgg16_model():
    global vgg16_model, vgg16_is_saved_model
    if vgg16_model is None:
        saved_model_path = 'models/cervical_cancer_vgg16_results/vgg16_saved_model'
        h5_model_path = VGG16_MODEL_PATH
        
        # Method 1: Try loading from saved_model directory using tf.saved_model.load (for Keras 3)
        if os.path.exists(saved_model_path):
            try:
                logger.info("Attempting to load VGG16 from saved_model directory using tf.saved_model.load...")
                saved_model_obj = tf.saved_model.load(saved_model_path)
                
                # Check available signatures
                signatures = list(saved_model_obj.signatures.keys())
                logger.info(f"Available signatures: {signatures}")
                
                # Use 'serving_default' if available, otherwise use the first signature
                signature_name = 'serving_default' if 'serving_default' in signatures else signatures[0]
                
                # Wrap the SavedModel to make it compatible with Keras model interface
                vgg16_model = SavedModelWrapper(saved_model_obj, signature_name=signature_name)
                vgg16_is_saved_model = True
                logger.info(f"VGG16 model loaded successfully from saved_model directory (using signature: {signature_name})")
                return vgg16_model
            except Exception as e1:
                logger.warning(f"Error loading from saved_model directory: {str(e1)}")
        
        # Method 2: Try loading from .h5 file with compile=False
        if os.path.exists(h5_model_path):
            try:
                logger.info("Attempting to load VGG16 from .h5 file...")
                vgg16_model = load_model(h5_model_path, compile=False)
                vgg16_is_saved_model = False
                logger.info("VGG16 model loaded successfully from .h5 file (without compilation)")
                return vgg16_model
            except Exception as e2:
                logger.warning(f"Error loading VGG16 .h5 with compile=False: {str(e2)}")
                # Try with compile=True as last resort
                try:
                    vgg16_model = load_model(h5_model_path, compile=True)
                    vgg16_is_saved_model = False
                    logger.info("VGG16 model loaded successfully from .h5 file (with compilation)")
                    return vgg16_model
                except Exception as e3:
                    logger.error(f"Error loading VGG16 .h5 with compile=True: {str(e3)}")
        
        # If all methods failed, raise an error
        error_msg = "Failed to load VGG16 model. Tried saved_model directory and .h5 file."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    return vgg16_model

def get_model(model_type):
    """Get the appropriate model based on type"""
    if model_type == 'cnn':
        return load_cnn_model()
    elif model_type == 'vgg16':
        return load_vgg16_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_metrics(model_type):
    """Load evaluation metrics from JSON file"""
    try:
        if model_type == 'cnn':
            metrics_path = CNN_METRICS_PATH
        elif model_type == 'vgg16':
            metrics_path = VGG16_METRICS_PATH
        else:
            return None
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Metrics file not found: {metrics_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        return None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img, model_type='vgg16', target_size=(224, 224)):
    """Preprocess image based on model type"""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_type == 'vgg16':
        # VGG16 requires specific preprocessing
        img_array = vgg_preprocess(img_array)
    elif model_type == 'cnn':
        # CNN typically uses normalization to [0, 1]
        img_array = img_array / 255.0
    else:
        # Default normalization
        img_array = img_array / 255.0
    
    return img_array

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get model type from form data or default to 'vgg16'
    model_type = request.form.get('model_type', 'vgg16').lower()
    if model_type not in ['cnn', 'vgg16']:
        return jsonify({'error': 'Invalid model type. Use "cnn" or "vgg16"'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Load appropriate model
            model = get_model(model_type)
            
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and preprocess image
            img = Image.open(filepath).convert('RGB')
            img_array = preprocess_image(img, model_type=model_type)
            
            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Prepare response
            result = {
                'prediction': CLASS_NAMES[predicted_class_idx],
                'confidence': confidence,
                'class_index': int(predicted_class_idx),
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'all_predictions': {
                    CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))
                }
            }
            
            logger.info(f"Prediction ({model_type.upper()}): {result['prediction']} (Confidence: {confidence:.2f})")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Prediction failed: ' + str(e)}), 500
    
    return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff'}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        cnn_status = "loaded" if cnn_model is not None else "not loaded"
        vgg16_status = "loaded" if vgg16_model is not None else "not loaded"
        return jsonify({
            'status': 'healthy', 
            'cnn_model_loaded': cnn_model is not None,
            'vgg16_model_loaded': vgg16_model is not None,
            'cnn_status': cnn_status,
            'vgg16_status': vgg16_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': CLASS_NAMES})

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models and their status"""
    return jsonify({
        'available_models': ['cnn', 'vgg16'],
        'cnn_loaded': cnn_model is not None,
        'vgg16_loaded': vgg16_model is not None
    })

@app.route('/api/metrics/<model_type>', methods=['GET'])
def get_metrics(model_type):
    """Get evaluation metrics for a specific model"""
    if model_type not in ['cnn', 'vgg16']:
        return jsonify({'error': 'Invalid model type. Use "cnn" or "vgg16"'}), 400
    
    metrics = load_metrics(model_type)
    if metrics is None:
        return jsonify({'error': 'Metrics not available for this model'}), 404
    
    return jsonify({
        'model_type': model_type,
        'metrics': metrics
    })

if __name__ == '__main__':
    # Pre-load models on startup (optional - can be lazy loaded)
    try:
        load_cnn_model()
        logger.info("CNN model pre-loaded")
    except Exception as e:
        logger.warning(f"Could not pre-load CNN model: {str(e)}")
    
    try:
        load_vgg16_model()
        logger.info("VGG16 model pre-loaded")
    except Exception as e:
        logger.warning(f"Could not pre-load VGG16 model: {str(e)}")
    
    logger.info("Server starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)