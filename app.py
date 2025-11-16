import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import os

# Page configuration
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #F1F8E9;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #8BC34A;
        margin: 1rem 0;
    }
    .disease-name {
        font-size: 2rem;
        font-weight: bold;
        color: #1B5E20;
        margin-bottom: 0.5rem;
    }
    .confidence {
        font-size: 1.2rem;
        color: #33691E;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Model paths
MODEL_PATHS = {
    'banana': 'banana_resnet50.keras',
    'rice': 'rice_resnet50.keras',
    'sugarcane': 'Sugarcane_resnet50.keras'
}

# Disease classes for each crop
DISEASE_CLASSES = {
    'banana': ['Cordana', 'Healthy', 'Pestalotiopsis', 'Sigatoka'],
    'rice': ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 
             'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot'],
    'sugarcane': ['Healthy', 'Mosaic', 'Red Rot', 'Rust', 'Yellow']
}

# Disease information
DISEASE_INFO = {
    'banana': {
        'Cordana': {
            'description': 'Cordana leaf spot is a fungal disease that causes brown spots with yellow halos.',
            'treatment': 'Remove infected leaves, apply fungicides, ensure proper drainage.'
        },
        'Healthy': {
            'description': 'The leaf appears healthy with no visible signs of disease.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Pestalotiopsis': {
            'description': 'Fungal disease causing leaf spots with dark margins and gray centers.',
            'treatment': 'Prune affected areas, improve air circulation, use copper-based fungicides.'
        },
        'Sigatoka': {
            'description': 'Yellow/Black Sigatoka causes streaks and spots, reducing photosynthesis.',
            'treatment': 'Remove infected leaves, apply systemic fungicides, use resistant varieties.'
        }
    },
    'rice': {
        'Bacterial Leaf Blight': {
            'description': 'Bacterial infection causing water-soaked lesions that turn yellow-white.',
            'treatment': 'Use resistant varieties, apply copper-based bactericides, maintain field hygiene.'
        },
        'Brown Spot': {
            'description': 'Fungal disease with circular brown spots on leaves, affecting grain quality.',
            'treatment': 'Apply fungicides, ensure proper nutrition, use certified seeds.'
        },
        'Healthy': {
            'description': 'The leaf appears healthy with no visible signs of disease.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Leaf Blast': {
            'description': 'Fungal disease causing diamond-shaped lesions that can destroy entire leaves.',
            'treatment': 'Use resistant varieties, apply fungicides like tricyclazole, avoid excess nitrogen.'
        },
        'Leaf Scald': {
            'description': 'Bacterial disease causing scalded appearance with alternating light and dark bands.',
            'treatment': 'Use disease-free seeds, apply bactericides, maintain proper water management.'
        },
        'Narrow Brown Spot': {
            'description': 'Fungal disease with narrow brown lesions, often confused with brown spot.',
            'treatment': 'Apply appropriate fungicides, improve field drainage, rotate crops.'
        }
    },
    'sugarcane': {
        'Healthy': {
            'description': 'The leaf appears healthy with no visible signs of disease.',
            'treatment': 'Continue regular care and monitoring.'
        },
        'Mosaic': {
            'description': 'Viral disease causing yellow-green mosaic patterns and stunted growth.',
            'treatment': 'Remove infected plants, control aphid vectors, use virus-free planting material.'
        },
        'Red Rot': {
            'description': 'Fungal disease causing red discoloration and hollowing of stalks.',
            'treatment': 'Use resistant varieties, hot water treatment of setts, field sanitation.'
        },
        'Rust': {
            'description': 'Fungal disease with orange-brown pustules on leaves reducing photosynthesis.',
            'treatment': 'Apply fungicides, remove infected leaves, use resistant varieties.'
        },
        'Yellow': {
            'description': 'Yellowing of leaves due to various factors including nutrient deficiency or disease.',
            'treatment': 'Assess nutrient levels, improve soil health, check for pests and diseases.'
        }
    }
}

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    for crop, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                models[crop] = keras.models.load_model(path)
                st.sidebar.success(f"✓ {crop.capitalize()} model loaded")
            except Exception as e:
                st.sidebar.error(f"✗ Error loading {crop} model: {str(e)}")
        else:
            st.sidebar.warning(f"⚠ {crop.capitalize()} model not found at {path}")
    return models

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def detect_green_area(image):
    """Detect green leaf area to help with crop detection"""
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Define range for green color
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_percentage = (np.sum(mask > 0) / mask.size) * 100
    
    return green_percentage

def auto_detect_crop(image, models):
    """Automatically detect crop type based on predictions from all models"""
    preprocessed = preprocess_image(image)
    confidences = {}
    
    for crop, model in models.items():
        prediction = model.predict(preprocessed, verbose=0)
        max_confidence = float(np.max(prediction))  # Convert to Python float
        confidences[crop] = max_confidence
    
    # Select crop with highest confidence
    detected_crop = max(confidences, key=confidences.get)
    return detected_crop, confidences

def predict_disease(image, crop, model):
    """Predict disease for the given crop"""
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed, verbose=0)
    
    # Get top prediction
    predicted_class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class_idx] * 100)  # Convert to Python float
    disease_name = DISEASE_CLASSES[crop][predicted_class_idx]
    
    # Get all predictions for display
    all_predictions = []
    for idx, prob in enumerate(prediction[0]):
        all_predictions.append({
            'disease': DISEASE_CLASSES[crop][idx],
            'confidence': float(prob * 100)  # Convert to Python float
        })
    
    # Sort by confidence
    all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
    
    return disease_name, confidence, all_predictions

def display_prediction_results(disease_name, confidence, all_predictions, crop):
    """Display prediction results with styling"""
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    
    # Main prediction
    st.markdown(f'<div class="disease-name">🔍 Detected: {disease_name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
    
    # Progress bar for confidence
    st.progress(confidence / 100)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Disease information
    if disease_name in DISEASE_INFO[crop]:
        info = DISEASE_INFO[crop][disease_name]
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**📋 Description:** {info['description']}")
        st.markdown(f"**💊 Treatment:** {info['treatment']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # All predictions
    with st.expander("📊 View All Predictions"):
        for pred in all_predictions:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(pred['disease'])
            with col2:
                st.write(f"{pred['confidence']:.2f}%")
            st.progress(pred['confidence'] / 100)

def main():
    # Header
    st.markdown('<div class="main-header">🌾 Crop Disease Detection System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Load models
    st.sidebar.subheader("📦 Model Status")
    models = load_models()
    
    if not models:
        st.error("❌ No models loaded! Please ensure model files are present at the specified paths.")
        st.info(f"Expected model locations:\n" + "\n".join([f"- {path}" for path in MODEL_PATHS.values()]))
        return
    
    st.sidebar.markdown("---")
    
    # Crop selection
    st.sidebar.subheader("🌱 Select Crop Type")
    crop_options = ['Auto Detect'] + [crop.capitalize() for crop in models.keys()]
    selected_crop = st.sidebar.selectbox("Choose crop:", crop_options)
    
    # Info about the app
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        "This app uses ResNet50 deep learning models to detect diseases in crop leaves. "
        "Upload a clear image of a leaf and get instant predictions!"
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📤 Upload Leaf Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image file (JPG, JPEG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of the crop leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.caption(f"Image size: {image.size[0]} x {image.size[1]} pixels")
            green_area = detect_green_area(image)
            st.caption(f"Green area detected: {green_area:.1f}%")
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<div class="sub-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
            
            # Predict button
            if st.button("🔬 Analyze Leaf", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Auto detect or use selected crop
                        if selected_crop == 'Auto Detect':
                            detected_crop, crop_confidences = auto_detect_crop(image, models)
                            
                            st.info(f"🤖 Auto-detected crop: **{detected_crop.capitalize()}**")
                            
                            # Show crop detection confidences
                            with st.expander("🌾 Crop Detection Confidences"):
                                for crop, conf in sorted(crop_confidences.items(), key=lambda x: x[1], reverse=True):
                                    st.write(f"{crop.capitalize()}: {conf*100:.2f}%")
                                    st.progress(conf)
                            
                            crop_to_use = detected_crop
                        else:
                            crop_to_use = selected_crop.lower()
                        
                        # Predict disease
                        if crop_to_use in models:
                            disease_name, confidence, all_predictions = predict_disease(
                                image, crop_to_use, models[crop_to_use]
                            )
                            
                            # Display results
                            display_prediction_results(disease_name, confidence, all_predictions, crop_to_use)
                            
                        else:
                            st.error(f"Model for {crop_to_use} not available!")
                            
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
        else:
            st.info("👈 Upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🌿 Crop Disease Detection System | Powered by ResNet50 & TensorFlow"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()