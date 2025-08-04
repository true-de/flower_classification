import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
# import cv2
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Try to import TensorFlow
tf_available = True
try:
    import tensorflow as tf
except ImportError:
    tf_available = False

CLASS_INFO = {
    'daisy': {
        'emoji': '🌼',
        'description': 'Simple white petals with yellow center',
        'color': '#FFFFE0'  # Light yellow-white
    },
    'lavender': {
        'emoji': '💜',
        'description': 'Purple spikes with a calming fragrance',
        'color': '#B57EDC'  # Lavender
    },
    'lotus': {
        'emoji': '🪷',
        'description': 'Sacred water lily with large round leaves',
        'color': '#FFC0CB'  # Pink
    },
    'sunflower': {
        'emoji': '🌻',
        'description': 'Large yellow petals with dark center',
        'color': '#FFD700'  # Gold (sunflower yellow)
    },
    'tulip': {
        'emoji': '🌷',
        'description': 'Cup-shaped flower with smooth petals',
        'color': '#FF69B4'  # Hot pink (common tulip color)
    }
}

CLASS_NAMES = list(CLASS_INFO.keys())

# Enhanced page configuration
st.set_page_config(
    page_title="Blossom AI",
    page_icon="🌼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling with modern design elements
st.markdown("""
<style>
    /* Main styles with enhanced colors and effects */
    .main-header {
        text-align: center;
        padding: 2.5rem 0;
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        border-radius: 15px;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        animation: gradient 15s ease infinite;
        background-size: 300% 300%;
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, rgba(255,255,255,0.15) 25%, transparent 25%, transparent 50%, rgba(255,255,255,0.15) 50%, rgba(255,255,255,0.15) 75%, transparent 75%);
        background-size: 25px 25px;
        opacity: 0.4;
        z-index: 0;
        animation: slide 20s linear infinite;
    }
    
    @keyframes slide {
        0% {background-position: 0 0;}
        100% {background-position: 100px 100px;}
    }
    
    .main-header > * {
        position: relative;
        z-index: 1;
    }
    
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.8rem;
        color: #3a7bd5;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08), 0 1px 8px rgba(0,0,0,0.03);
        margin: 1.2rem 0;
        border-left: 6px solid #3a7bd5;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        right: 0;
        width: 120px;
        height: 120px;
        background: linear-gradient(135deg, transparent 50%, rgba(58, 123, 213, 0.1) 50%);
        border-radius: 0 0 18px 0;
    }
    
    .prediction-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0,0,0,0.12), 0 5px 15px rgba(0,0,0,0.06);
    }
    
    .prediction-card h4 {
        color: #3a7bd5;
        margin-bottom: 1.2rem;
        font-size: 1.4rem;
        border-bottom: 2px solid rgba(58, 123, 213, 0.2);
        padding-bottom: 0.7rem;
        letter-spacing: 0.5px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 18px;
        text-align: center;
        margin: 1.2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15), 0 5px 10px rgba(0,0,0,0.05);
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
        opacity: 0;
        transition: opacity 0.5s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05) translateY(-7px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2), 0 15px 20px rgba(0,0,0,0.1);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .info-box {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        border: none;
        border-radius: 18px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1), 0 3px 6px rgba(0,0,0,0.05);
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        color: white;
    }
    
    .info-box::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 0 60px 60px 0;
        border-color: transparent rgba(255,255,255,0.15) transparent transparent;
    }
    
    .info-box:hover {
        transform: translateY(-7px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15), 0 10px 10px rgba(0,0,0,0.05);
    }
    
    /* Additional styling for better UI */
    h1 {
        color: white;
        font-weight: 800;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        font-size: 2.5rem;
    }
    
    h2, h3 {
        color: #3a7bd5;
        font-weight: 700;
        position: relative;
        padding-bottom: 0.7rem;
        letter-spacing: 0.5px;
    }
    
    h2::after, h3::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
        border-radius: 4px;
    }
    
    h4 {
        color: #3a7bd5;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        z-index: 1;
        box-shadow: 0 5px 15px rgba(58, 123, 213, 0.3);
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: all 0.7s ease;
        z-index: -1;
    }
    
    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(58, 123, 213, 0.4);
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
        color: white;
        border-radius: 18px;
        margin-top: 4rem;
        box-shadow: 0 -10px 30px rgba(0,0,0,0.1), 0 5px 15px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at center, rgba(255,255,255,0.3) 0%, transparent 70%);
    }
    
    .footer h4 {
        color: white;
        margin-bottom: 1.2rem;
        position: relative;
        display: inline-block;
        font-size: 1.5rem;
        letter-spacing: 1px;
    }
    
    .footer h4::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 20%;
        width: 60%;
        height: 3px;
        background: rgba(255,255,255,0.6);
        border-radius: 3px;
    }
    
    /* Streamlit component overrides - Fixed Slider Styling */
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.9);
        border-radius: 12px;
        transition: all 0.4s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .stSelectbox > div > div:hover {
        background-color: rgba(255,255,255,1);
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }

    /* Enhanced Slider Styling */
    .stSlider {
        padding: 1rem;
    }

    .stSlider > div {
        color: #000;
        background-color: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1.5rem;
        transition: all 0.4s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 2px solid rgba(58, 123, 213, 0.1);
    }

    .stSlider > div:hover {
        background-color: rgba(255,255,255,1);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        border-color: rgba(58, 123, 213, 0.3);
        transform: translateY(-2px);
    }

    /* Slider track styling */
    .stSlider .stSlider > div > div > div {
        background: linear-gradient(90deg, #e9ecef 0%, #dee2e6 100%);
        border-radius: 10px;
        height: 8px;
    }

    /* Slider thumb styling */
    .stSlider .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
        border-radius: 50%;
        width: 20px;
        height: 20px;
        box-shadow: 0 4px 12px rgba(58, 123, 213, 0.4);
        transition: all 0.3s ease;
    }

    .stSlider .stSlider > div > div > div > div:hover {
        transform: scale(1.2);
        box-shadow: 0 6px 16px rgba(58, 123, 213, 0.6);
    }

    /* Slider label styling */
    .stSlider > label {
        color: #3a7bd5;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        letter-spacing: 0.3px;
    }

    /* Slider value display */
    .stSlider > div > div > div > div:last-child {
        color: #3a7bd5;
        font-weight: 700;
        font-size: 1.1rem;
        background: linear-gradient(135deg, rgba(58, 123, 213, 0.1), rgba(0, 210, 255, 0.1));
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        border: 1px solid rgba(58, 123, 213, 0.2);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        border-radius: 12px;
        border: 2px dashed rgba(58, 123, 213, 0.3);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: rgba(58, 123, 213, 0.6);
        background-color: rgba(58, 123, 213, 0.05);
    }
    
    /* Custom animations */
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-12px) rotate(2deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .float-animation {
        animation: float 6s ease-in-out infinite;
    }
    
    .pulse-animation {
        animation: pulse 3s ease-in-out infinite;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 0 20px 20px 0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
        border-radius: 10px;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        width: 200px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    if not tf_available:
        return None
    
    try:
        if os.path.exists('model1.keras'):
            return tf.keras.models.load_model('model1.keras')
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Enhanced image preprocessing with augmentation options
def preprocess_image(image, augment=False):
    if not tf_available:
        return None
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Optional augmentation for better predictions
    if augment:
        # Random rotation (-30 to 30 degrees)
        rotation_angle = np.random.uniform(-30, 30)
        image = image.rotate(rotation_angle)
        
        # Random brightness adjustment (0.8 to 1.2)
        brightness_factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        # Random contrast adjustment (0.8 to 1.2)
        contrast_factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        # Random sharpness adjustment (0.8 to 1.2)
        sharpness_factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness_factor)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Resize and normalize
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Create confidence visualization
def create_confidence_chart(predictions, class_names):
    fig = go.Figure(data=[
        go.Bar(
            x=[f"{CLASS_INFO[name]['emoji']} {name}" for name in class_names],
            y=predictions * 100,
            marker_color=[CLASS_INFO[name]['color'] for name in class_names],
            text=[f"{pred:.1f}%" for pred in predictions * 100],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Flower Type",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=400
    )
    
    return fig

# Gradient analysis (simulated feature importance)
def analyze_image_features(image):
    """Simulate feature analysis for educational purposes"""
    image_array = np.array(image.resize((128, 128)))

    # Simulate color analysis
    avg_colors = np.mean(image_array, axis=(0, 1))
    color_dominance = {
        'Red': avg_colors[0] / 255.0,
        'Green': avg_colors[1] / 255.0,
        'Blue': avg_colors[2] / 255.0
    }
    
    # Simulate texture analysis (using edge detection)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (128 * 128)
    
    return color_dominance, edge_density

def make_prediction_with_enhanced_messages(model, image, confidence_threshold=0.5, enhance_image=True):
    """
    Make prediction with enhanced error handling and user-friendly messages
    """
    # Preprocess image
    processed_image = preprocess_image(image, augment=enhance_image)

    # Make prediction
    predictions = model.predict(processed_image, verbose=0)[0]

    # Get sorted predictions
    pred_indices = np.argsort(predictions)[::-1]

    # Main prediction
    top_pred_idx = pred_indices[0]
    top_class = CLASS_NAMES[top_pred_idx]
    top_confidence = predictions[top_pred_idx]

    # Calculate prediction entropy (measures uncertainty)
    entropy = -np.sum(predictions * np.log(predictions + 1e-10))

    # Calculate confidence spread (difference between top 2 predictions)
    confidence_spread = predictions[pred_indices[0]] - predictions[pred_indices[1]]
    
    # Enhanced decision logic
    if top_confidence >= confidence_threshold and confidence_spread > 0.1:
        # High confidence, clear winner
        st.success(f"🎉 I'm {top_confidence*100:.1f}% confident this is a **{top_class}**!")
        return "success", top_class, top_confidence, predictions, pred_indices
        
    elif top_confidence >= 0.3 and confidence_spread > 0.05:
        # Medium confidence
        st.warning(f"🤔 I think this might be a **{top_class}** ({top_confidence*100:.1f}% confidence), but I'm not entirely sure.")
        st.info("💡 **Tip**: Try uploading a clearer image with better lighting for more accurate results.")
        return "uncertain", top_class, top_confidence, predictions, pred_indices
        
    elif entropy > 1.5:  # High entropy indicates uniform distribution
        # Very uncertain - likely not a flower or very unclear image
        st.error("❌ **This doesn't appear to be a recognizable flower image.**")
        st.markdown("""
        **Possible reasons:**
        - The image might not contain a flower
        - The image quality is too poor
        - The flower is too small or unclear in the image
        - The lighting conditions are not suitable
        """)
        st.info("🔄 **Please try**: Uploading a clear, well-lit image of a flower that fills most of the frame.")
        return "not_flower", None, top_confidence, predictions, pred_indices
        
    elif top_confidence < 0.3:
        # Low confidence - likely unsupported flower type
        st.error("❌ **This flower type is not in my training database.**")
        st.markdown(f"""
        **I can only identify these flower types:**
        {', '.join([f"{CLASS_INFO[name]['emoji']} {name}" for name in CLASS_NAMES])}
        
        **Your image might contain:**
        - A flower type I haven't been trained on
        - A flower that's too damaged or unclear to identify
        - An object that's not a flower
        """)
        st.info("🌸 **Suggestion**: Try uploading an image of one of the supported flower types listed above.")
        return "unsupported_flower", None, top_confidence, predictions, pred_indices
        
    else:
        # Edge case - moderate confidence but low spread
        st.warning(f"🤷‍♂️ I'm having trouble deciding between a few flower types. My best guess is **{top_class}** ({top_confidence*100:.1f}% confidence).")
        st.markdown("""
        **This could mean:**
        - The image shows characteristics of multiple flower types
        - The angle or lighting makes identification difficult
        - The flower is in an unusual state (wilted, partially visible, etc.)
        """)
        st.info("📸 **Try**: Taking a photo from a different angle or with better lighting.")
        return "ambiguous", top_class, top_confidence, predictions, pred_indices

def assess_image_quality(image):
    """
    Assess image quality to help users understand why predictions might fail
    """
    import cv2
    
    # Convert to array
    img_array = np.array(image)
    
    # Brightness assessment
    brightness = np.mean(img_array) / 255.0
    
    # Sharpness assessment using Laplacian variance
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0  # Normalized
    
    # Size assessment
    size_score = min(image.size[0] * image.size[1] / (500 * 500), 1.0)  # Normalized to 500x500
    
    # Overall score
    overall_score = (
        (0.5 if 0.3 <= brightness <= 0.8 else 0.2) +  # Good brightness range
        (min(sharpness, 1.0) * 0.3) +  # Sharpness contribution
        (size_score * 0.2)  # Size contribution
    )
    
    return {
        'brightness': brightness,
        'sharpness': min(sharpness, 1.0),
        'size_score': size_score,
        'overall_score': overall_score
    }

def show_helpful_suggestions():
    """
    Show helpful suggestions for better flower recognition
    """
    st.markdown("### 💡 Tips for Better Recognition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📸 Photography Tips:**
        - Fill the frame with the flower
        - Use natural daylight when possible
        - Avoid shadows on the flower
        - Take photos from multiple angles
        - Ensure the flower is in focus
        """)
        
    with col2:
        st.markdown("""
        **🌸 Supported Flowers:**
        - 🌼 Daisy - White petals, yellow center
        - 🌻 Sunflower - Large, bright yellow
        - 🌹 Rose - Various colors, layered petals
        - 🌷 Tulip - Cup-shaped, smooth petals
        - 🌾 Dandelion - Yellow, spiky petals
        """)

# Main app
def main():
    # Enhanced header with more engaging content
    st.markdown("""
    <div class="main-header">
        <h1>🌼 Blossom AI – Smart Flower Classifier</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">Advanced AI-powered flower identification with detailed analysis</p>
        <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 1rem;">
            <span style="background: rgba(255,255,255,0.3); padding: 0.4rem 0.8rem; border-radius: 20px; backdrop-filter: blur(5px);">🔍 High Accuracy</span>
            <span style="background: rgba(255,255,255,0.3); padding: 0.4rem 0.8rem; border-radius: 20px; backdrop-filter: blur(5px);">📊 Detailed Analysis</span>
            <span style="background: rgba(255,255,255,0.3); padding: 0.4rem 0.8rem; border-radius: 20px; backdrop-filter: blur(5px);">🌸 5 Flower Types</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model not available")
        if not tf_available:
            st.info("📦 TensorFlow not installed. Install with: `pip install tensorflow`")
        else:
            st.info("🏃‍♂️ Train the model first: `python train_flower_classifier.py`")
        return
    
    # Sidebar configuration
    st.sidebar.markdown("## 🎛️ Configuration")
    enhance_image = st.sidebar.checkbox("🔧 Enhance image quality", value=True)
    show_analysis = st.sidebar.checkbox("📊 Show detailed analysis", value=True)
    confidence_threshold = st.sidebar.slider("🎯 Confidence threshold", 0.0, 1.0, 0.5)

    # File upload section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📁 Upload Your Flower Image")
        uploaded_file = st.file_uploader(
            "Choose a flower image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a flower for best results"
        )

    with col2:
        if uploaded_file:
            st.markdown("### 📊 Quick Stats")
            file_size = len(uploaded_file.getvalue())
            st.markdown(f"""
            <div class="info-box">
                <strong>File:</strong> {uploaded_file.name}<br>
                <strong>Size:</strong> {file_size/1024:.1f} KB<br>
                <strong>Type:</strong> {uploaded_file.type}
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)

            # Image display section
            st.markdown("### 🖼️ Uploaded Image")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption='Your Flower Image', use_container_width=True)

            # Prediction section with enhanced error handling
            st.markdown("### 🔮 AI Prediction")

            with st.spinner("🤖 Analyzing your flower..."):
                # Preprocess image
                processed_image = preprocess_image(image, augment=enhance_image)

                # Make prediction
                predictions = model.predict(processed_image, verbose=0)[0]

                # Get sorted predictions
                pred_indices = np.argsort(predictions)[::-1]

                # Main prediction
                top_pred_idx = pred_indices[0]
                top_class = CLASS_NAMES[top_pred_idx]
                top_confidence = predictions[top_pred_idx]

                # Calculate prediction entropy and confidence spread
                entropy = -np.sum(predictions * np.log(predictions + 1e-10))
                confidence_spread = predictions[pred_indices[0]] - predictions[pred_indices[1]]

                # Enhanced decision logic with better messages
                show_detailed_analysis = True

                if top_confidence >= confidence_threshold and confidence_spread > 0.1:
                    # High confidence, clear winner
                    st.success(f"🎉 I'm {top_confidence*100:.1f}% confident this is a **{top_class}**!")

                elif top_confidence >= 0.3 and confidence_spread > 0.05:
                    # Medium confidence
                    st.warning(f"🤔 I think this might be a **{top_class}** ({top_confidence*100:.1f}% confidence), but I'm not entirely sure.")
                    st.info("💡 **Tip**: Try uploading a clearer image with better lighting for more accurate results.")

                elif entropy > 1.5:  # High entropy indicates uniform distribution
                    # Very uncertain - likely not a flower or very unclear image
                    st.error("❌ **This doesn't appear to be a recognizable flower image.**")
                    st.markdown("""
                    **Possible reasons:**
                    - The image might not contain a flower
                    - The image quality is too poor
                    - The flower is too small or unclear in the image
                    - The lighting conditions are not suitable
                    """)
                    st.info("🔄 **Please try**: Uploading a clear, well-lit image of a flower that fills most of the frame.")
                    show_detailed_analysis = False

                elif top_confidence < 0.3:
                    # Low confidence - likely unsupported flower type
                    st.error("❌ **This flower type is not in my training database.**")
                    st.markdown(f"""
                    **I can only identify these flower types:**
                    {', '.join([f"{CLASS_INFO[name]['emoji']} {name}" for name in CLASS_NAMES])}
                    
                    **Your image might contain:**
                    - A flower type I haven't been trained on
                    - A flower that's too damaged or unclear to identify
                    - An object that's not a flower
                    """)
                    st.info("🌸 **Suggestion**: Try uploading an image of one of the supported flower types listed above.")
                    show_detailed_analysis = False
                    
                else:
                    # Edge case - moderate confidence but low spread
                    st.warning(f"🤷‍♂️ I'm having trouble deciding between a few flower types. My best guess is **{top_class}** ({top_confidence*100:.1f}% confidence).")
                    st.markdown("""
                    **This could mean:**
                    - The image shows characteristics of multiple flower types
                    - The angle or lighting makes identification difficult
                    - The flower is in an unusual state (wilted, partially visible, etc.)
                    """)
                    st.info("📸 **Try**: Taking a photo from a different angle or with better lighting.")

                # Show reference predictions even for failed cases
                if not show_detailed_analysis:
                    st.markdown("#### 🔍 For reference, here's what I detected:")
                    for i, idx in enumerate(pred_indices[:3]):
                        class_name = CLASS_NAMES[idx]
                        conf = predictions[idx]
                        emoji = CLASS_INFO[class_name]['emoji']
                        st.markdown(f"{emoji} {class_name}: {conf*100:.1f}%")

                # Image quality assessment
                st.markdown("#### 📊 Image Quality Assessment")
                image_quality = assess_image_quality(image)

                if image_quality['overall_score'] < 0.5:
                    st.warning("⚠️ **Image quality could be improved:**")
                    if image_quality['brightness'] < 0.3:
                        st.markdown("- Image appears too dark")
                    if image_quality['brightness'] > 0.8:
                        st.markdown("- Image appears too bright")
                    if image_quality['sharpness'] < 0.3:
                        st.markdown("- Image appears blurry")
                    if image_quality['size_score'] < 0.5:
                        st.markdown("- Image resolution is quite low")
                else:
                    st.success("✅ Image quality looks good!")

                # Show your existing detailed predictions only if we have a reasonable result
                if show_detailed_analysis:
                    # Continue with your existing detailed predictions code here
                    st.markdown("### 📈 Detailed Predictions")

                    # Create two columns for predictions
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Top 3 Predictions")
                        for i, idx in enumerate(pred_indices[:3]):
                            class_name = CLASS_NAMES[idx]
                            confidence = predictions[idx]
                            emoji = CLASS_INFO[class_name]['emoji']
                            desc = CLASS_INFO[class_name]['description']

                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>{emoji} {class_name}</h4>
                                <p><strong>Confidence:</strong> {confidence*100:.2f}%</p>
                                <p><small>{desc}</small></p>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        # Confidence chart
                        fig = create_confidence_chart(predictions, CLASS_NAMES)
                        st.plotly_chart(fig, use_container_width=True)

                    # Continue with your existing advanced analysis section
                    if show_analysis:
                        st.markdown("### 🔬 Advanced Analysis")
                        # ... your existing advanced analysis code here

                # Show helpful suggestions for failed predictions
                elif not show_detailed_analysis:
                    show_helpful_suggestions()

            # Continue with your existing prediction history code
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []

            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction': top_class,
                'confidence': float(top_confidence),
                'filename': uploaded_file.name
            })

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            st.info("Please try with a different image or check the file format.")
            show_helpful_suggestions()

    # Model evaluation section
    st.markdown("---")
    st.markdown("## 📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists('training_history.png'):
            st.markdown("### 📈 Training History")
            st.image('training_history.png', use_container_width=True)
        else:
            st.info("📈 Training history not available. Train the model first.")

    with col2:
        if os.path.exists('confusion_matrix.png'):
            st.markdown("### 🔄 Confusion Matrix")
            st.image('confusion_matrix.png', use_container_width=True)
        else:
            st.info("🔄 Confusion matrix not available. Train the model first.")

    # Prediction history
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.markdown("### 📝 Prediction History")
        history_df = st.session_state.prediction_history

        # Convert to display format
        for record in history_df[-5:]:  # Show last 5 predictions
            st.markdown(f"**{record['timestamp']}** - {record['filename']}: **{record['prediction']}** ({record['confidence']*100:.1f}%)")

    # Footer with enhanced styling
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4>🌼 Blossom AI</h4>
        <p>Built with ❤️ using Streamlit, TensorFlow, and Plotly</p>
        <p><small>Supports: Daisy, lavender, lotus, Sunflower, Tulip</small></p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar information
def setup_sidebar():
    st.sidebar.markdown("## 🚀 How to Use")
    st.sidebar.markdown("""
    1. **Train the model**: `python fileName.py`
    2. **Upload image**: Click 'Browse files' and select a flower image
    3. **Get predictions**: View AI analysis and confidence scores
    4. **Explore**: Enable detailed analysis for more insights
    """)

    st.sidebar.markdown("## 🌸 Supported Flowers")
    for name, info in CLASS_INFO.items():
        st.sidebar.markdown(f"**{info['emoji']} {name}**")
        st.sidebar.markdown(f"*{info['description']}*")

    st.sidebar.markdown("## 💡 Tips")
    st.sidebar.markdown("""
    - Use clear, well-lit images
    - Ensure the flower is the main subject
    - Try different angles if confidence is low
    - Enable image enhancement for better results
    """)

if __name__ == "__main__":
    setup_sidebar()
    main()
