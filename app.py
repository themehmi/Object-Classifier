import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
import io
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Vision | Pro", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def get_model():
    return load_model('cifar10_model.keras')

model = get_model()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- CSS HACK: OVERLAY UPLOADER & HIDE UI ---
st.markdown("""
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container { padding: 0 !important; }
        
        /* Transparent uploader overlay: Responsive Positioning */
        .stFileUploader {
            position: absolute;
            top: 250px; /* Aligned with the visual main section */
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 580px;
            height: 326px;
            z-index: 100;
            opacity: 0;
        }
        [data-testid="stFileUploaderDropzone"] { 
            height: 326px !important; 
        }

        /* Adjust uploader position on mobile */
        @media (max-width: 768px) {
            .stFileUploader { top: 200px; height: 250px; }
        }
    </style>
""", unsafe_allow_html=True)

# --- APP LOGIC ---
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

img_path_data = ""
results_html = ""
top_prediction_html = ""

if uploaded_file:
    img = Image.open(uploaded_file)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
        
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_path_data = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

    # Prediction Logic
    img_resized = img.resize((32, 32))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    top_indices = probabilities.argsort()[-3:][::-1]
    
    # Master Prediction
    master_idx = top_indices[0]
    master_label = class_names[master_idx]
    master_conf = round(float(probabilities[master_idx]) * 100, 2)
    
    # Reset Button styled as a horizontal cylinder
    top_prediction_html = f"""
    <div class="master-prediction">
        <span class="result-label">Predicted Object</span>
        <span class="master-value">{master_label.upper()}</span>
        <span class="master-conf">{master_conf}% Confidence</span>
        <button class="try-again-btn" onclick="window.parent.location.reload();">Try Another Image</button>
    </div>
    """
    
    # Results Grid
    for i in top_indices:
        results_html += f"""
        <div class="result-card">
            <span class="result-label">{class_names[i]}</span>
            <span class="result-value">{round(float(probabilities[i]) * 100, 2)}%</span>
        </div>
        """

# --- THE FULL HTML MIRROR ---
st.components.v1.html(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        :root {{ 
            --bg: #0a0a0a; 
            --accent: #ffffff; 
            --secondary: #888888; 
            --glass: rgba(255, 255, 255, 0.03); 
            --border: rgba(255, 255, 255, 0.1); 
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }}
        body {{ 
            background-color: var(--bg); 
            color: var(--accent); 
            min-height: 100vh; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            overflow-x: hidden; 
            padding-bottom: 50px;
        }}
        
        /* Orb scaling */
        .orb {{ 
            position: fixed; 
            width: 150vw; 
            height: 150vw; 
            background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, rgba(0,0,0,0) 70%); 
            top: -20%; 
            z-index: -1; 
            filter: blur(80px); 
        }}
        
        header {{ 
            width: 100%; 
            max-width: 1000px; 
            padding: 20px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
        }}
        
        main {{ 
            width: 100%; 
            max-width: 900px; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            padding: 0 15px; 
        }}
        
        h1 {{ 
            font-size: clamp(2.5rem, 10vw, 4rem); 
            font-weight: 300; 
            background: linear-gradient(to bottom, #fff, #888); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }}

        /* Responsive File Zone */
        .file-zone {{ 
            width: 100%; 
            max-width: 580px; 
            aspect-ratio: 16 / 9; 
            border: 1px solid var(--border); 
            border-radius: 24px; 
            background: var(--glass); 
            backdrop-filter: blur(10px); 
            position: relative; 
            overflow: hidden; 
        }}

        .master-prediction {{ 
            width: 100%; 
            max-width: 580px; 
            margin-top: 24px; 
            padding: clamp(15px, 5vw, 30px); 
            background: rgba(255,255,255,0.05); 
            border: 1px solid rgba(255,255,255,0.2); 
            border-radius: 24px; 
        }}
        
        .master-value {{ 
            font-size: clamp(1.5rem, 8vw, 2.5rem); 
            font-weight: 600; 
        }}

        /* Results Grid Scaling */
        .results-grid {{ 
            width: 100%; 
            max-width: 580px; 
            margin-top: 20px; 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); 
            gap: 12px; 
        }}

        /* Mobile specific adjustments */
        @media (max-width: 480px) {{
            header {{ padding: 15px; }}
            .logo {{ font-size: 0.9rem; }}
            .source-link {{ display: none; }} /* Hide source on very small screens to save space */
            .file-zone {{ aspect-ratio: 4 / 3; }}
        }}
    </style>
</head>
...
""", height=1200 if uploaded_file else 800)
