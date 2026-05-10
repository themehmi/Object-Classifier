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

        

        /* Transparent uploader overlay */

        .stFileUploader {

            position: absolute;

            top: 360px;

            left: 50%;

            transform: translateX(-50%);

            width: 580px;

            height: 326px;

            z-index: 10;

            opacity: 0;

        }

        [data-testid="stFileUploaderDropzone"] { height: 326px; }

    </style>

""", unsafe_allow_html=True)



# --- APP LOGIC ---

uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])



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

    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">

    <style>

        :root {{ --bg: #0a0a0a; --accent: #ffffff; --secondary: #888888; --glass: rgba(255, 255, 255, 0.03); --border: rgba(255, 255, 255, 0.1); }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }}

        body {{ background-color: var(--bg); color: var(--accent); min-height: 100vh; display: flex; flex-direction: column; align-items: center; overflow-x: hidden; }}

        

        .orb {{ position: fixed; width: 80vw; height: 80vw; background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, rgba(0,0,0,0) 70%); top: -10%; left: 50%; transform: translateX(-50%); z-index: -1; filter: blur(80px); animation: breathe 8s infinite alternate ease-in-out; }}

        @keyframes breathe {{ from {{ opacity: 0.5; transform: translateX(-50%) scale(1); }} to {{ opacity: 0.8; transform: translateX(-50%) scale(1.1); }} }}

        

        header {{ width: 100%; max-width: 1000px; padding: 30px 40px; display: flex; justify-content: space-between; align-items: center; }}

        .logo {{ font-weight: 600; letter-spacing: -0.05em; font-size: 1.1rem; opacity: 0.9; }}

        .source-link {{ font-size: 11px; font-weight: 600; letter-spacing: 0.1em; color: #fff; text-decoration: none; padding: 8px 16px; border-radius: 20px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15); }}

        .system-status {{ font-size: 11px; font-weight: 600; color: var(--secondary); display: flex; align-items: center; gap: 10px; background: var(--glass); padding: 8px 16px; border-radius: 20px; border: 1px solid var(--border); }}

        .status-dot {{ width: 6px; height: 6px; background: #00ff88; border-radius: 50%; box-shadow: 0 0 12px #00ff88; }}

        

        main {{ width: 100%; max-width: 900px; display: flex; flex-direction: column; align-items: center; padding: 0 24px; }}

        h1 {{ font-size: clamp(2rem, 6vw, 3.5rem); font-weight: 300; background: linear-gradient(to bottom, #fff, #888); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }}

        

        .file-zone {{ width: 100%; max-width: 580px; aspect-ratio: 16 / 9; border: 1px solid var(--border); border-radius: 24px; display: flex; flex-direction: column; justify-content: center; align-items: center; background: var(--glass); backdrop-filter: blur(10px); position: relative; overflow: hidden; }}

        .upload-btn-ui {{ padding: 12px 24px; background: white; color: black; border-radius: 30px; font-weight: 600; font-size: 0.8rem; margin-top: 15px; border: none; }}

        

        .master-prediction {{ width: 100%; max-width: 580px; margin-top: 24px; padding: 30px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.2); border-radius: 24px; text-align: center; }}

        .master-value {{ display: block; font-size: 2.5rem; font-weight: 600; color: white; margin: 5px 0; }}

        .master-conf {{ font-size: 0.9rem; color: #00ff88; font-weight: 600; display: block; margin-bottom: 20px; }}



        /* Cylindrical Button */

        .try-again-btn {{

            background: transparent; color: white; border: 1px solid rgba(255,255,255,0.3);

            padding: 8px 40px; border-radius: 50px; font-size: 11px; font-weight: 600;

            text-transform: uppercase; letter-spacing: 0.1em; cursor: pointer; transition: 0.3s;

        }}

        .try-again-btn:hover {{ background: white; color: black; }}



        /* Scanner Bar: Now turns off based on a timeout logic in JS or CSS state */

        .scanner-bar {{ 

            position: absolute; top: 0; width: 100%; height: 3px; 

            background: linear-gradient(to right, transparent, #fff, transparent); 

            animation: scanMove 2s ease-in-out forwards; 

            display: {"block" if uploaded_file else "none"}; 

        }}

        @keyframes scanMove {{ 

            0% {{ top: 0%; opacity: 0; }} 

            20% {{ opacity: 1; }} 

            80% {{ opacity: 1; }} 

            100% {{ top: 100%; opacity: 0; }} 

        }}

        

        .results-grid {{ width: 100%; max-width: 580px; margin-top: 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 16px; }}

        .result-card {{ padding: 18px; background: var(--glass); border-radius: 18px; border: 1px solid var(--border); }}

        .result-label {{ font-size: 10px; text-transform: uppercase; color: var(--secondary); display: block; }}

        .preview-img {{ width: 100%; height: 100%; object-fit: contain; }}

    </style>

</head>

<body>

    <div class="orb"></div>

    <header>

        <div class="logo">VISION.PRO</div>

        <div style="display:flex; align-items:center; gap:12px;">

            <a href="https://github.com/themehmi/Object-Detection" target="_blank" class="source-link">SOURCE CODE</a>

            <div class="system-status"><div class="status-dot"></div><span>LIVE ENGINE</span></div>

        </div>

    </header>

    <main>

        <div style="text-align:center; margin-bottom: 30px;">

            <h1>Intelligence</h1>

            <p style="color:var(--secondary); font-size: 0.9rem;">Neural classification active.</p>

        </div>

        <div class="file-zone">

            <div class="scanner-bar"></div>

            {"<img src='" + img_path_data + "' class='preview-img'>" if img_path_data else 

            "<p style='color:var(--secondary); font-size: 0.9rem;'>Drop source here</p><button class='upload-btn-ui'>Click to Upload</button>"}

        </div>

        {top_prediction_html}

        <div class="results-grid">{results_html}</div>

    </main>

</body>

</html>

""", height=1100 if uploaded_file else 750)
