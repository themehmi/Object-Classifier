import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
import io
from PIL import Image

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Vision | Pro", layout="wide")

# ─── MODEL LOADING ──────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model('cifar10_model.keras')

model = get_model()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────
# The file uploader is invisible and floated over the drop zone inside the iframe.
# We use CSS custom properties + media queries so it follows the zone at every
# viewport width.  All magic numbers are replaced by relative / clamp() values.
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; margin: 0 !important; }

    /* ── Invisible overlay uploader ──────────────────────────────────────── */
    .stFileUploader {
        position: fixed;           /* fixed keeps it inside the viewport */
        top: var(--zone-top, 38%); /* JS will set this via a <style> tag   */
        left: 50%;
        transform: translateX(-50%);
        width: min(580px, 90vw);
        height: var(--zone-h, 220px);
        z-index: 999;
        opacity: 0.01;
        pointer-events: all;
    }

    [data-testid="stFileUploaderDropzone"] {
        width: 100% !important;
        height: var(--zone-h, 220px) !important;
        min-height: unset !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── FILE UPLOAD ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

img_path_data   = ""
results_html    = ""
top_prediction_html = ""

if uploaded_file:
    img = Image.open(uploaded_file)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_path_data = (
        f"data:image/png;base64,"
        f"{base64.b64encode(buffered.getvalue()).decode()}"
    )

    # Prediction
    img_resized  = img.resize((32, 32))
    img_array    = image.img_to_array(img_resized) / 255.0
    img_array    = np.expand_dims(img_array, axis=0)
    predictions  = model.predict(img_array)
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    top_indices  = probabilities.argsort()[-3:][::-1]

    master_idx   = top_indices[0]
    master_label = class_names[master_idx]
    master_conf  = round(float(probabilities[master_idx]) * 100, 2)

    top_prediction_html = f"""
    <div class="master-prediction">
        <span class="result-label">Predicted Object</span>
        <span class="master-value">{master_label.upper()}</span>
        <span class="master-conf">{master_conf}% Confidence</span>
        <button class="try-again-btn"
                onclick="window.parent.location.reload();">
            Try Another Image
        </button>
    </div>
    """

    for i in top_indices:
        results_html += f"""
        <div class="result-card">
            <span class="result-label">{class_names[i]}</span>
            <span class="result-value">
                {round(float(probabilities[i]) * 100, 2)}%
            </span>
        </div>
        """

# ─── FULL HTML MIRROR ────────────────────────────────────────────────────────
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600&family=DM+Mono:wght@400;500&display=swap"
      rel="stylesheet">
<style>
  /* ── Design tokens ─────────────────────────────────────────────────────── */
  :root {{
    --bg:      #080808;
    --accent:  #ffffff;
    --sub:     #666666;
    --muted:   #333333;
    --glass:   rgba(255,255,255,0.03);
    --border:  rgba(255,255,255,0.09);
    --green:   #00e87a;

    --r-sm:  16px;
    --r-md:  20px;
    --r-lg:  28px;

    /* Zone geometry – read by Streamlit overlay via postMessage */
    --zone-w: min(580px, 90vw);
    --zone-h: clamp(200px, 32vw, 326px);
  }}

  /* ── Reset ─────────────────────────────────────────────────────────────── */
  *, *::before, *::after {{
    margin: 0; padding: 0;
    box-sizing: border-box;
    font-family: 'DM Sans', sans-serif;
  }}

  html, body {{
    background: var(--bg);
    color: var(--accent);
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* ── Ambient orb ───────────────────────────────────────────────────────── */
  .orb {{
    position: fixed;
    width: clamp(300px, 80vw, 900px);
    height: clamp(300px, 80vw, 900px);
    background: radial-gradient(
        circle,
        rgba(255,255,255,0.06) 0%,
        transparent 70%
    );
    top: -20%;
    left: 50%;
    transform: translateX(-50%);
    z-index: 0;
    filter: blur(80px);
    pointer-events: none;
    animation: breathe 9s ease-in-out infinite alternate;
  }}
  @keyframes breathe {{
    from {{ opacity: .45; transform: translateX(-50%) scale(1);   }}
    to   {{ opacity: .75; transform: translateX(-50%) scale(1.1); }}
  }}

  /* ── Layout ────────────────────────────────────────────────────────────── */
  .page {{
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: clamp(12px, 4vw, 40px);
    gap: clamp(16px, 3vw, 28px);
  }}

  /* ── Header ────────────────────────────────────────────────────────────── */
  header {{
    width: 100%;
    max-width: 860px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }}

  .logo {{
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    font-size: clamp(0.75rem, 2.2vw, 1rem);
    letter-spacing: 0.12em;
    opacity: .85;
  }}

  .header-right {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }}

  .source-link {{
    font-family: 'DM Mono', monospace;
    font-size: clamp(9px, 1.8vw, 11px);
    font-weight: 500;
    letter-spacing: 0.1em;
    color: var(--accent);
    text-decoration: none;
    padding: 7px clamp(10px, 2vw, 16px);
    border-radius: 50px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    transition: background .2s;
  }}
  .source-link:hover {{ background: rgba(255,255,255,0.1); }}

  .system-status {{
    font-family: 'DM Mono', monospace;
    font-size: clamp(9px, 1.8vw, 11px);
    font-weight: 500;
    letter-spacing: 0.08em;
    color: var(--sub);
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--glass);
    padding: 7px clamp(10px, 2vw, 16px);
    border-radius: 50px;
    border: 1px solid var(--border);
  }}

  .status-dot {{
    width: 6px;
    height: 6px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--green);
    flex-shrink: 0;
  }}

  /* ── Hero text ─────────────────────────────────────────────────────────── */
  .hero {{
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }}

  h1 {{
    font-size: clamp(2rem, 8vw, 4rem);
    font-weight: 300;
    line-height: 1;
    background: linear-gradient(175deg, #fff 20%, #444 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
  }}

  .subtitle {{
    font-family: 'DM Mono', monospace;
    font-size: clamp(10px, 2.5vw, 13px);
    color: var(--sub);
    letter-spacing: 0.06em;
  }}

  /* ── Drop zone ─────────────────────────────────────────────────────────── */
  .file-zone {{
    width: var(--zone-w);
    height: var(--zone-h);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 14px;
    background: var(--glass);
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
    transition: border-color .3s;
    cursor: pointer;
  }}
  .file-zone:hover {{ border-color: rgba(255,255,255,0.22); }}

  .scanner-bar {{
    position: absolute;
    top: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #fff, transparent);
    animation: scanMove 1.8s ease-in-out forwards;
    display: {"block" if uploaded_file else "none"};
  }}
  @keyframes scanMove {{
    0%   {{ top: 0%;    opacity: 0; }}
    15%  {{ opacity: 1; }}
    85%  {{ opacity: 1; }}
    100% {{ top: 100%;  opacity: 0; }}
  }}

  .drop-icon {{
    width: clamp(28px, 5vw, 40px);
    height: clamp(28px, 5vw, 40px);
    opacity: .35;
  }}

  .drop-text {{
    font-size: clamp(11px, 2.5vw, 14px);
    color: var(--sub);
  }}

  .upload-btn-ui {{
    padding: clamp(8px, 1.5vw, 12px) clamp(18px, 3vw, 28px);
    background: white;
    color: black;
    border-radius: 50px;
    font-weight: 600;
    font-size: clamp(10px, 2vw, 13px);
    border: none;
    cursor: pointer;
    letter-spacing: 0.04em;
    transition: opacity .2s;
  }}
  .upload-btn-ui:hover {{ opacity: .85; }}

  .preview-img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    border-radius: var(--r-lg);
  }}

  /* ── Master prediction ─────────────────────────────────────────────────── */
  .master-prediction {{
    width: var(--zone-w);
    padding: clamp(18px, 4vw, 32px);
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: var(--r-lg);
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    animation: fadeUp .4s ease both;
  }}
  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0);    }}
  }}

  .result-label {{
    font-family: 'DM Mono', monospace;
    font-size: clamp(9px, 2vw, 11px);
    letter-spacing: 0.12em;
    color: var(--sub);
    text-transform: uppercase;
    display: block;
  }}

  .master-value {{
    display: block;
    font-size: clamp(1.6rem, 6vw, 2.8rem);
    font-weight: 600;
    color: white;
    margin: 4px 0 2px;
    letter-spacing: -0.02em;
  }}

  .master-conf {{
    font-family: 'DM Mono', monospace;
    font-size: clamp(11px, 2.5vw, 14px);
    color: var(--green);
    font-weight: 500;
    display: block;
    margin-bottom: 16px;
  }}

  .try-again-btn {{
    background: transparent;
    color: white;
    border: 1px solid rgba(255,255,255,0.25);
    padding: clamp(7px, 1.5vw, 10px) clamp(24px, 5vw, 44px);
    border-radius: 50px;
    font-size: clamp(9px, 2vw, 11px);
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    transition: background .25s, color .25s;
  }}
  .try-again-btn:hover {{ background: white; color: black; }}

  /* ── Results grid ──────────────────────────────────────────────────────── */
  .results-grid {{
    width: var(--zone-w);
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(clamp(110px, 28vw, 160px), 1fr));
    gap: clamp(10px, 2vw, 16px);
    animation: fadeUp .5s .1s ease both;
  }}

  .result-card {{
    padding: clamp(14px, 3vw, 20px);
    background: var(--glass);
    border-radius: var(--r-md);
    border: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}

  .result-value {{
    font-family: 'DM Mono', monospace;
    font-size: clamp(1rem, 3.5vw, 1.4rem);
    font-weight: 500;
    color: var(--accent);
  }}

  /* ── Thin divider ──────────────────────────────────────────────────────── */
  .divider {{
    width: var(--zone-w);
    height: 1px;
    background: var(--border);
  }}
</style>
</head>
<body>
<div class="orb"></div>

<div class="page">

  <!-- Header -->
  <header>
    <div class="logo">VISION.PRO</div>
    <div class="header-right">
      <a href="https://github.com/themehmi/Object-Detection"
         target="_blank"
         class="source-link">SOURCE CODE</a>
      <div class="system-status">
        <div class="status-dot"></div>
        <span>LIVE ENGINE</span>
      </div>
    </div>
  </header>

  <!-- Hero -->
  <div class="hero">
    <h1>Intelligence</h1>
    <p class="subtitle">Neural classification active.</p>
  </div>

  <!-- Drop zone -->
  <div class="file-zone" id="dropZone">
    <div class="scanner-bar"></div>
    {"<img src='" + img_path_data + "' class='preview-img' alt='uploaded'>" if img_path_data else """
    <svg class="drop-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
    <p class="drop-text">Drop source here</p>
    <button class="upload-btn-ui">Click to Upload</button>
    """}
  </div>

  {top_prediction_html}

  {"<div class='divider'></div>" if results_html else ""}

  <div class="results-grid">{results_html}</div>

</div>

<!-- ── Responsive overlay sync ─────────────────────────────────────────────
     Measure the drop zone position inside the iframe and post it to the
     parent Streamlit page, which injects a <style> tag to move the invisible
     file-uploader overlay into exact alignment.
──────────────────────────────────────────────────────────────────────────── -->
<script>
(function () {{
  function syncOverlay() {{
    const zone = document.getElementById('dropZone');
    if (!zone) return;

    const rect    = zone.getBoundingClientRect();
    const iframeY = window.frameElement
        ? window.frameElement.getBoundingClientRect().top
        : 0;

    const absTop = iframeY + rect.top + window.scrollY;
    const zoneH  = rect.height;
    const zoneW  = rect.width;

    window.parent.postMessage({{
      type    : 'visionZoneSync',
      top     : absTop,
      height  : zoneH,
      width   : zoneW
    }}, '*');
  }}

  // Fire on load and on every resize
  window.addEventListener('load',   syncOverlay);
  window.addEventListener('resize', syncOverlay);

  // Also poll briefly after fonts / images settle
  setTimeout(syncOverlay, 300);
  setTimeout(syncOverlay, 800);
}})();
</script>
</body>
</html>
"""

# ─── OVERLAY SYNC ────────────────────────────────────────────────────────────
# Inject a tiny script into the Streamlit parent page that listens for zone
# coordinates from the iframe and updates the invisible uploader position.
st.markdown("""
<script>
window.addEventListener('message', function(e) {
    if (!e.data || e.data.type !== 'visionZoneSync') return;

    const { top, height, width } = e.data;
    let style = document.getElementById('__visionOverlayStyle');
    if (!style) {
        style = document.createElement('style');
        style.id = '__visionOverlayStyle';
        document.head.appendChild(style);
    }
    style.textContent = `
        .stFileUploader {
            position: fixed !important;
            top:    ${top}px !important;
            left:   50% !important;
            transform: translateX(-50%) !important;
            width:  ${width}px !important;
            height: ${height}px !important;
            z-index: 9999 !important;
            opacity: 0.01 !important;
            pointer-events: all !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            width:  ${width}px  !important;
            height: ${height}px !important;
            min-height: unset !important;
        }
    `;
});
</script>
""", unsafe_allow_html=True)

# ─── RENDER ──────────────────────────────────────────────────────────────────
# Height is driven by content; give extra room on result pages.
iframe_height = 900 if uploaded_file else 620

st.components.v1.html(html_content, height=iframe_height, scrolling=False)
