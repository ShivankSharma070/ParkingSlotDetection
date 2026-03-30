"""
Smart Parking Lot Detection System
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import json
import tempfile
from pathlib import Path
from detector import ParkingSlotDetector
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Smart Parking Detection",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 1.2rem !important; max-width: 1100px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}

/* ── Header ── */
.app-header { padding: 0.6rem 0 0.2rem; }
.app-header h1 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 2px;
}

/* ── Section Labels ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1rem 0 0.5rem;
}

/* ── Thumbnail Grid ── */
.thumb-col {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.thumb-wrap {
    width: 100%;
    height: 150px;
    background: #020817;
    border: 1px solid #1e293b;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: border-color 0.2s ease;
    cursor: pointer;
}
.thumb-wrap:hover { border-color: #334155; }
.thumb-wrap.selected {
    border-color: #3b82f6;
    box-shadow: 0 0 0 1px #3b82f6;
}
.thumb-wrap img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}
.thumb-name {
    font-size: 0.75rem;
    color: #94a3b8;
    text-align: center;
    padding: 8px 0 12px 0;
    margin-top: 4px;
}
.thumb-name.active { color: #3b82f6; }

/* ── Stats Row ── */
.stats-row {
    display: flex;
    gap: 10px;
    margin: 0.8rem 0;
}
.stat-card {
    flex: 1;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 10px;
    text-align: center;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.2;
}
.stat-label {
    font-size: 0.62rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 3px;
}
.stat-card.occ .stat-value { color: #ef4444; }
.stat-card.free .stat-value { color: #22c55e; }
.stat-card.rate .stat-value { color: #f59e0b; }

/* ── Detection Table ── */
.det-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
}
.det-table th {
    text-align: left;
    padding: 8px 10px;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.65rem;
    border-bottom: 1px solid #1e293b;
}
.det-table td {
    padding: 7px 10px;
    border-bottom: 1px solid #0f172a;
    color: #cbd5e1;
}
.det-table tr:hover td { background: rgba(255,255,255,0.02); }
.badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
}
.badge-car { background: rgba(239,68,68,0.15); color: #ef4444; }
.badge-free { background: rgba(34,197,94,0.15); color: #22c55e; }

/* ── Model Card ── */
.model-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}
.model-card h4 {
    margin: 0 0 8px;
    color: #94a3b8;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.model-card .info-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    font-size: 0.75rem;
}
.model-card .info-key { color: #64748b; }
.model-card .info-val { color: #f8fafc; font-weight: 500; }

/* ── Legend ── */
.legend-row { display: flex; gap: 14px; margin: 4px 0; }
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: #94a3b8;
}
.legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
}

/* ── Buttons ── */
.stButton > button {
    background: #3b82f6 !important;
    color: #fff !important;
    border: none !important;
    padding: 0.45rem 1.2rem !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    transition: opacity 0.2s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
}
.stButton > button:disabled {
    background: transparent !important;
    color: #3b82f6 !important;
    border: 1px solid #3b82f6 !important;
}

/* ── Select buttons smaller ── */
.thumb-col .stButton > button {
    font-size: 0.7rem !important;
    padding: 0.25rem 0.5rem !important;
}

/* ── Upload ── */
[data-testid="stFileUploader"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
}
[data-testid="stFileUploader"] > section {
    padding: 1rem !important; /* Ensure internal padding is adequate */
}

/* ── Images ── */
div[data-testid="stImage"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #1e293b;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 0.78rem !important;
    padding: 6px 12px !important;
    color: #64748b !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f8fafc !important;
}

/* ── Misc ── */
hr { border-color: #1e293b !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───────────────────────────────────────────────────
DEFAULT_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")  # Empty by default without .env
DATA_DIR = Path("data")


def find_sample_images():
    """Discover images in the data/ folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return []
    return sorted(
        [str(f) for f in DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() in exts]
    )


def make_thumbnail(img_path, size=(300, 150)):
    """Create uniform thumbnail: original aspect ratio on black background."""
    img = Image.open(img_path).convert("RGB")
    thumb = Image.new("RGB", size, (10, 10, 10))
    img.thumbnail(size, Image.LANCZOS)
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    thumb.paste(img, offset)
    return thumb


def render_stats(s):
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">{s['total_slots']}</div>
            <div class="stat-label">Total Slots</div>
        </div>
        <div class="stat-card occ">
            <div class="stat-value">{s['occupied']}</div>
            <div class="stat-label">Occupied</div>
        </div>
        <div class="stat-card free">
            <div class="stat-value">{s['free']}</div>
            <div class="stat-label">Available</div>
        </div>
        <div class="stat-card rate">
            <div class="stat-value">{s['occupancy_rate']}%</div>
            <div class="stat-label">Occupancy Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_detection_table(predictions):
    preds = predictions.get("predictions", [])
    if not preds:
        return
    rows = ""
    for i, p in enumerate(sorted(preds, key=lambda x: -x["confidence"]), 1):
        cls = p["class"]
        badge = "badge-car" if cls == "car" else "badge-free"
        label = "Occupied" if cls == "car" else "Vacant"
        rows += f"""<tr>
            <td>{i}</td>
            <td><span class="badge {badge}">{label}</span></td>
            <td>{p['confidence']:.1%}</td>
            <td>{int(p['x'])}, {int(p['y'])}</td>
            <td>{int(p['width'])}×{int(p['height'])}</td>
        </tr>"""
    st.markdown(f"""
    <table class="det-table">
        <thead><tr><th>#</th><th>Status</th><th>Conf</th><th>Center</th><th>Size</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown("#### Detection Parameters")

    confidence = st.slider("Confidence", 0, 100, 40, format="%d%%")
    overlap = st.slider("Overlap (IoU)", 0, 100, 30, format="%d%%")

    st.markdown("---")

    st.markdown("""
    <div class="model-card">
        <h4>Model</h4>
        <div class="info-row"><span class="info-key">Architecture</span><span class="info-val">YOLOv8s</span></div>
        <div class="info-row"><span class="info-key">Task</span><span class="info-val">Object Detection</span></div>
        <div class="info-row"><span class="info-key">Classes</span><span class="info-val">car, free</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="legend-row" style="margin-top:12px; margin-bottom:12px;">
        <div class="legend-item">
            <div class="legend-dot" style="background:#ef4444;"></div>
            <span>Occupied</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#22c55e;"></div>
            <span>Vacant</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # API Key button state
    if "show_api" not in st.session_state:
        st.session_state.show_api = False
        
    if st.button("🔑 Custom API Key"):
        st.session_state.show_api = not st.session_state.show_api
        
    if st.session_state.show_api:
        api_key_input = st.text_input(
            "Enter API Key", 
            type="password",
            placeholder="SWKtw9IhE...",
            label_visibility="collapsed"
        )
        api_key = api_key_input.strip() if api_key_input.strip() else DEFAULT_API_KEY
    else:
        api_key = DEFAULT_API_KEY


# ─── Main ─────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>Smart Parking Detection</h1>
    <p>Parking slot occupancy detection using YOLOv8s</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Image Selection ──
sample_images = find_sample_images()

if "selected_img" not in st.session_state:
    st.session_state.selected_img = sample_images[0] if sample_images else None
if "img_source" not in st.session_state:
    st.session_state.img_source = "sample"

st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)

if sample_images:
    cols = st.columns(len(sample_images), gap="small")
    for idx, img_path in enumerate(sample_images):
        is_selected = (
            st.session_state.img_source == "sample"
            and st.session_state.selected_img == img_path
        )
        with cols[idx]:
            # Create uniform thumbnail with black letterboxing
            thumb = make_thumbnail(img_path, size=(300, 150))
            name = Path(img_path).name
            st.image(thumb, use_container_width=True)

            # Filename label
            label_cls = "active" if is_selected else ""
            prefix = "● " if is_selected else ""
            st.markdown(
                f'<div class="thumb-name {label_cls}">{prefix}{name}</div>',
                unsafe_allow_html=True,
            )

            if st.button(
                "✓ Selected" if is_selected else "Select",
                key=f"sel_{idx}",
                use_container_width=True,
                disabled=is_selected,
            ):
                st.session_state.selected_img = img_path
                st.session_state.img_source = "sample"
                st.rerun()

# Upload option
st.markdown(
    '<div class="section-label" style="margin-top:0.5rem;">Or upload an image</div>',
    unsafe_allow_html=True,
)
uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)
if uploaded is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(uploaded.getvalue())
    tmp.close()
    st.session_state.selected_img = tmp.name
    st.session_state.img_source = "upload"

image_path = st.session_state.selected_img

st.markdown("---")

# ── Run Detection ──
col_l, col_c, col_r = st.columns([1, 1, 1])
with col_c:
    run_btn = st.button("Detect Parking Slots", use_container_width=True)

if run_btn:
    if not image_path:
        st.error("Select or upload an image first.")
        st.stop()

    detector = ParkingSlotDetector(api_key)

    with st.status("Running detection…", expanded=True) as status:
        st.write("Loading image…")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            st.error("Failed to read image.")
            st.stop()
        h, w = image_bgr.shape[:2]
        time.sleep(0.2)
        st.write(f"✓ Image loaded — {w}×{h}px")

        st.write("Running YOLOv8s inference…")
        try:
            predictions = detector.predict(image_path, confidence=confidence, overlap=overlap)
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()
        inf_time = predictions.get("_inference_time", 0)
        n = len(predictions.get("predictions", []))
        st.write(f"✓ {n} detections in {inf_time:.2f}s")

        st.write("Post-processing…")
        stats = detector.get_statistics(predictions)
        time.sleep(0.15)
        st.write(f"✓ {stats['occupied']} occupied · {stats['free']} vacant")

        st.write("Generating annotations…")
        annotated_bgr = detector.annotate(image_bgr, predictions, show_labels=False)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        time.sleep(0.1)
        st.write("✓ Done")

        status.update(label="Detection complete", state="complete", expanded=False)

    # ── Results ──
    render_stats(stats)

    res_tab1, res_tab2 = st.tabs(["Annotated", "Original"])
    with res_tab1:
        st.image(annotated_rgb, use_container_width=True)
    with res_tab2:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    det_tab1, det_tab2 = st.tabs(["Detections", "JSON"])
    with det_tab1:
        render_detection_table(predictions)
    with det_tab2:
        clean = {k: v for k, v in predictions.items() if not k.startswith("_")}
        st.code(json.dumps(clean, indent=2), language="json")

    st.caption(f"{n} detections · {inf_time:.2f}s · conf ≥ {confidence}% · overlap ≤ {overlap}%")
