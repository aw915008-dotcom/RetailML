import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime

# =============================
# Config
# =============================
st.set_page_config(page_title="🧠 RetailML Dashboard", layout="wide", initial_sidebar_state="expanded")

# =============================
# Dark Glass Theme (CSS)
# =============================
st.markdown(
    """
    <style>
    /* ---------- Background + font ---------- */
    :root {
        --bg: #0b0f14;
        --card: rgba(255,255,255,0.03);
        --glass: rgba(255,255,255,0.04);
        --muted: #aeb6c2;
        --accent: #5ee0ff;
        --primary: #1e88e5;
    }
    .stApp {
        background: linear-gradient(180deg, #071018 0%, #0a1116 100%);
        color: #e6eef3;
        font-family: "Inter", sans-serif;
    }
    /* Header / Logo */
    .header {
        display:flex;
        align-items:center;
        gap:16px;
    }
    .logo {
        font-weight:700;
        font-size:20px;
        color: var(--accent);
    }
    .sublogo {
        color:var(--muted);
        font-size:12px;
    }

    /* Glass cards */
    .glass {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.04);
        border-radius:12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(2,6,12,0.6);
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] .css-1lcbmhc {
        background: linear-gradient(180deg, #071018 0%, #071018 100%);
    }
    /* Inputs */
    input, textarea {
        background: #0f1316 !important;
        color: #e6eef3 !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #1E88E5, #1565C0) !important;
        color: white !important;
        border-radius: 10px;
        padding: 8px 14px;
        font-weight:600;
    }
    .stButton > button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
    }

    /* Metric color */
    [data-testid="stMetricValue"] {
        color: #A7F3D0 !important;
    }

    /* Small text helper */
    .muted {
        color: var(--muted);
        font-size:12px;
    }

    /* make images responsive */
    img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Constants
# =============================
GATEWAY_URL = "http://127.0.0.1:5050/gateway"
SERVICES = {
    "Customer Segmentation (Labeling)": "customer_segmentation_labeling",
    "Customer Segmentation (Classification)": "customer_segmentation_classification",
    "Sentiment Analysis": "sentiment_analysis",
    "Product Purchase Prediction": "product_purchase_prediction",
    "Branch Profit Prediction": "branch_profit_prediction",
    "Demand Forecasting": "demand_forecasting"
}

# =============================
# Sidebar - Navigation + Status + History
# =============================
with st.sidebar:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;align-items:center;gap:12px'>", unsafe_allow_html=True)
    st.markdown("<div class='logo'>🧠 RetailML</div>", unsafe_allow_html=True)
    st.markdown("<div class='sublogo'>Gateway Control Center</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("Services")
    selected = st.radio("Choose service", list(SERVICES.keys()))
    st.markdown("---")
    st.subheader("Gateway")
    st.write("URL")
    st.code(GATEWAY_URL, language="bash")
    st.markdown("---")
    if "history" not in st.session_state:
        st.session_state.history = []
    st.subheader("Request History")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-6:]):
            st.write(f"- **{h['service']}** • {h['time']} • {h['status']}")
    else:
        st.write("No history yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== Main layout =====
st.markdown("<div class='glass' style='padding:18px;margin-bottom:12px'>", unsafe_allow_html=True)
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown(f"<div class='header'><div style='font-size:22px;font-weight:700;color:#E6F9FF'>🧠 RetailML Dashboard</div>"
                f"<div style='margin-left:8px' class='muted'>— Gateway control & EDA</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='muted' style='text-align:right'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ===== Input area =====
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader(f"Service ▶️  {selected}")
st.markdown("<div class='muted'>Configure inputs for the selected service and click <b>Run</b>.</div>", unsafe_allow_html=True)
st.markdown("---")

# dynamic request payload
payload = {"service": SERVICES[selected]}

# per-service inputs
if "prediction" in payload["service"] or "forecasting" in payload["service"]:
    st.markdown("### Prediction Input (two sample rows)")
    r1c1, r1c2, r2c1, r2c2 = st.columns(4)
    total1 = r1c1.number_input("TotalAmount (Row 1)", value=30000.0, step=100.0)
    price1 = r1c2.number_input("UnitPrice (Row 1)", value=150.0, step=1.0)
    total2 = r2c1.number_input("TotalAmount (Row 2)", value=70000.0, step=100.0)
    price2 = r2c2.number_input("UnitPrice (Row 2)", value=200.0, step=1.0)
    payload["new_data"] = [
        {"TotalAmount": float(total1), "UnitPrice": float(price1)},
        {"TotalAmount": float(total2), "UnitPrice": float(price2)},
    ]

elif "sentiment" in payload["service"]:
    st.markdown("### Enter one review per line")
    reviews_text = st.text_area("Reviews", placeholder="Great product...\nWorst ever...\nOK experience...")
    reviews = [r.strip() for r in reviews_text.splitlines() if r.strip()]
    payload["new_reviews"] = reviews

elif "segmentation" in payload["service"]:
    st.markdown("### Clustering Configuration")
    n_clusters = st.number_input("Number of clusters", min_value=2, max_value=20, value=3, step=1)
    payload["params"] = {"clusters": int(n_clusters)}

# actions
st.markdown("---")
run_col, clear_col = st.columns([1, 1])
with run_col:
    run_clicked = st.button("🚀 Run via Gateway")
with clear_col:
    clear_clicked = st.button("🧹 Clear Inputs")

if clear_clicked:
    # clear session factories (not form state)
    if "last_response" in st.session_state:
        del st.session_state["last_response"]
    st.experimental_rerun()

# ===== Execute request =====
if run_clicked:
    with st.spinner(f"Calling {payload['service']} ..."):
        try:
            resp = requests.post(GATEWAY_URL, json=payload, timeout=45)
            status = "success" if resp.status_code == 200 else f"error {resp.status_code}"
            entry = {"service": payload["service"], "time": datetime.now().strftime("%H:%M:%S"), "status": status}
            st.session_state.history.append(entry)

            if resp.status_code != 200:
                st.error(f"API returned status {resp.status_code}")
                st.code(resp.text)
            else:
                api_res = resp.json().get("response", {})
                st.session_state.last_response = api_res

                # ===== Display top-level metrics =========
                st.success("✅ Request succeeded")
                metric_keys = [k for k, v in api_res.items() if isinstance(v, (int, float))]
                list_keys = [k for k, v in api_res.items() if isinstance(v, list)]
                text_keys = [k for k, v in api_res.items() if isinstance(v, str) and not k.startswith("plot_")]

                # Metrics row
                if metric_keys:
                    cols = st.columns(min(3, len(metric_keys)))
                    for i, k in enumerate(metric_keys[:3]):
                        cols[i].metric(label=k, value=round(api_res[k], 4))

                # Strings & lists
                for k in text_keys:
                    st.markdown(f"**{k}**")
                    st.write(api_res[k])

                for k in list_keys:
                    st.markdown(f"**{k}**")
                    st.write(api_res[k])

                # ===== EDA images (base64) =====
                plot_keys = [k for k in api_res.keys() if "plot" in k and isinstance(api_res[k], str)]
                if plot_keys:
                    st.markdown("### 📊 EDA Visualizations")
                    cols = st.columns(len(plot_keys))
                    for i, pk in enumerate(plot_keys):
                        try:
                            b64 = api_res[pk]
                            img = Image.open(BytesIO(base64.b64decode(b64)))
                            cols[i].image(img, caption=pk.replace("_", " ").title(), use_column_width=True)
                        except Exception as e:
                            st.write(f"Could not render {pk}: {e}")

        except requests.exceptions.RequestException as e:
            entry = {"service": payload["service"], "time": datetime.now().strftime("%H:%M:%S"), "status": "connection_error"}
            st.session_state.history.append(entry)
            st.error(f"Connection error: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ===== Show last response for inspection =====
st.markdown("<div style='margin-top:12px' class='glass'>", unsafe_allow_html=True)
st.subheader("Last Response (raw) — for debugging")
if "last_response" in st.session_state:
    with st.expander("Click to view JSON-like response"):
        st.json(st.session_state.last_response)
else:
    st.write("No response yet. Run a request to populate this area.")
st.markdown("</div>", unsafe_allow_html=True)
