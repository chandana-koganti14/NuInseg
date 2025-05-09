import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
import requests

# ========================
# Data Configuration
# ========================
metrics_data = {
    "Model": ["YOLOv10n", "YOLO11n", "YOLOv12n"],
    "mAP50": [0.773, 0.854, 0.859],
    "mAP50-95": [0.421, 0.495, 0.5],
    "Precision": [0.766, 0.829, 0.835],
    "Recall": [0.697, 0.774, 0.786],
    "GFLOPs": [8.2, 6.3, 6.3],
    "Inference Speed (it/s)": [1.47, 2.92, 4.24],
    "Parameters": ["2,694,806", "2,582,347", "2,556,923"]
}

df = pd.DataFrame(metrics_data)

# ========================
# Pre-trained models configuration
# ========================
MODELS = {
    "YOLOv10n": {
        "path": "https://drive.google.com/uc?export=download&id=1ngcrDc5GLt7FgWnLAXNw6d5CM3xJDJ22",
        "description": "Base model with 8.2 GFLOPs"
    },
    "YOLO11n": {
        "path": "https://drive.google.com/uc?export=download&id=1hKb7x7rRmfBfyBQgN7qk1TtVuJruqUis",
        "description": "Intermediate model with 6.3 GFLOPs"
    },
    "YOLOv12n": {
        "path": "https://drive.google.com/uc?export=download&id=1MvPSOzE-DOH6CkCCLcqXLbPqIy6ov8BR",
        "description": "Optimized model with 6.3 GFLOPs and 4.24it/s speed"
    }
}

# ========================
# Model Loading
# ========================
@st.cache_resource
def load_model(model_name):
    try:
        model_url = MODELS[model_name]["path"]
        # Extract the file ID from the Google Drive URL
        file_id = model_url.split("=")[-1]
        local_path = Path("models") / f"{model_name}_best.pt"  # Use a proper filename

        # Download the model if it doesn't exist locally
        if not local_path.exists():
            os.makedirs(local_path.parent, exist_ok=True)
            response = requests.get(model_url)
            if response.status_code != 200:
                st.error(f"Failed to download model from: {model_url}")
                return None
            with open(local_path, "wb") as f:
                f.write(response.content)

        # Verify the file size
        file_size = local_path.stat().st_size / (1024 * 1024)  # Size in MB
        
        # Load the model
        
        try:
            model = YOLO(local_path)
            st.success(f"Model {model_name} loaded successfully")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None
# ========================
# Streamlit Configuration
# ========================
st.set_page_config(
    page_title="NuInSeg Comprehensive Suite",
    page_icon="🔬",
    layout="wide"
)

# ========================
# Main Interface
# ========================
st.title("🔬 NuInSeg Nuclei Analysis Suite")

# Navigation
analysis_tab, metrics_tab = st.tabs(["Nuclei Detection", "Model Evaluation"])

with analysis_tab:
    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose Detection Model",
        list(MODELS.keys()),
        format_func=lambda x: f"{x} ({MODELS[x]['description']})"
    )

    # Load selected model
    model = load_model(selected_model)

    if model is not None:
        # Main detection interface
        uploaded_file = st.file_uploader("Upload histology image", type=["png", "jpg", "tiff"], key="detection_uploader")

        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(image, caption="Original Image")
                conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, key="detection_conf")

                if st.button("Run Nuclei Analysis"):
                    with st.spinner("Analyzing nuclei..."):
                        image_np = np.array(image)

                # Run prediction
                    try:
                        results = model.predict(source=image_np, conf=conf_thresh, save=False)
                        st.session_state.results = results[0]
                        st.session_state.boxes = results[0].boxes.data.cpu().numpy()
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

            if 'results' in st.session_state:
                with col2:
                    st.image(st.session_state.results.plot(), caption="Detection Results")
                    
                    # Detection metrics
                    nuclei_count = len(st.session_state.boxes)
                    avg_confidence = st.session_state.boxes[:, 4].mean()
                    
                    st.metric("Total Nuclei Detected", nuclei_count)
                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
                    
                    # Dataframe
                    detection_df = pd.DataFrame(
                        st.session_state.boxes, 
                        columns=["x1", "y1", "x2", "y2", "conf", "class"]
                    )
                    st.dataframe(
                        detection_df.sort_values("conf", ascending=False),
                        use_container_width=True
                    )
    else:
        st.error("Failed to load the selected model.")

with metrics_tab:
    # Evaluation metrics dashboard
    st.header("📊 Model Performance Evaluation")
    
    # Key metrics
    st.subheader("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best mAP50", "85.9%", "YOLOv12n")
        st.metric("Top Speed", "4.24 it/s", "YOLOv12n")
    with col2:
        st.metric("Most Precise", "83.5%", "YOLOv12n")
        st.metric("Best Recall", "78.6%", "YOLOv12n")
    with col3:
        st.metric("Lowest GFLOPs", "6.3", "YOLO11n/YOLOv12n")
        st.metric("Fewest Parameters", "2.55M", "YOLOv12n")

    # Interactive table
    st.subheader("Model Comparison Table")
    st.dataframe(
        df.style
        .format({"mAP50": "{:.3f}", "mAP50-95": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}"})
        .highlight_max(subset=["mAP50", "mAP50-95", "Precision", "Recall", "Inference Speed (it/s)"], color="#90EE90")
        .highlight_min(subset=["GFLOPs", "Parameters"], color="#FFCCCB"),
        use_container_width=True
    )

    # Visualizations
    st.subheader("Performance Visualizations")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Accuracy Metrics", "Efficiency Analysis", "Model Breakdown"])
    
    with viz_tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y="mAP50", palette="viridis", ax=ax1)
        ax1.set_title("mAP50 Comparison")
        st.pyplot(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y="mAP50-95", palette="magma", ax=ax2)
        ax2.set_title("mAP50-95 Comparison")
        st.pyplot(fig2)
    
    with viz_tab2:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y="GFLOPs", palette="coolwarm", ax=ax3)
        ax3.set_title("Computational Requirements")
        st.pyplot(fig3)
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df, x="Model", y="Inference Speed (it/s)", marker="o", color="#FF6B6B", ax=ax4)
        ax4.set_title("Inference Speed Comparison")
        st.pyplot(fig4)
    
    with viz_tab3:
        selected_model = st.selectbox("Select Model for Detailed Analysis", df["Model"], key="metrics_model")
        model_data = df[df["Model"] == selected_model].iloc[0]
        
        st.subheader(f"🧬 {selected_model} Specifications")
        cols = st.columns(4)
        cols[0].metric("mAP50", f"{model_data['mAP50']*100:.1f}%")
        cols[1].metric("mAP50-95", f"{model_data['mAP50-95']*100:.1f}%")
        cols[2].metric("Precision", f"{model_data['Precision']*100:.1f}%")
        cols[3].metric("Recall", f"{model_data['Recall']*100:.1f}%")
        
        # Radar chart
        st.subheader("Performance Radar Chart")
        categories = ["mAP50", "mAP50-95", "Precision", "Recall", "Inference Speed (it/s)"]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig5 = plt.figure(figsize=(8, 8))
        ax5 = fig5.add_subplot(111, polar=True)
        
        for idx, model in df.iterrows():
            values = model[categories].tolist()
            values += values[:1]
            ax5.plot(angles, values, label=model["Model"])
            ax5.fill(angles, values, alpha=0.25)
        
        ax5.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        st.pyplot(fig5)

    # Data export
    st.subheader("Data Export")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Full Metrics CSV",
        csv,
        "yolo_metrics.csv",
        "text/csv",
        key='download-csv'
    )

# ========================
# Custom Styling
# ========================
st.markdown("""
<style>
    .stMetric {
        border: 1px solid #e1e4e8;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .stDataFrame {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)
