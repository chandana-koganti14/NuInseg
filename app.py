import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
import streamlit as st
from pathlib import Path


import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from pathlib import Path
import subprocess

# Move page config to ABSOLUTE FIRST Streamlit command
st.set_page_config(
    page_title="NuInSeg Analyzer",
    page_icon=":microscope:",
    layout="wide"
)


# Configure models
MODELS = {
    "YOLOv10n": {
        "path": str(Path(__file__).parent / "models/yolov10n.pt"),
        "description": "Fastest model for quick analysis"
    },
    "YOLOv11n": {
        "path": str(Path(__file__).parent / "models/yolov11n.pt"),
        "description": "Balanced accuracy and speed"
    }
}

# Validate models
for model_name, config in MODELS.items():
    if not Path(config["path"]).exists():
        st.error(f"Critical error: Model file missing at {config['path']}")
        st.stop()

# Model loading with caching
@st.cache_resource
def load_model(path):
    if not Path(path).exists():
        st.error(f"Model file not found: {path}")
        st.stop()
    return YOLO(path)

model = load_model(MODELS[st.session_state.get('selected_model', 'YOLOv10n')]["path"])

# Main interface
st.title("NuInSeg Nuclear Analysis")

# Sidebar controls
with st.sidebar:
    st.header("Model Configuration")
    selected_model = st.selectbox(
        "Select Model",
        options=list(MODELS.keys()),
        format_func=lambda x: f"{x} ({MODELS[x]['description']})",
        key='selected_model'
    )

# File uploader
uploaded_file = st.file_uploader("Upload histology image", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="Original Image")
            conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
            
            if st.button("Analyze Nuclei"):
                with st.spinner("Processing image..."):
                    results = model.predict(image, conf=conf_thresh)
                    st.session_state.results = results[0]
                    st.session_state.boxes = results[0].boxes.data.cpu().numpy()

        if 'results' in st.session_state:
            with col2:
                st.image(
                    st.session_state.results.plot(),
                    caption="Detection Results",
                    channels="BGR"
                )
                
                # Metrics
                nuclei_count = len(st.session_state.boxes)
                avg_confidence = st.session_state.boxes[:, 4].mean()
                
                st.metric("Detected Nuclei", nuclei_count)
                st.metric("Average Confidence", f"{avg_confidence:.2%}")
                
                # Data table
                detection_df = pd.DataFrame(
                    st.session_state.boxes,
                    columns=["x_min", "y_min", "x_max", "y_max", "confidence", "class"]
                )
                st.dataframe(
                    detection_df.sort_values("confidence", ascending=False),
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
else:
    st.info("Please upload an image to begin analysis")



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
        fig1, ax1 = plt.subplots(figsize=(10,6))
        sns.barplot(data=df, x="Model", y="mAP50", palette="viridis", ax=ax1)
        ax1.set_title("mAP50 Comparison")
        st.pyplot(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.barplot(data=df, x="Model", y="mAP50-95", palette="magma", ax=ax2)
        ax2.set_title("mAP50-95 Comparison")
        st.pyplot(fig2)
    
    with viz_tab2:
        fig3, ax3 = plt.subplots(figsize=(10,6))
        sns.barplot(data=df, x="Model", y="GFLOPs", palette="coolwarm", ax=ax3)
        ax3.set_title("Computational Requirements")
        st.pyplot(fig3)
        
        fig4, ax4 = plt.subplots(figsize=(10,6))
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
        
        fig5 = plt.figure(figsize=(8,8))
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
