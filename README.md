# NuInSeg Nuclei Segmentation and Evaluation

This repository contains a Streamlit-based interactive application for **nuclei segmentation** using multiple YOLO models (YOLOv10n, YOLO11n, YOLOv12n) trained on the NuInseg dataset. The app allows users to:

- Upload histology images and run nuclei detection using selected YOLO models.
- Visualize detection results with bounding boxes and confidence scores.
- Explore detailed model evaluation metrics including mAP, precision, recall, GFLOPs, and inference speed.
- Compare model performances via interactive charts and tables.

---

## Features

- **Multi-model support:** Select from YOLOv10n, YOLO11n, and YOLOv12n pre-trained models.
- **Interactive visualization:** View original and annotated images side-by-side.
- **Performance dashboard:** Detailed metrics and radar charts for model comparison.
- **Data export:** Download evaluation metrics in CSV format.
- **Confidence threshold adjustment:** Customize detection sensitivity.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning)

### Clone the repository

```bash
git clone https://github.com/chandana-koganti14/NuInseg.git
cd NuInseg
```
---

## Usage

### Run locally

```bash
streamlit run app.py
```
- Open your browser at `http://localhost:8501`
- Select a model from the sidebar
- Upload histology images (`png`, `jpg`, `tiff`)
- Adjust confidence threshold and click **Run Nuclei Analysis**
- Explore detection results and evaluation metrics

---

## Model Weights

- Model weights are included in the `models/` directory (if size permits).
- For large weights, the app downloads them automatically from hosted URLs.
- You can replace or update weights by modifying the paths or URLs in `app.py`.

---

## Deployment

You can deploy this app easily on [Streamlit Community Cloud](https://share.streamlit.io/):

1. Push your code to a GitHub repository.
2. Connect the repo on Streamlit Community Cloud.
3. Specify `app.py` as the main file.
4. Deploy and share the generated public URL.

---

## Requirements

Key dependencies are listed in `requirements.txt`, including:

- `streamlit==1.31.0`
- `ultralytics==8.3.40`
- `opencv-python-headless==4.10.0.84`
- `pandas==2.0.3`
- `numpy==1.24.4`
- `matplotlib==3.7.5`
- `seaborn==0.13.2`
- `Pillow==10.4.0`
- `torch==2.4.1`
- `torchvision==0.19.1`

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, features, or improvements.

---

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- NuInseg Dataset contributors

---

*Happy nuclei detecting!* 🧬🔬



 
