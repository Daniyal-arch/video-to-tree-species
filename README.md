# Video-to-Tree-Species

Detect, segment, and identify tree species in videos using state-of-the-art computer vision (GroundedSAM2, GroundingDINO) and the PlantNet API.

---

## 🚀 Features

- **Frame Extraction:** Extracts frames from input videos.
- **Tree Detection & Segmentation:** Uses GroundedSAM2 and GroundingDINO for robust tree detection and segmentation.
- **Species Identification:** Identifies tree species in each frame using the PlantNet API.
- **Tracking:** Tracks detected trees across frames.
- **Comprehensive Output:** Saves annotated videos, masks, JSON metadata, and summary statistics.

---

## 🗂️ Project Structure

```
video-to-tree-species/
├── notebooks/
│   └── Trees_Dataset_Creation.ipynb      # Example workflow and experiments
├── src/
│   └── tree_detection.py                 # Main detection and identification pipeline
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
└── .gitignore                            # Files to ignore in git
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Daniyal-arch/video-to-tree-species.git
cd video-to-tree-species
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download and Set Up Models

- **GroundedSAM2** and **GroundingDINO** must be set up as described in their official repositories.
- Download required checkpoints and place them in the appropriate directories (see notebook for details).

### 4. Configure PlantNet API

- Get your free API key from [PlantNet](https://my.plantnet.org/).
- Set your API key in your code or as an environment variable.

---

## 📝 Usage

### **A. Notebook Workflow**

- Open `notebooks/Trees_Dataset_Creation.ipynb` for a step-by-step workflow, including setup, frame extraction, and running the pipeline.

### **B. Python Script**

You can use the main pipeline directly:

```python
from src.tree_detection import IntegratedTreeDetectionSystem

tree_system = IntegratedTreeDetectionSystem(
    plantnet_api_key="YOUR_API_KEY",
    plantnet_project="all",  # or "weurope", "canada", "australia"
    video_start_time=1751270034249,  # or your video timestamp
    video_fps=30.0
)

tree_system.process_video_with_species_identification(
    video_dir="frames/",
    output_dir="./outputs_with_species",
    text_prompt="tree.",
    step=100,
    confidence_threshold=0.1
)
```

---

## 📦 Output

- **outputs_with_species/**
    - `mask_data/` — Numpy masks for each frame
    - `json_data/` — Per-frame detection and species metadata
    - `result/` — Annotated result images
    - `species_data/` — Per-frame and summary JSONs
    - `tree_detection_with_species.mp4` — Annotated output video

---

## 🧩 Dependencies

- Python 3.8+
- torch, torchvision, torchaudio
- opencv-python
- transformers
- diffusers
- accelerate
- supervision
- Pillow
- matplotlib
- scikit-image
- pycocotools
- timm
- huggingface_hub
- sahi
- dds-cloudapi-sdk
- setuptools, wheel, hydra-core, addict, yapf, iopath, requests

See `requirements.txt` for the full list.

---

## 📝 Citation

If you use this code, please also cite the original [GroundedSAM2](https://github.com/IDEA-Research/Grounded-SAM-2), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [PlantNet](https://my.plantnet.org/) projects.

---

## 👤 Maintainer

[Daniyal-arch](https://github.com/Daniyal-arch)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
