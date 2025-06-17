# 🛠️ PCB Defect Detection: YOLOv8 vs Faster R-CNN vs EfficientDet

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-blue)](https://github.com/ultralytics/ultralytics)
[![TensorFlow Hub](https://img.shields.io/badge/TensorFlow%20Hub-EfficientDet-yellow)](https://tfhub.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project compares the performance of **YOLOv8**, **Faster R-CNN**, and **EfficientDet** object detection models for identifying defects on **Printed Circuit Boards (PCBs)**. It is designed to run on **Kaggle Notebooks** with GPU acceleration.

---

## 🚀 Highlights

- 🔍 **Model Benchmarking** – YOLOv8, Faster R-CNN, and EfficientDet compared for defect detection accuracy and inference speed.
- 🧠 **Custom Training** – YOLOv8 and Faster R-CNN trained on a labeled PCB dataset.
- 🧪 **EfficientDet Inference** – Loaded from TF Hub for benchmarking.
- 📊 **Evaluation Metrics** – `mAP@50`, `mAP@50-95`, and inference time analyzed.
- 📈 **Visualizations** – Training curves, mAP plots, confusion matrix, prediction samples.
- 🔁 **Reproducible** – Deterministic results with seed settings.
- 💻 **Kaggle-Ready** – Optimized for Kaggle GPUs and notebook structure.

---

## 📂 Dataset

- **Name**: PCB Defects Dataset (PASCAL VOC format)
- **Classes** (6): `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`
- **Total Images**: 693  
    - Train: 485  
    - Validation: 104  
    - Test: 104  
- **Annotations**: Pascal VOC XML

> 📌 Dataset used was added via Kaggle’s "+ Add Data" option. A similar one can be found [here](https://www.kaggle.com/datasets/). Ensure it matches your directory structure.

---

## 🧠 Models

### ✅ YOLOv8 (Ultralytics)
- Fast and accurate object detection.
- Pre-trained weights: `yolov8s.pt`
- Framework: [Ultralytics](https://github.com/ultralytics/ultralytics)

### ✅ Faster R-CNN (TorchVision)
- Two-stage detector known for high precision.
- Backbone: ResNet-50 with FPN.
- Framework: `torchvision.models.detection`

### ✅ EfficientDet D0 (TensorFlow Hub)
- Lightweight and scalable detector.
- Loaded from: `https://tfhub.dev/tensorflow/efficientdet/d0/1`
- Inference only (no fine-tuning in this project)

---

## ⚙️ Setup & Installation

Run on **Kaggle Notebook**:

1. Add the PCB dataset using "+ Add Data".
2. Enable GPU (T4 / P100) from notebook settings.
3. Install dependencies:

```bash
!pip install -U ultralytics
!pip install opencv-python matplotlib pandas pyyaml seaborn tqdm scikit-learn
!pip install torchmetrics pycocotools
!pip install tensorflow tensorflow_hub
!pip install ipywidgets
```

### 📌 Key Configurations

| Parameter        | Value |
|------------------|--------|
| IMG_SIZE_YOLO    | 640    |
| IMG_SIZE_RCNN    | 512    |
| EPOCHS_YOLO      | 50     |
| EPOCHS_RCNN      | 30     |
| BATCH_SIZE_YOLO  | 16     |
| BATCH_SIZE_RCNN  | 4      |
| SEED             | 42     |
| DEVICE           | CUDA (if available) |

---

## 📊 Results

| Model         | mAP@50-95 (Test) | mAP@50 (Test) | Inference Time (ms/img) | Notes |
|---------------|------------------|---------------|---------------------------|-------|
| YOLOv8s       | 0.4944           | 0.9365        | ~10.0 (GPU)               | Trained from pre-trained weights |
| Faster R-CNN  | 0.1734           | 0.4604        | ~63.11 (GPU)              | Trained with ResNet50 FPN |
| EfficientDet D0 | N/A            | N/A           | ~5536.26 (CPU)            | Inference only (not fine-tuned) |

---

## 📸 Visualizations

- 📉 `training_loss_vs_epoch.png`
- 📈 `validation_map_vs_epoch.png`
- 📊 `final_metrics_comparison.png`
- 🔀 `yolov8_confusion_matrix_ultralytics.png`
- 🖼️ Prediction examples for each model on test images

> 📁 Plots saved in: `/kaggle/working/evaluation_plots/`

---

## 💾 Model Download

After committing and saving your Kaggle Notebook:

1. Go to **Versions** > Select your latest run.
2. Navigate to the **Output** tab.
3. Download model files:

```
model_checkpoints/
├── yolov8s_best.pt
└── faster_rcnn_best.pth
```

Also check for: `working_dir.zip` containing plots, checkpoints, and logs.

---

## 🔭 Future Work

- 📌 Hyperparameter tuning (YOLOv8 + Faster R-CNN)
- 🔄 Fine-tune EfficientDet on PCB data
- 🔧 Add SSD, RetinaNet, DETR for comparison
- 🧪 Detailed error analysis and heatmaps
- 🧠 Ensemble multiple models for performance boost
- 📦 Deploy best model in a real-time inspection pipeline

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

- 🧠 Ultralytics for YOLOv8
- 🔥 PyTorch & TorchVision
- 🧪 TensorFlow Hub
- 📊 Kaggle for GPU compute and hosting

---

## 🔗 Repository

**GitHub Repo:** [github.com/Nvm-seff/PCB-Defect-Detection-Using-YOLO-RCNN-EffiecientDet](https://github.com/Nvm-seff/PCB-Defect-Detection-Using-YOLO-RCNN-EffiecientDet)

⭐ Star this repository if you find it useful, and feel free to fork and contribute!