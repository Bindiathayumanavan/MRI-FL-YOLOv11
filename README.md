# 🧠 MRI-FL-YOLOv11

A hybrid system for brain tumor detection using a combination of **image classification**, **object detection (YOLOv11)**, and **federated learning**. This project simulates a real-world privacy-preserving medical imaging pipeline with an end-to-end solution — from data preprocessing to model training and evaluation.

---

## 📁 Project Structure

```
MRI-FL-YOLOv11/
│
├── BrainMRI/                 # Preprocessed dataset with train/val split
│   ├── train/
│   │   ├── yes/
│   │   └── no/
│   └── val/
│       ├── yes/
│       └── no/
│
├── yolov11_dataset/          # YOLOv11-compatible dataset (images + labels)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── models/                   # Custom model definitions
│
├── utils/                    # Utility scripts (splitting, loading, etc.)
│
├── clustering/               # Federated learning node simulation
│
├── federated/                # Core federated averaging logic
│
├── train.py                  # Main federated training script
├── evaluate.py               # Evaluation pipeline
├── train_classifier.py       # Tumor classification using ResNet18
├── split_dataset.py          # For splitting raw dataset into train/val
├── requirements.txt          # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧪 Components

- **Tumor Classification** – Binary classifier using ResNet18 on MRI scans
- **Object Detection with YOLOv11** – (Planned) Detect tumor location in MRI
- **Federated Learning Simulation** – Emulates client-server architecture with privacy-preserving local training

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

Make sure the dataset is structured as follows under `BrainMRI`:

```
BrainMRI/
├── train/
│   ├── yes/
│   └── no/
└── val/
    ├── yes/
    └── no/
```

You can run the dataset splitter:

```bash
python split_dataset.py
```

---

### 3. Train Tumor Classifier

```bash
python train_classifier.py
```

This uses `ResNet18` and saves the model as:

```
tumor_classifier.pth
```

---

### 4. (Planned) YOLOv11 Integration

Prepare YOLOv11-style dataset inside `yolov11_dataset/` folder:

```
yolov11_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Stay tuned — YOLOv11 training and detection code will be added soon!

---

### 5. Federated Learning Simulation

You can run the main federated pipeline (once ready):

```bash
python train.py
```

---

## 📚 Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- opencv-python
- tqdm

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♀️ Author

**Bindia Thayumanavan**  
[GitHub](https://github.com/Bindiathayumanavan)
