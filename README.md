# End-to-End Lane Detection using YOLOv8

This repository provides a complete workflow for training, evaluating, and comparing **YOLOv8n-seg** and **YOLOv8s-seg** models for the task of real-time, end-to-end lane detection. The implementation uses the Ultralytics framework to fine-tune models on a custom dataset, creating a robust solution for identifying lane boundaries in complex driving scenarios.



---

## 📋 Table of Contents

- [Main Features](#main-features)
- [Getting Started](#getting-started)
- [Project Workflow](#project-workflow)
- [Performance and Results](#performance-and-results)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

---

## 🌟 Main Features

-   **High Performance**: Built on the state-of-the-art YOLOv8 architecture.
-   **Multi-Task**: Utilizes instance segmentation to produce both bounding boxes and pixel-level masks for lanes.
-   **Easy to Use**: Includes simple Python scripts for data preparation, training, and inference.
-   **Comparative Analysis**: Provides a direct performance comparison between the `YOLOv8n-seg` (nano) and `YOLOv8s-seg` (small) models.

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* Python 3.10+
* PyTorch 2.0+
* An **NVIDIA GPU** with **CUDA** and **cuDNN** installed is required for efficient training.
* Git for cloning the repository.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```

2.  **Navigate into the project directory:**
    ```bash
    cd your-repo-name
    ```

3.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

4.  **Activate the virtual environment:**
    * On Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

5.  **Install dependencies:**
    ```bash
    pip install ultralytics scikit-learn pandas
    ```

---

## 🛠️ Project Workflow

The project is designed to be run in a sequence of steps using the provided Python scripts.

### Step 1: Prepare the Dataset

Your dataset should be in the YOLO format. If you only have a single `train` set, use the `split_data.py` script to automatically create `train` and `val` subsets.

1.  Place your raw dataset in a folder named `dataset/` (e.g., `dataset/images/train`, `dataset/labels/train`).
2.  Run the splitting script:
    ```bash
    python split_data.py
    ```
    This will create a new `dataset_split/` directory with the correctly organized data.

### Step 2: Train the Model

The `train.py` script handles the training process. You can edit the script to choose the model size you want to train (`yolov8n-seg.pt` or `yolov8s-seg.pt`).

```bash
python train.py
```
* The script automatically creates the necessary `data.yaml` file.
* Trained models and results will be saved in the `training_results/` directory.

### Step 3: Evaluate and Test

After training, use the `predict.py` script to run your trained model on new images or videos.

1.  Edit `predict.py` to point to your `best.pt` model file and your source video/image.
2.  Run the script:
    ```bash
    python predict.py
    ```
    The output with the detected lanes will be saved in the `runs/segment/predict/` directory.

---

## 📈 Performance and Results

This table summarizes the final performance of the **YOLOv8n-seg** and **YOLOv8s-seg** models on the validation set after 100 epochs.

| Metric            | YOLOv8n-seg | YOLOv8s-seg |
| :---------------- | :---------: | :---------: |
| **mAP50-95 (Box)** |    0.840    |    0.840    |
| **mAP50-95 (Mask)** |    0.202    |    0.202    |
| **Precision (Box)** |    0.969    |    0.969    |
| **Recall (Box)** |    0.775    |    0.775    |

### Training Progress

The following graph shows the learning progress (mAP for boxes and masks) over the training epochs.



**Conclusion**: Both models achieve identical performance on this validation set. This indicates that for this specific dataset and training configuration, the additional capacity of the YOLOv8s model did not yield a significant accuracy improvement over the more lightweight YOLOv8n model, making the **YOLOv8n-seg version more efficient for this task**.

---

## 💡 Troubleshooting

-   **CUDA Errors (`Invalid CUDA device` or `unknown error`)**: Ensure you have a compatible NVIDIA GPU, the latest drivers, the CUDA Toolkit, and the CUDA-enabled version of PyTorch installed.
-   **File Not Found Errors**: Double-check that your folder structure exactly matches the paths specified in your `.yaml` file and Python scripts.
-   **Windows Memory Error (`shared file mapping`)**: This is a common issue with multiprocessing on Windows. The simplest fix is to set `workers=0` in your `train.py` script.

---

## Acknowledgments

This project is built upon the excellent [Ultraclytics YOLOv8](https://github.com/ultralytics/ultralytics) repository. We thank the authors for their significant contributions to the open-source community.

---
