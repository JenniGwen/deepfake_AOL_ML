# isitfake 🕵️
isitfake is a web-based deepfake detection application where users can easily upload an image and instantly receive a probability score indicating whether the image is real or artificially generated (fake).

## Model Architecture 
The underlying deepfake detection model (v5) was rigorously trained and evaluated to ensure high reliability.
- Dataset: Trained on a balanced dataset of 120,000 samples (50% real, 50% fake).
- Data Split: The data was split 70/15/15 (Train: 84,000 | Val: 18,000 | Test: 18,000).
- Data Sources: 
    - Real images were sourced from celebdff, 140k, df_real, dfd, and dffd_ffhq (deduplicated). 
    - Fake images were sourced from celebdff, 140k, gan, stylegan, dfd, and 5 DFFD types.

**Feature Extraction:** The system uses a 1536-dimensional feature vector consisting of 1280 CNN features (from a 4-channel EfficientNet-B0), 128 FFT azimuthal features, and 128 noise FFT features.

**Classifier Results:**
RBF SVM: Accuracy: 0.9418 | AUC: 0.9780 | F1: 0.9417 | Precision: 0.9436 | Recall: 0.9399.

## Tech Stack & Project Structure
This project consists of two parts:
1. Python Backend (Flask + AI Model)
2. React Frontend (Vite + Tailwind CSS v4)

**Directory Structure**
- backend/: Contains the Python backend application (app.py) and the saved machine learning models including .pth and .onnx
- src/: Contains the frontend React application code

## 💻 Getting Started
(Note: Ensure you have Node.js (LTS Version) and Python (3.10 or above) installed before running)

1. Backend Setup
```bash
cd backend
pip install requirements.txt
python app.py
```
**Optional: Download the k-NN Feature Weights**
Due to GitHub's strict file size limits, the `features_v5.npz` file (466 MB) could not be included in this repository. This file contains the training features used by the k-NN algorithm to determine the specific **Manipulation Type** (e.g., Face Swap, StyleGAN, etc.).

**To enable the Manipulation Type breakdown:**
1. Download `features_v5.npz` from this https://drive.google.com/file/d/1f7CG9VzMqRb9-zaBI-jy9NqUAaXLMtFB/view?usp=sharing.
2. Place the downloaded file directly inside the `backend/` directory.

*Note: The application is designed to be fully modular. If you choose not to download this file, the deepfake detection model and backend API will still run perfectly, it will simply skip the specific "Detected Models" confidence breakdown.*

2. Frontend Setup
Open a SECOND terminal / command prompt then:
```bash
cd isitfake #root directory
npm install
npm run dev
```

Open http://localhost:5173


## HOW TO USE THE WEB APP
1. On the website, click the "Upload" icon or drag an image.
2. Once the preview appears, click "Run Analysis."
3. The "Aura" glow will appear while the AI is thinking.
4. View the result on the screen