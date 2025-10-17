# Deepfake-Proof eKYC System

## Overview
This project implements a deepfake-proof electronic Know Your Customer (eKYC) pipeline designed for secure identity verification. It combines multiple AI techniques to detect deepfake forgeries, verify face identity, and check liveness for images or videos. The system utilizes the Sentinel_FaceV1 dataset for training and evaluation.

## Features
- **Face Detection and Alignment:** Uses MTCNN from facenet-pytorch for robust face cropping.
- **Identity Verification:** Employs InceptionResnetV1 (pretrained on VGGFace2) with cosine similarity to verify identity between ID and selfie images.
- **Deepfake Detection:** Fine-tunes EfficientNet-B0 on frequency-domain (DCT) features for deepfake forgery classification.
- **Liveness Detection:** Uses frame variance for simple video-based liveness assessment.
- **Explainability:** Supports Grad-CAM visualization for interpretability of model decisions.
- **Demo UI:** Includes a Streamlit app for real-time web-based deployment and demonstration.

## Dataset
- Sentinel_FaceV1, augmented for robustness.
- Labels include `real` and `fake` (GAN_swap etc.) classes.
- Dataset structure assumed as per ZenTej AI hackathon.

## Usage

### Training
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Update dataset path (`BASE_PATH`) in the notebook and script.
3. Run training cells to fine-tune deepfake detection model.

### Inference
- Use the `process_input` function to predict match score, liveness score, and authenticity label on images or videos.

### Grad-CAM
- Use the `grad_cam` function for explanation heatmaps highlighting decision regions. (in inference.py)

### Deployment
- Run the included Streamlit app (`app.py`) for live demo at:
  ```
  streamlit run app.py
  ```

## Requirements
- Python 3.8+
- torch, torchvision
- facenet-pytorch
- albumentations
- opencv-python
- pandas, numpy, scikit-learn
- matplotlib
- streamlit

## Project Structure
- `train.py` — Training pipeline and dataset loader
- `inference.py` — Inference and Grad-CAM code
- `app.py` — Streamlit demo web app
- `deepfake_model.pth` — Saved model weights
- `requirements.txt` — Library dependencies
- `final_notebook.ipynb` — Notebook (used for development of model)
- `Sentinel_FaceV1.zip` — Dataset

## Results
- Achieves ~0.9 AUC F1 score on validation set (example metric).
- Real-time inference feasible with GPU support.
- Model robust to diverse lighting, poses, and forgery types.

## Limitations
- Liveness detection is basic, using only frame variance.
- Tested primarily on Sentinel_FaceV1 dataset; generalization to other datasets may require retraining.
- Grad-CAM visualization supports only EfficientNet module currently.

## License
This project uses open source libraries and datasets under their respective licenses.
