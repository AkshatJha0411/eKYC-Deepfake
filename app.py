import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MTCNN face detector
mtcnn = MTCNN(keep_all=True, device=device)
# Load FaceNet embeddings
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load your fine-tuned model checkpoint
deepfake_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = deepfake_model.classifier[1].in_features
deepfake_model.classifier[1] = torch.nn.Linear(in_features, 2)
deepfake_model.load_state_dict(torch.load('deepfake_model.pth', map_location=device))
deepfake_model.eval().to(device)

st.title("Deepfake Detection & Face Verification Demo")

uploaded_file1 = st.file_uploader("Upload First Image (ID Photo):", type=["jpg", "png"])
uploaded_file2 = st.file_uploader("Upload Second Image (Selfie) [Optional]:", type=["jpg", "png"])
uploaded_video = st.file_uploader("Upload Selfie Video [Optional]:", type=["mp4"])

def analyze(img1,img2=None,video=None):
    if video:
        cap = cv2.VideoCapture(video)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            face = mtcnn(frame)
            if face is not None:
                frames.append(face)
        cap.release()
        if not frames:
            return None,None,None
        face1 = frames[0]
        liveness = np.var([f.mean().item() for f in frames]) > 0.1
        match = None
    else:
        face1 = mtcnn(np.array(img1))
        if img2:
            face2 = mtcnn(np.array(img2))
            if face1 is None or face2 is None:
                return None,None,None
            emb1 = resnet(face1.unsqueeze(0).to(device)).detach().cpu().numpy()
            emb2 = resnet(face2.unsqueeze(0).to(device)).detach().cpu().numpy()
            match = cosine_similarity(emb1, emb2)[0][0]
        else:
            match = None
        liveness = 1.0

    if face1 is not None:
        face1 = face1.to(device)
        dct1 = torch.fft.fft2(face1.unsqueeze(0)).real
        pred = deepfake_model(dct1)
        authenticity_prob = F.softmax(pred, dim=1)[0][0].item()
        authenticity = "Authentic" if authenticity_prob > 0.5 else "Forged"
    else:
        authenticity = None

    return match, liveness, authenticity

if uploaded_file1 and (uploaded_file2 or uploaded_video):
    img1 = Image.open(uploaded_file1)
    img2 = Image.open(uploaded_file2) if uploaded_file2 else None

    # If video uploaded, save temp file
    if uploaded_video:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        match, liveness, auth = analyze(img1, None, video_path)
    else:
        match, liveness, auth = analyze(img1, img2, None)

    st.write(f"Match Score: {match if match is not None else 'N/A'}")
    st.write(f"Liveness Score: {liveness}")
    st.write(f"Authenticity: {auth}")
else:
    st.info("Upload first image plus a second image or selfie video for analysis.")
