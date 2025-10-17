import os
import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity

class DeepfakeVerifier:
    def __init__(self, model_path, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval().to(self.device)

    def process_images(self, img1, img2=None):
        face1 = self.mtcnn(img1)
        if img2 is not None:
            face2 = self.mtcnn(img2)
            if face1 is None or face2 is None:
                return None, None, None
            emb1 = self.resnet(face1.unsqueeze(0).to(self.device)).detach().cpu().numpy()
            emb2 = self.resnet(face2.unsqueeze(0).to(self.device)).detach().cpu().numpy()
            match_score = cosine_similarity(emb1, emb2)[0][0]
        else:
            match_score = None

        if face1 is not None:
            face1 = face1.to(self.device)
            dct1 = torch.fft.fft2(face1.unsqueeze(0)).real
            auth_pred = self.model(dct1)
            auth_prob = F.softmax(auth_pred, dim=1)[0][0].item()
            authenticity = 'Authentic' if auth_prob > 0.5 else 'Forged'
        else:
            authenticity = None

        return match_score, authenticity

    def grad_cam(self, input_tensor, target_class=0):
        self.model.eval()
        grads, activations = [], []

        def backward_hook(module, grad_in, grad_out):
            grads.append(grad_out[0])

        def forward_hook(module, input, output):
            activations.append(output)

        b_handle = self.model.features[-1].register_backward_hook(backward_hook)
        f_handle = self.model.features[-1].register_forward_hook(forward_hook)

        output = self.model(input_tensor)
        self.model.zero_grad()
        target_class = min(target_class, output.size(1) - 1)
        output[0][target_class].backward()

        pooled_grads = torch.mean(grads[0], dim=[0, 2, 3])
        heatmap = torch.sum(pooled_grads.unsqueeze(-1).unsqueeze(-1) * activations[0], dim=1)
        heatmap = F.relu(heatmap)

        b_handle.remove()
        f_handle.remove()

        return heatmap.detach().cpu().numpy()
