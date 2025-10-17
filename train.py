import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, RandomBrightnessContrast, HueSaturationValue
import pandas as pd
from torch import nn
from torch.optim import Adam
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Dataset class for loading images and labels
class ForgeryDataset(Dataset):
    def __init__(self, df, base_path, transform=None):
        self.df = df
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_path, 'Forgery_Dataset', row['image_path'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 0 if row['label'] == "real" else 1

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = torch.from_numpy(image).float() / 255.0
        return image, label

# Training function
def train_model(base_path, csv_path, num_epochs=2, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    labels_df = pd.read_csv(csv_path)
    valid_df = labels_df[labels_df.apply(lambda row: os.path.exists(os.path.join(base_path, 'Forgery_Dataset', row['image_path'])), axis=1)]

    from albumentations import Compose, RandomBrightnessContrast, HueSaturationValue
    aug_transform = Compose([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    ])

    def collate_fn(batch):
        batch = [item for item in batch if item[0] is not None]
        if len(batch) == 0:
            return None, None
        return torch.utils.data.dataloader.default_collate(batch)

    train_dataset = ForgeryDataset(valid_df, base_path, transform=aug_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    deepfake_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).to(device)
    in_features = deepfake_model.classifier[1].in_features
    deepfake_model.classifier[1] = nn.Linear(in_features, 2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(deepfake_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        deepfake_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            if images is None:  # skipping bad batch
                continue
            images = images.to(device)
            labels = labels.to(device)
            dct_imgs = torch.fft.fft2(images).real
            outputs = deepfake_model(dct_imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Save final model
    save_path = os.path.join(base_path, 'deepfake_model.pth')
    torch.save(deepfake_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    BASE_PATH = '/path/to/Sentinel_FaceV1'  # update as needed
    CSV_PATH = os.path.join(BASE_PATH, 'Forgery_Dataset', 'train_labels.csv')
    train_model(BASE_PATH, CSV_PATH)
