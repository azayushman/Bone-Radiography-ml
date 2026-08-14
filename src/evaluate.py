from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_resnet50_classifier.pth"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "normal",
    "osteopenia",
    "osteoporosis"
]

CLASS_TO_INDEX = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("BONE RADIOGRAPHY - MODEL EVALUATION")
print("=" * 60)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# DATASET
# ============================================================

class BoneXrayDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = PROJECT_ROOT / row["image_path"]

        image = Image.open(
            image_path
        ).convert("RGB")

        label = CLASS_TO_INDEX[
            row["diagnosis"]
        ]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORM
# ============================================================

weights = ResNet50_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_CSV)

print(
    f"\nTest images: {len(test_df)}"
)

print("\nTest distribution:")

print(
    test_df["diagnosis"]
    .value_counts()
)


test_dataset = BoneXrayDataset(
    test_df,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model = models.resnet50(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.30),
    nn.Linear(
        num_features,
        len(CLASS_NAMES)
    )
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"\nTest Accuracy: {accuracy:.4f}"
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


print("\nConfusion Matrix:\n")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)

print("\nClass order:")

for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")