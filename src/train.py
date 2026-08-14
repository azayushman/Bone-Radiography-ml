from pathlib import Path
import copy
import json

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "best_resnet50_classifier.pth"
HISTORY_FILE = MODEL_DIR / "training_history.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5

NUM_WORKERS = 0

CLASS_NAMES = [
    "normal",
    "osteopenia",
    "osteoporosis",
]

CLASS_TO_INDEX = {
    name: i for i, name in enumerate(CLASS_NAMES)
}


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("BONE RADIOGRAPHY - RESNET50 TRAINING")
print("=" * 60)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
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

        image = Image.open(image_path).convert("RGB")

        label = CLASS_TO_INDEX[row["diagnosis"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

weights = ResNet50_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=7),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    ),
])


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("\nDataset:")
print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(val_df)}")

print("\nTraining distribution:")
print(train_df["diagnosis"].value_counts())


train_dataset = BoneXrayDataset(
    train_df,
    transform=train_transform
)

val_dataset = BoneXrayDataset(
    val_df,
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df["diagnosis"]
    .value_counts()
    .reindex(CLASS_NAMES)
)

class_weights = len(train_df) / (
    len(CLASS_NAMES) * class_counts
)

class_weights = torch.tensor(
    class_weights.values,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")

for name, weight in zip(CLASS_NAMES, class_weights):
    print(f"{name:15s}: {weight.item():.4f}")


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained ResNet-50...")

model = models.resnet50(
    weights=weights
)

# Freeze pretrained backbone initially
for parameter in model.parameters():
    parameter.requires_grad = False


# Replace classifier
num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.30),
    nn.Linear(num_features, len(CLASS_NAMES))
)

model = model.to(DEVICE)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
best_model_weights = copy.deepcopy(model.state_dict())

epochs_without_improvement = 0

history = []


for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_accuracy = val_correct / val_total


    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Acc:   {val_accuracy:.4f}"
    )


    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "model_name": "resnet50"
            },
            MODEL_FILE
        )

        print("✓ New best model saved.")

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

        if epochs_without_improvement >= PATIENCE:

            print("\nEarly stopping.")

            break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(best_model_weights)

with open(HISTORY_FILE, "w") as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Best validation loss: {best_val_loss:.4f}")

print(f"\nModel saved to:")
print(MODEL_FILE)

print("\nTraining history saved to:")
print(HISTORY_FILE)