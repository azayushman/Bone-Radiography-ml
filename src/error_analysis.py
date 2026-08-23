from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test_final.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "best_resnet50_classifier.pth"

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "error_analysis.csv"

IMAGE_SIZE = 224

CLASS_TO_INDEX = {
    "normal": 0,
    "osteopenia": 1,
    "osteoporosis": 2,
}

INDEX_TO_CLASS = {
    0: "normal",
    1: "osteopenia",
    2: "osteoporosis",
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("BONE RADIOGRAPHY - ERROR ANALYSIS")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TEST_CSV)

print("\nTest images:", len(df))


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading ResNet-50...")

weights = ResNet50_Weights.DEFAULT

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(
        model.fc.in_features,
        len(CLASS_TO_INDEX),
    ),
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

results = []

with torch.no_grad():

    for i, row in df.iterrows():

        image_path = PROJECT_ROOT / row["image_path"]

        image = Image.open(image_path).convert("RGB")

        image = transform(image).unsqueeze(0).to(device)

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

        predicted_index = prediction.item()
        predicted_label = INDEX_TO_CLASS[predicted_index]

        actual_label = row["label"]

        results.append({
            "image_hash": row["image_hash"],
            "image_path": row["image_path"],
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "confidence": float(confidence.item()),
            "correct": actual_label == predicted_label,
            "source_dataset": row["source_dataset"],
            "normal_probability": float(probabilities[0][0]),
            "osteopenia_probability": float(probabilities[0][1]),
            "osteoporosis_probability": float(probabilities[0][2]),
        })

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(df)}")


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS SUMMARY")
print("=" * 60)

print("\nTotal:", len(results_df))

print(
    "Correct:",
    results_df["correct"].sum()
)

print(
    "Incorrect:",
    (~results_df["correct"]).sum()
)


print("\nConfusion pairs:")

errors = results_df[
    ~results_df["correct"]
].copy()

if len(errors) > 0:

    confusion = (
        errors
        .groupby(
            ["actual_label", "predicted_label"]
        )
        .size()
        .sort_values(ascending=False)
    )

    print(confusion.to_string())


print("\nAverage confidence:")

print(
    results_df.groupby("correct")["confidence"]
    .mean()
    .to_string()
)


print("\nLowest-confidence predictions:")

print(
    results_df
    .sort_values("confidence")
    [
        [
            "image_path",
            "actual_label",
            "predicted_label",
            "confidence",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print("\nSaved:")
print(OUTPUT_FILE)

print("\nError analysis complete.")