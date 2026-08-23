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

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_final.csv"
)

# IMPORTANT:
# Analyze Experiment 1, NOT the old 71.32% baseline.
MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_resnet50_exp1.pth"
)

# Keep baseline error analysis untouched.
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "error_analysis_exp1.csv"
)

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

CLASS_NAMES = [
    "normal",
    "osteopenia",
    "osteoporosis",
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("BONE RADIOGRAPHY - EXPERIMENT 1 ERROR ANALYSIS")
print("=" * 60)

print(f"Device: {device}")

if device.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


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
        std=weights.transforms().std,
    ),
])


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_CSV)

print(
    f"\nTest images: {len(df)}"
)

print("\nTest distribution:")

print(
    df["label"]
    .value_counts()
)


# ============================================================
# LOAD EXPERIMENT 1 MODEL
# ============================================================

print("\nLoading ResNet-50 Experiment 1...")

print(
    f"Checkpoint: {MODEL_FILE}"
)

model = models.resnet50(
    weights=None
)


# ------------------------------------------------------------
# Freeze everything first
# ------------------------------------------------------------

for parameter in model.parameters():
    parameter.requires_grad = False


# ------------------------------------------------------------
# Recreate Experiment 1 classifier
# ------------------------------------------------------------

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(
        p=0.35
    ),
    nn.Linear(
        num_features,
        len(CLASS_NAMES),
    ),
)


# ------------------------------------------------------------
# Recreate Experiment 1 fine-tuning
# ------------------------------------------------------------

for parameter in model.layer3.parameters():
    parameter.requires_grad = True

for parameter in model.layer4.parameters():
    parameter.requires_grad = True


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(device)

model.eval()


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\nModel configuration:")

print("Architecture: ResNet-50")
print("Layer 1:      FROZEN")
print("Layer 2:      FROZEN")
print("Layer 3:      TRAINABLE")
print("Layer 4:      TRAINABLE")
print("FC:           TRAINABLE")
print("Dropout:      0.35")


# ============================================================
# PREDICTIONS
# ============================================================

results = []

print("\nRunning predictions...")


with torch.no_grad():

    for i, row in df.iterrows():

        image_path = (
            PROJECT_ROOT
            / row["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(
            image
        ).unsqueeze(0).to(device)


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        predicted_index = (
            prediction.item()
        )

        predicted_label = (
            INDEX_TO_CLASS[
                predicted_index
            ]
        )

        actual_label = row["label"]

        correct = (
            actual_label
            == predicted_label
        )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "image_hash":
                row["image_hash"],

            "image_path":
                row["image_path"],

            "actual_label":
                actual_label,

            "predicted_label":
                predicted_label,

            "confidence":
                float(
                    confidence.item()
                ),

            "correct":
                correct,

            "source_dataset":
                row["source_dataset"],

            "normal_probability":
                float(
                    probabilities[0][0]
                    .item()
                ),

            "osteopenia_probability":
                float(
                    probabilities[0][1]
                    .item()
                ),

            "osteoporosis_probability":
                float(
                    probabilities[0][2]
                    .item()
                ),
        })


        if (i + 1) % 25 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(df)}"
            )


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# BASIC SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 1 ERROR ANALYSIS")
print("=" * 60)


total = len(results_df)

correct_predictions = (
    results_df[
        results_df["correct"]
    ]
)

incorrect_predictions = (
    results_df[
        ~results_df["correct"]
    ]
)


print(
    f"\nTotal predictions:     {total}"
)

print(
    f"Correct predictions:   "
    f"{len(correct_predictions)}"
)

print(
    f"Incorrect predictions: "
    f"{len(incorrect_predictions)}"
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = (
    len(correct_predictions)
    / total
)


print(
    f"\nCalculated accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Calculated accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "-" * 60)
print("CONFIDENCE ANALYSIS")
print("-" * 60)


correct_confidence = (
    correct_predictions[
        "confidence"
    ].mean()
)

incorrect_confidence = (
    incorrect_predictions[
        "confidence"
    ].mean()
)


print(
    f"\nCorrect predictions average "
    f"confidence:   "
    f"{correct_confidence:.4f}"
)

print(
    f"Incorrect predictions average "
    f"confidence: "
    f"{incorrect_confidence:.4f}"
)


confidence_gap = (
    correct_confidence
    - incorrect_confidence
)


print(
    f"\nConfidence gap: "
    f"{confidence_gap:.4f}"
)


# ============================================================
# ERROR PAIRS
# ============================================================

print("\n" + "-" * 60)
print("MISCLASSIFICATION PAIRS")
print("-" * 60)


if len(incorrect_predictions) > 0:

    confusion_pairs = (
        incorrect_predictions
        .groupby(
            [
                "actual_label",
                "predicted_label"
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        confusion_pairs.to_string()
    )

else:

    print(
        "No incorrect predictions."
    )


# ============================================================
# ERRORS BY ACTUAL CLASS
# ============================================================

print("\n" + "-" * 60)
print("ERRORS BY ACTUAL CLASS")
print("-" * 60)


if len(incorrect_predictions) > 0:

    errors_by_class = (
        incorrect_predictions[
            "actual_label"
        ]
        .value_counts()
    )

    print(
        errors_by_class.to_string()
    )

else:

    print(
        "No errors."
    )


# ============================================================
# ERRORS BY PREDICTED CLASS
# ============================================================

print("\n" + "-" * 60)
print("ERRORS BY PREDICTED CLASS")
print("-" * 60)


if len(incorrect_predictions) > 0:

    errors_by_prediction = (
        incorrect_predictions[
            "predicted_label"
        ]
        .value_counts()
    )

    print(
        errors_by_prediction.to_string()
    )

else:

    print(
        "No errors."
    )


# ============================================================
# LOWEST-CONFIDENCE PREDICTIONS
# ============================================================

print("\n" + "-" * 60)
print("20 LOWEST-CONFIDENCE PREDICTIONS")
print("-" * 60)


lowest_confidence = (
    results_df
    .sort_values(
        "confidence"
    )
    [
        [
            "image_path",
            "actual_label",
            "predicted_label",
            "confidence",
            "normal_probability",
            "osteopenia_probability",
            "osteoporosis_probability",
        ]
    ]
    .head(20)
)


print(
    lowest_confidence
    .to_string(
        index=False
    )
)


# ============================================================
# HIGHEST-CONFIDENCE INCORRECT
# ============================================================

print("\n" + "-" * 60)
print(
    "HIGHEST-CONFIDENCE INCORRECT "
    "PREDICTIONS"
)
print("-" * 60)


if len(incorrect_predictions) > 0:

    highest_confidence_errors = (
        incorrect_predictions
        .sort_values(
            "confidence",
            ascending=False
        )
        [
            [
                "image_path",
                "actual_label",
                "predicted_label",
                "confidence",
                "normal_probability",
                "osteopenia_probability",
                "osteoporosis_probability",
            ]
        ]
        .head(20)
    )

    print(
        highest_confidence_errors
        .to_string(
            index=False
        )
    )

else:

    print(
        "No incorrect predictions."
    )


# ============================================================
# NORMAL → OSTEOPOROSIS
# ============================================================

print("\n" + "-" * 60)
print("NORMAL → OSTEOPOROSIS ERRORS")
print("-" * 60)


normal_to_osteoporosis = (
    results_df[
        (
            results_df["actual_label"]
            == "normal"
        )
        &
        (
            results_df["predicted_label"]
            == "osteoporosis"
        )
    ]
)


print(
    f"Count: "
    f"{len(normal_to_osteoporosis)}"
)


if len(normal_to_osteoporosis) > 0:

    print(
        normal_to_osteoporosis[
            [
                "image_path",
                "confidence",
                "normal_probability",
                "osteopenia_probability",
                "osteoporosis_probability",
            ]
        ]
        .sort_values(
            "confidence",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# OSTEOPOROSIS → NORMAL
# ============================================================

print("\n" + "-" * 60)
print("OSTEOPOROSIS → NORMAL ERRORS")
print("-" * 60)


osteoporosis_to_normal = (
    results_df[
        (
            results_df["actual_label"]
            == "osteoporosis"
        )
        &
        (
            results_df["predicted_label"]
            == "normal"
        )
    ]
)


print(
    f"Count: "
    f"{len(osteoporosis_to_normal)}"
)


if len(osteoporosis_to_normal) > 0:

    print(
        osteoporosis_to_normal[
            [
                "image_path",
                "confidence",
                "normal_probability",
                "osteopenia_probability",
                "osteoporosis_probability",
            ]
        ]
        .sort_values(
            "confidence",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# OSTEOPENIA ERRORS
# ============================================================

print("\n" + "-" * 60)
print("OSTEOPENIA ERRORS")
print("-" * 60)


osteopenia_errors = (
    results_df[
        (
            results_df["actual_label"]
            == "osteopenia"
        )
        &
        (
            results_df["predicted_label"]
            != "osteopenia"
        )
    ]
)


print(
    f"Count: "
    f"{len(osteopenia_errors)}"
)


if len(osteopenia_errors) > 0:

    print(
        osteopenia_errors[
            [
                "image_path",
                "predicted_label",
                "confidence",
                "normal_probability",
                "osteopenia_probability",
                "osteoporosis_probability",
            ]
        ]
        .sort_values(
            "confidence",
            ascending=False
        )
        .to_string(
            index=False
        )
    )
else:

    print(
        "No osteopenia errors."
    )


# ============================================================
# PER-CLASS CONFIDENCE
# ============================================================

print("\n" + "-" * 60)
print("AVERAGE CONFIDENCE BY ACTUAL CLASS")
print("-" * 60)


class_confidence = (
    results_df
    .groupby("actual_label")[
        "confidence"
    ]
    .mean()
)


print(
    class_confidence.to_string()
)


# ============================================================
# PER-CLASS CORRECT CONFIDENCE
# ============================================================

print("\n" + "-" * 60)
print("AVERAGE CONFIDENCE OF CORRECT PREDICTIONS")
print("-" * 60)


correct_class_confidence = (
    correct_predictions
    .groupby("actual_label")[
        "confidence"
    ]
    .mean()
)


print(
    correct_class_confidence
    .to_string()
)


# ============================================================
# PER-CLASS INCORRECT CONFIDENCE
# ============================================================

print("\n" + "-" * 60)
print("AVERAGE CONFIDENCE OF INCORRECT PREDICTIONS")
print("-" * 60)


if len(incorrect_predictions) > 0:

    incorrect_class_confidence = (
        incorrect_predictions
        .groupby("actual_label")[
            "confidence"
        ]
        .mean()
    )

    print(
        incorrect_class_confidence
        .to_string()
    )

else:

    print(
        "No incorrect predictions."
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)

print(
    f"\nResults saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nExperiment 1 checkpoint analyzed:"
)

print(
    MODEL_FILE
)

print("\nTest set was not modified.")