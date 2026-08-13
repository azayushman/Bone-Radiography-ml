from pathlib import Path
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "processed" / "primary_dataset.csv"

df = pd.read_csv(CSV_FILE)

print("=" * 50)
print("DATASET QUALITY AUDIT")
print("=" * 50)

print(f"\nTotal images: {len(df)}")

print("\nClass distribution:")
print(df["diagnosis"].value_counts())

print("\nClass percentages:")
print((df["diagnosis"].value_counts(normalize=True) * 100).round(2))

# --------------------------------------------------
# Check images
# --------------------------------------------------

valid_images = 0
corrupted_images = []
image_sizes = []

print("\nChecking images...")

for _, row in df.iterrows():

    image_path = PROJECT_ROOT / row["image_path"]

    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            image_sizes.append(img.size)

        valid_images += 1

    except Exception as e:
        corrupted_images.append({
            "patient_id": row["patient_id"],
            "path": str(image_path),
            "error": str(e)
        })

print(f"\nValid images: {valid_images}")
print(f"Corrupted images: {len(corrupted_images)}")

if corrupted_images:
    print("\nCorrupted files:")
    for item in corrupted_images:
        print(item)

# --------------------------------------------------
# Image dimensions
# --------------------------------------------------

if image_sizes:

    sizes = pd.Series(image_sizes)

    print("\nImage dimensions:")
    print(sizes.value_counts().head(20))

# --------------------------------------------------
# T-score
# --------------------------------------------------

print("\nT-score statistics:")
print(df["t_score"].describe())

print("\nT-score missing values:")
print(df["t_score"].isna().sum())

print("\n" + "=" * 50)
print("AUDIT COMPLETE")
print("=" * 50)