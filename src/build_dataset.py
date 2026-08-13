import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "primary"
    / "Osteoporosis Knee X-ray"
)

EXCEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "metadata"
    / "Osteoporosis Knee X-ray"
    / "patient details.xlsx"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "primary_dataset.csv"
)

# --------------------------------------------------
# Load patient metadata
# --------------------------------------------------

print("Loading Excel metadata...")

df = pd.read_excel(EXCEL_FILE)

print(f"Excel records: {len(df)}")

# Remove rows without diagnosis
df = df.dropna(subset=["Diagnosis"]).copy()

# Clean patient IDs
df["Patient Id"] = df["Patient Id"].astype(str).str.strip()

# Clean diagnosis
df["Diagnosis"] = df["Diagnosis"].astype(str).str.strip().str.lower()

# --------------------------------------------------
# Find all X-ray images
# --------------------------------------------------

print("Scanning X-ray images...")

image_files = []

for path in IMAGE_ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        image_files.append(path)

print(f"Images found: {len(image_files)}")

# --------------------------------------------------
# Create image lookup
# --------------------------------------------------

image_lookup = {}

for path in image_files:
    patient_id = path.stem.strip()

    if patient_id in image_lookup:
        print(f"WARNING: duplicate image ID: {patient_id}")

    image_lookup[patient_id] = path

# --------------------------------------------------
# Match metadata to images
# --------------------------------------------------

records = []
missing_images = []

for _, row in df.iterrows():

    patient_id = row["Patient Id"]

    if patient_id not in image_lookup:
        missing_images.append(patient_id)
        continue

    image_path = image_lookup[patient_id]

    records.append({
        "patient_id": patient_id,
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "diagnosis": row["Diagnosis"],
        "t_score": row.get("T-score Value"),
        "z_score": row.get("Z-score Value"),
    })

# --------------------------------------------------
# Create final dataset
# --------------------------------------------------

dataset = pd.DataFrame(records)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

dataset.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# Report
# --------------------------------------------------

print("\n========================================")
print("DATASET AUDIT")
print("========================================")

print(f"Excel records with diagnosis : {len(df)}")
print(f"Images found                 : {len(image_files)}")
print(f"Matched image/metadata pairs : {len(dataset)}")
print(f"Missing images               : {len(missing_images)}")

print("\nClass distribution:")
print(dataset["diagnosis"].value_counts())

print("\nMissing image IDs:")

if missing_images:
    for patient_id in missing_images:
        print("  ", patient_id)
else:
    print("   None")

print("\nT-score availability:")
print(dataset["t_score"].notna().value_counts())

print("\nSaved dataset:")
print(OUTPUT_FILE)