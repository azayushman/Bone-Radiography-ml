from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "primary_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42

df = pd.read_csv(INPUT_FILE)

# --------------------------------------------------
# First split: 70% train, 30% temporary
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["diagnosis"],
    random_state=RANDOM_STATE
)

# --------------------------------------------------
# Second split: temporary -> 15% validation / 15% test
# --------------------------------------------------

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["diagnosis"],
    random_state=RANDOM_STATE
)

# --------------------------------------------------
# Save
# --------------------------------------------------

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# --------------------------------------------------
# Report
# --------------------------------------------------

print("=" * 50)
print("DATASET SPLIT")
print("=" * 50)

for name, split in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df),
]:

    print(f"\n{name}")
    print(f"Images: {len(split)}")

    print(
        split["diagnosis"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nPercent:"
    )

    print(
        (split["diagnosis"].value_counts(normalize=True) * 100)
        .round(2)
        .sort_index()
        .to_string()
    )

print("\nFiles created:")
print("data/processed/train.csv")
print("data/processed/val.csv")
print("data/processed/test.csv")
