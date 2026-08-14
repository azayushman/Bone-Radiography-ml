from pathlib import Path
import hashlib
import pandas as pd

PRIMARY_DIR = Path("data/raw/primary")
SECONDARY_DIR = Path("data/raw/secondary")

TRUTH_MAP = Path("data/processed/image_truth_map.csv")

OUTPUT = Path("data/processed/clean_dataset.csv")
EXCLUDED = Path("data/processed/excluded_conflicts.csv")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)

    return h.hexdigest()


def get_label(path):
    parts = [p.lower() for p in path.parts]

    if "normal" in parts:
        return "normal"

    if "osteopenia" in parts:
        return "osteopenia"

    if "osteoporosis" in parts:
        return "osteoporosis"

    return None


def scan_images(root, dataset):
    records = []

    images = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in EXTENSIONS
    ]

    print(f"{dataset}: {len(images)} images")

    for i, path in enumerate(images, 1):

        if i % 250 == 0:
            print(f"Processed {i}/{len(images)}")

        records.append({
            "hash": sha256(path),
            "path": str(path),
            "dataset": dataset,
            "label": get_label(path)
        })

    return pd.DataFrame(records)


def main():

    print("=" * 60)
    print("BUILD CLEAN MASTER DATASET")
    print("=" * 60)

    truth = pd.read_csv(TRUTH_MAP)

    primary = scan_images(PRIMARY_DIR, "primary")
    secondary = scan_images(SECONDARY_DIR, "secondary")

    images = pd.concat(
        [primary, secondary],
        ignore_index=True
    )

    # Only groups that passed the truth audit
    eligible = truth[truth["training_eligible"] == True].copy()

    print("\nEligible unique images:", len(eligible))

    clean_records = []

    for _, row in eligible.iterrows():

        image_hash = row["hash"]

        candidates = images[
            images["hash"] == image_hash
        ]

        # Prefer primary image when available.
        primary_candidates = candidates[
            candidates["dataset"] == "primary"
        ]

        if len(primary_candidates) > 0:
            selected = primary_candidates.iloc[0]
        else:
            selected = candidates.iloc[0]

        clean_records.append({
            "image_hash": image_hash,
            "image_path": selected["path"],
            "label": row["labels"],
            "source_dataset": selected["dataset"]
        })

    clean = pd.DataFrame(clean_records)

    # Safety checks
    assert clean["image_hash"].nunique() == len(clean)
    assert clean["label"].notna().all()

    allowed = {"normal", "osteopenia", "osteoporosis"}

    assert set(clean["label"]).issubset(allowed)

    print("\nFINAL CLEAN DATASET")
    print("-" * 40)

    print("Images:", len(clean))

    print("\nClass distribution:")
    print(clean["label"].value_counts().to_string())

    print("\nClass percentages:")
    print(
        clean["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )

    # Save clean dataset
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT, index=False)

    # Save excluded groups for audit trail
    excluded = truth[
        truth["training_eligible"] == False
    ].copy()

    excluded.to_csv(EXCLUDED, index=False)

    print("\nSaved:")
    print(OUTPUT)

    print(EXCLUDED)

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()