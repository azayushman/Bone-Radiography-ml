from pathlib import Path
import hashlib
import pandas as pd

PRIMARY_DIR = Path("data/raw/primary")
SECONDARY_DIR = Path("data/raw/secondary")

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


def scan_dataset(root, dataset_name):
    records = []

    images = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in EXTENSIONS
    ]

    print(f"{dataset_name}: {len(images)} files")

    for i, path in enumerate(images, 1):

        if i % 250 == 0:
            print(f"  Processed {i}/{len(images)}")

        records.append({
            "dataset": dataset_name,
            "path": str(path),
            "filename": path.name,
            "label": get_label(path),
            "hash": sha256(path)
        })

    return records


def classify_truth(group):
    labels = set(group["label"].dropna())

    primary_labels = set(
        group.loc[group.dataset == "primary", "label"].dropna()
    )

    secondary_labels = set(
        group.loc[group.dataset == "secondary", "label"].dropna()
    )

    has_primary = len(primary_labels) > 0
    has_secondary = len(secondary_labels) > 0

    # ---------------------------------------------------------
    # No usable label anywhere
    # ---------------------------------------------------------
    if len(labels) == 0:
        return "NO_LABEL"

    # ---------------------------------------------------------
    # PRIMARY DATASET EXISTS
    # ---------------------------------------------------------
    if has_primary:

        # Primary itself contains multiple labels.
        # This is ambiguous and must not automatically be resolved.
        if len(primary_labels) > 1:
            return "PRIMARY_INTERNAL_CONFLICT"

        primary_label = next(iter(primary_labels))

        # Secondary agrees exactly with primary.
        if has_secondary and secondary_labels == {primary_label}:
            return "AGREEMENT"

        # Secondary contains the primary label AND other labels.
        if has_secondary and primary_label in secondary_labels:
            return "SECONDARY_LABEL_CONFLICT"

        # Secondary has a completely different label.
        if has_secondary and primary_label not in secondary_labels:
            return "CROSS_DATASET_CONFLICT"

        # Primary only.
        return "PRIMARY_ONLY"

    # ---------------------------------------------------------
    # SECONDARY DATASET ONLY
    # ---------------------------------------------------------

    # Secondary has multiple labels for the same exact image.
    if len(secondary_labels) > 1:
        return "SECONDARY_INTERNAL_CONFLICT"

    # One clean secondary label.
    return "SECONDARY_ONLY"


def main():

    print("=" * 60)
    print("IMAGE TRUTH MAP - ENHANCED AUDIT")
    print("=" * 60)

    primary = scan_dataset(PRIMARY_DIR, "primary")
    secondary = scan_dataset(SECONDARY_DIR, "secondary")

    df = pd.DataFrame(primary + secondary)

    print("\nTotal files:", len(df))

    unique_hashes = df["hash"].nunique()

    print("Unique image contents:", unique_hashes)

    groups = []

    for image_hash, group in df.groupby("hash"):

        labels = set(group["label"].dropna())

        primary_labels = set(
            group.loc[group.dataset == "primary", "label"].dropna()
        )

        secondary_labels = set(
            group.loc[group.dataset == "secondary", "label"].dropna()
        )

        truth_status = classify_truth(group)

        groups.append({
            "hash": image_hash,
            "files": len(group),

            "datasets": ",".join(
                sorted(group.dataset.unique())
            ),

            "labels": ",".join(
                sorted(labels)
            ),

            "primary_labels": ",".join(
                sorted(primary_labels)
            ),

            "secondary_labels": ",".join(
                sorted(secondary_labels)
            ),

            "truth_status": truth_status,

            "primary_present": "primary" in group.dataset.values,

            "secondary_present": "secondary" in group.dataset.values
        })

    truth = pd.DataFrame(groups)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRUTH STATUS")
    print("=" * 60)

    print(
        truth["truth_status"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # CROSS-DATASET SUMMARY
    # ---------------------------------------------------------

    cross = truth[
        truth["primary_present"]
        & truth["secondary_present"]
    ]

    print("\n" + "=" * 60)
    print("CROSS-DATASET SUMMARY")
    print("=" * 60)

    print(
        "Cross-dataset unique images:",
        len(cross)
    )

    print("\nCross-dataset truth status:")

    print(
        cross["truth_status"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # TRAINING ELIGIBILITY
    # ---------------------------------------------------------

    truth["training_eligible"] = truth["truth_status"].isin([
        "AGREEMENT",
        "PRIMARY_ONLY",
        "SECONDARY_ONLY"
    ])

    print("\n" + "=" * 60)
    print("TRAINING ELIGIBILITY")
    print("=" * 60)

    print(
        truth["training_eligible"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output = Path(
        "data/processed/image_truth_map.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    truth.to_csv(
        output,
        index=False
    )

    print("\nSaved:")
    print(output)

    print("\nAudit complete.")


if __name__ == "__main__":
    main()