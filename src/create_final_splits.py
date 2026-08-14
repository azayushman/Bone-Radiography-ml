from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT = Path("data/processed/clean_dataset.csv")
NEAR_DUPLICATES = Path("data/processed/near_duplicate_audit.csv")

TRAIN_OUTPUT = Path("data/processed/train_final.csv")
VAL_OUTPUT = Path("data/processed/val_final.csv")
TEST_OUTPUT = Path("data/processed/test_final.csv")

RANDOM_STATE = 42


def main():

    print("=" * 60)
    print("FINAL DATASET SPLIT")
    print("=" * 60)

    df = pd.read_csv(INPUT)

    print(f"\nClean images before near-duplicate removal: {len(df)}")

    # ---------------------------------------------------------
    # Remove one member of each near-duplicate pair
    # ---------------------------------------------------------

    near = pd.read_csv(NEAR_DUPLICATES)

    remove_paths = set()

    if len(near) > 0:

        for _, row in near.iterrows():

            # Keep image1, remove image2.
            # This makes the process deterministic.
            remove_paths.add(row["image2"])

    print("Near-duplicate pairs:", len(near))
    print("Images removed:", len(remove_paths))

    df = df[~df["image_path"].isin(remove_paths)].copy()

    print("Images after near-duplicate removal:", len(df))

    # ---------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------

    assert df["image_hash"].nunique() == len(df)
    assert df["image_path"].nunique() == len(df)
    assert df["label"].notna().all()

    # ---------------------------------------------------------
    # 70% TRAIN / 15% VAL / 15% TEST
    # ---------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE
    )

    # ---------------------------------------------------------
    # Verify no overlap
    # ---------------------------------------------------------

    train_hashes = set(train_df["image_hash"])
    val_hashes = set(val_df["image_hash"])
    test_hashes = set(test_df["image_hash"])

    assert len(train_hashes & val_hashes) == 0
    assert len(train_hashes & test_hashes) == 0
    assert len(val_hashes & test_hashes) == 0

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    def report(name, data):

        print(f"\n{name}")
        print("-" * 40)

        print("Images:", len(data))

        counts = data["label"].value_counts()

        print("\nClass distribution:")
        print(counts.to_string())

        print("\nPercent:")
        print(
            data["label"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .to_string()
        )

    report("TRAIN", train_df)
    report("VALIDATION", val_df)
    report("TEST", test_df)

    print("\nOVERLAP CHECK")
    print("-" * 40)

    print("TRAIN ∩ VAL :", len(train_hashes & val_hashes))
    print("TRAIN ∩ TEST:", len(train_hashes & test_hashes))
    print("VAL ∩ TEST  :", len(val_hashes & test_hashes))

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    val_df.to_csv(VAL_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    print("\nFiles created:")
    print(TRAIN_OUTPUT)
    print(VAL_OUTPUT)
    print(TEST_OUTPUT)

    print("\nFinal split complete.")


if __name__ == "__main__":
    main()