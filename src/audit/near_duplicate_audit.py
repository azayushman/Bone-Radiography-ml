from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image
import numpy as np

CLEAN_DATASET = Path("data/processed/clean_dataset.csv")
OUTPUT = Path("data/processed/near_duplicate_audit.csv")

HASH_SIZE = 32


def dhash(image_path):
    """
    Perceptual difference hash.
    Similar images should have similar hashes even if
    resized/recompressed.
    """
    image = Image.open(image_path).convert("L")
    image = image.resize((HASH_SIZE + 1, HASH_SIZE))

    pixels = np.asarray(image)

    diff = pixels[:, 1:] > pixels[:, :-1]

    bits = "".join("1" if x else "0" for x in diff.flatten())

    return int(bits, 2)


def hamming_distance(a, b):
    return (a ^ b).bit_count()


def main():

    print("=" * 60)
    print("NEAR-DUPLICATE AUDIT")
    print("=" * 60)

    df = pd.read_csv(CLEAN_DATASET)

    print(f"\nClean images: {len(df)}")

    hashes = []

    for i, row in df.iterrows():

        path = Path(row["image_path"])

        try:
            h = dhash(path)
            hashes.append(h)

        except Exception as e:
            print(f"ERROR: {path}")
            print(e)
            hashes.append(None)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    df["perceptual_hash"] = hashes

    valid = df[df["perceptual_hash"].notna()].copy()

    print("\nValid images:", len(valid))

    # Compare every image against every other image.
    # 865 images = ~374k comparisons, which is fine.
    results = []

    values = valid["perceptual_hash"].tolist()
    indices = valid.index.tolist()

    print("\nComparing images...")

    for i in range(len(values)):

        if i % 100 == 0:
            print(f"Compared {i}/{len(values)}")

        for j in range(i + 1, len(values)):

            distance = hamming_distance(
                values[i],
                values[j]
            )

            # Very small distance = potentially same image
            if distance <= 5:

                idx1 = indices[i]
                idx2 = indices[j]

                results.append({
                    "image1": df.loc[idx1, "image_path"],
                    "image2": df.loc[idx2, "image_path"],
                    "label1": df.loc[idx1, "label"],
                    "label2": df.loc[idx2, "label"],
                    "hash_distance": distance
                })

    result_df = pd.DataFrame(results)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(OUTPUT, index=False)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print("Potential near-duplicate pairs:", len(result_df))

    if len(result_df) > 0:

        print("\nPotential near-duplicates:")

        print(
            result_df
            .sort_values("hash_distance")
            .head(30)
            .to_string(index=False)
        )

        print("\nLabel relationships:")

        print(
            result_df
            .apply(
                lambda x: (
                    f"{x['label1']} -> {x['label2']}"
                ),
                axis=1
            )
            .value_counts()
            .to_string()
        )

    else:

        print("No near-duplicate pairs detected.")

    print("\nSaved:")
    print(OUTPUT)

    print("\nAudit complete.")


if __name__ == "__main__":
    main()