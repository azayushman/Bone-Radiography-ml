from pathlib import Path
import hashlib
from collections import defaultdict

SECONDARY_DIR = Path("data/raw/secondary")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def file_hash(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)

    return h.hexdigest()


def main():

    images = [
        p for p in SECONDARY_DIR.rglob("*")
        if p.is_file() and p.suffix in EXTENSIONS
    ]

    print("=" * 50)
    print("SECONDARY DATASET INTERNAL DUPLICATE AUDIT")
    print("=" * 50)

    print(f"Total secondary images: {len(images)}")

    hashes = defaultdict(list)

    print("\nHashing images...")

    for i, image in enumerate(images, 1):

        if i % 250 == 0:
            print(f"Processed: {i}/{len(images)}")

        try:
            h = file_hash(image)
            hashes[h].append(image)

        except Exception as e:
            print(f"ERROR: {image}")
            print(e)

    duplicate_groups = {
        h: paths
        for h, paths in hashes.items()
        if len(paths) > 1
    }

    duplicate_files = sum(
        len(paths) for paths in duplicate_groups.values()
    )

    extra_copies = sum(
        len(paths) - 1
        for paths in duplicate_groups.values()
    )

    unique_images = len(hashes)

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)

    print(f"Total files:             {len(images)}")
    print(f"Unique images:           {unique_images}")
    print(f"Duplicate groups:        {len(duplicate_groups)}")
    print(f"Files in duplicate groups:{duplicate_files}")
    print(f"Extra duplicate copies:  {extra_copies}")

    if duplicate_groups:

        print("\nExamples:")

        shown = 0

        for paths in duplicate_groups.values():

            print("\nDuplicate group:")

            for path in paths:
                print(f"  {path}")

            shown += 1

            if shown >= 20:
                break

    print("\nAudit complete.")


if __name__ == "__main__":
    main()