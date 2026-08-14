from pathlib import Path
from PIL import Image
import hashlib
import csv

PRIMARY = Path("data/raw/primary")
SECONDARY = Path("data/raw/secondary")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def image_files(root):
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in EXTENSIONS
    ]


def md5(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def get_class(path):
    parts = [p.lower() for p in path.parts]

    for cls in ["normal", "osteopenia", "osteoporosis"]:
        if cls in parts:
            return cls

    return "unknown"


def main():

    primary = image_files(PRIMARY)
    secondary = image_files(SECONDARY)

    print("=" * 60)
    print("DUPLICATE + LABEL CONFLICT AUDIT")
    print("=" * 60)

    print(f"Primary images:   {len(primary)}")
    print(f"Secondary images: {len(secondary)}")

    print("\nHashing primary images...")

    primary_hashes = {}

    for p in primary:
        try:
            h = md5(p)

            if h not in primary_hashes:
                primary_hashes[h] = []

            primary_hashes[h].append(p)

        except Exception as e:
            print(f"ERROR: {p} -> {e}")

    print("Hashing secondary images...")

    rows = []

    for p in secondary:

        try:
            h = md5(p)

            if h not in primary_hashes:
                continue

            secondary_class = get_class(p)

            for primary_path in primary_hashes[h]:

                primary_class = get_class(primary_path)

                rows.append({
                    "secondary_path": str(p),
                    "secondary_class": secondary_class,
                    "primary_path": str(primary_path),
                    "primary_class": primary_class,
                    "label_conflict": secondary_class != primary_class
                })

        except Exception as e:
            print(f"ERROR: {p} -> {e}")

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Exact duplicate pairs: {len(rows)}")

    conflicts = [
        r for r in rows
        if r["label_conflict"]
    ]

    agreements = [
        r for r in rows
        if not r["label_conflict"]
    ]

    print(f"Label agreements:      {len(agreements)}")
    print(f"Label conflicts:       {len(conflicts)}")

    print("\n" + "=" * 60)
    print("CONFLICT MATRIX")
    print("=" * 60)

    matrix = {}

    for r in conflicts:

        key = (
            r["secondary_class"],
            r["primary_class"]
        )

        matrix[key] = matrix.get(key, 0) + 1

    for (secondary_class, primary_class), count in sorted(matrix.items()):

        print(
            f"Secondary {secondary_class:12s}"
            f" -> Primary {primary_class:12s}"
            f": {count}"
        )

    output = Path("data/processed/duplicate_audit.csv")

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "secondary_path",
                "secondary_class",
                "primary_path",
                "primary_class",
                "label_conflict"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nDetailed report saved to:")
    print(output)

    print("\nAudit complete.")


if __name__ == "__main__":
    main()