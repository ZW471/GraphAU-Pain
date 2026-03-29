#!/usr/bin/env python3
"""
Generate SynPAIN metadata.csv from image filenames.

Image filename format: {numeric_id}_{Pain|NoPain}_{gender}_{age}.jpg
Gender is normalized to M/F, age kept as-is (Old/Young).

Pairing strategy:
- Pain pairs (label=1): pair each Pain image (expr) with a random NoPain image
  from the same demographic group (neutral).
- NoPain pairs (label=0): pair each NoPain image (expr) with another random
  NoPain image from the same demographic group (neutral).

Split: 80/10/10 train/val/test assigned randomly per subject_id.
"""

import csv
import os
import random
import re
from collections import defaultdict

SEED = 42
random.seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "SyncPain")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "SynPAIN")  # will be symlink
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

IMAGE_DIRS = [
    ("SynPain_Part1/Images_Part1", os.path.join(DATA_DIR, "SynPain_Part1", "Images_Part1")),
    ("SynPain_Part2/Images_Part2", os.path.join(DATA_DIR, "SynPain_Part2", "Images_Part2")),
]

# Regex: numeric_id _ (Pain|NoPain) _ gender _ age .jpg
FNAME_RE = re.compile(
    r'^(\d+)_(Pain|NoPain)_([A-Za-z]+)_([A-Za-z]+)\.jpg$'
)


def normalize_gender(raw: str) -> str:
    """Normalize gender variants (man/Man/woman/Woman) to M/F."""
    low = raw.lower()
    if low == "man":
        return "M"
    elif low == "woman":
        return "F"
    else:
        raise ValueError(f"Unknown gender: {raw}")


def parse_images():
    """Parse all images, return list of dicts with metadata."""
    images = []
    for rel_dir, abs_dir in IMAGE_DIRS:
        if not os.path.isdir(abs_dir):
            print(f"WARNING: directory not found: {abs_dir}")
            continue
        for fname in sorted(os.listdir(abs_dir)):
            m = FNAME_RE.match(fname)
            if not m:
                print(f"WARNING: skipping unmatched filename: {fname}")
                continue
            numeric_id, pain_label, raw_gender, age = m.groups()
            images.append({
                "subject_id": numeric_id,
                "is_pain": pain_label == "Pain",
                "gender": normalize_gender(raw_gender),
                "age_group": age,  # Old or Young
                "rel_path": os.path.join(rel_dir, fname),
            })
    return images


def make_demo_key(img):
    """Demographic group key: (gender_normalized, age_group_lower)."""
    return (img["gender"], img["age_group"].lower())


def assign_splits(images, train_frac=0.8, val_frac=0.1):
    """Assign train/val/test splits by subject_id."""
    subject_ids = sorted(set(img["subject_id"] for img in images))
    random.shuffle(subject_ids)
    n = len(subject_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    split_map = {}
    for i, sid in enumerate(subject_ids):
        if i < n_train:
            split_map[sid] = "train"
        elif i < n_train + n_val:
            split_map[sid] = "val"
        else:
            split_map[sid] = "test"
    return split_map


def create_pairs(images, split_map):
    """Create pain and no-pain pairs."""
    # Group NoPain images by demographic key
    nopain_by_demo = defaultdict(list)
    pain_images = []
    nopain_images = []

    for img in images:
        key = make_demo_key(img)
        if img["is_pain"]:
            pain_images.append(img)
        else:
            nopain_by_demo[key].append(img)
            nopain_images.append(img)

    print(f"\nTotal images: {len(images)}")
    print(f"  Pain images: {len(pain_images)}")
    print(f"  NoPain images: {len(nopain_images)}")
    print(f"\nNoPain images by demographic group:")
    for key in sorted(nopain_by_demo.keys()):
        print(f"  {key}: {len(nopain_by_demo[key])}")

    pairs = []

    # Pain pairs: each Pain image paired with a random NoPain from same demo
    skipped_pain = 0
    for img in pain_images:
        key = make_demo_key(img)
        pool = nopain_by_demo.get(key, [])
        if not pool:
            skipped_pain += 1
            continue
        neutral = random.choice(pool)
        pairs.append({
            "neutral_path": neutral["rel_path"],
            "expr_path": img["rel_path"],
            "label": 1,
            "split": split_map[img["subject_id"]],
            "age_group": img["age_group"],
            "gender": img["gender"],
            "ethnicity": "unknown",
            "subject_id": img["subject_id"],
        })

    if skipped_pain:
        print(f"\nWARNING: Skipped {skipped_pain} Pain images with no matching NoPain demographic group")

    # NoPain pairs: each NoPain image paired with another random NoPain from same demo
    for img in nopain_images:
        key = make_demo_key(img)
        pool = [x for x in nopain_by_demo[key] if x["subject_id"] != img["subject_id"]]
        if not pool:
            # Fall back to same-subject different image if needed
            pool = [x for x in nopain_by_demo[key] if x["rel_path"] != img["rel_path"]]
        if not pool:
            # Only one NoPain image in this group -- pair with itself
            pool = [img]
        neutral = random.choice(pool)
        pairs.append({
            "neutral_path": neutral["rel_path"],
            "expr_path": img["rel_path"],
            "label": 0,
            "split": split_map[img["subject_id"]],
            "age_group": img["age_group"],
            "gender": img["gender"],
            "ethnicity": "unknown",
            "subject_id": img["subject_id"],
        })

    return pairs


def write_csv(pairs, output_path):
    """Write pairs to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "neutral_path", "expr_path", "label", "split",
        "age_group", "gender", "ethnicity", "subject_id"
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    print(f"\nWrote {len(pairs)} rows to {output_path}")


def print_summary(pairs):
    """Print summary statistics."""
    from collections import Counter

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # By label
    label_counts = Counter(p["label"] for p in pairs)
    print(f"\nBy label:")
    for label in sorted(label_counts):
        name = "Pain" if label == 1 else "NoPain"
        print(f"  {name} (label={label}): {label_counts[label]}")

    # By split
    split_counts = Counter(p["split"] for p in pairs)
    print(f"\nBy split:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {split_counts[split]}")

    # By split x label
    print(f"\nBy split x label:")
    for split in ["train", "val", "test"]:
        subset = [p for p in pairs if p["split"] == split]
        lc = Counter(p["label"] for p in subset)
        print(f"  {split}: NoPain={lc[0]}, Pain={lc[1]}")

    # By demographic group
    print(f"\nBy demographic group (gender, age):")
    demo_counts = Counter((p["gender"], p["age_group"]) for p in pairs)
    for key in sorted(demo_counts):
        print(f"  {key}: {demo_counts[key]}")

    # By demographic x label
    print(f"\nBy demographic group x label:")
    for key in sorted(demo_counts):
        subset = [p for p in pairs if (p["gender"], p["age_group"]) == key]
        lc = Counter(p["label"] for p in subset)
        print(f"  {key}: NoPain={lc[0]}, Pain={lc[1]}")

    # Unique subject_ids
    sids = set(p["subject_id"] for p in pairs)
    print(f"\nUnique subject_ids: {len(sids)}")


def create_symlink():
    """Create data/SynPAIN -> data/SyncPain symlink if needed."""
    synpain_path = os.path.join(PROJECT_ROOT, "data", "SynPAIN")
    syncpain_path = os.path.join(PROJECT_ROOT, "data", "SyncPain")

    if os.path.islink(synpain_path):
        target = os.readlink(synpain_path)
        print(f"Symlink already exists: {synpain_path} -> {target}")
        return
    elif os.path.isdir(synpain_path):
        print(f"Directory already exists at {synpain_path} (not a symlink)")
        return

    os.symlink(syncpain_path, synpain_path)
    print(f"Created symlink: {synpain_path} -> {syncpain_path}")


def main():
    print("Generating SynPAIN metadata.csv")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")

    # Create symlink first
    create_symlink()

    # Parse all images
    images = parse_images()
    if not images:
        raise RuntimeError("No images found!")

    # Assign splits by subject_id
    split_map = assign_splits(images)
    split_counts = defaultdict(int)
    for s in split_map.values():
        split_counts[s] += 1
    print(f"\nSubject split: train={split_counts['train']}, "
          f"val={split_counts['val']}, test={split_counts['test']}")

    # Create pairs
    pairs = create_pairs(images, split_map)

    # Shuffle before writing
    random.shuffle(pairs)

    # Write CSV
    write_csv(pairs, OUTPUT_CSV)

    # Print summary
    print_summary(pairs)

    # Verify a few paths exist
    print("\nVerifying sample paths...")
    for p in pairs[:5]:
        neu_full = os.path.join(OUTPUT_DIR, p["neutral_path"])
        expr_full = os.path.join(OUTPUT_DIR, p["expr_path"])
        neu_ok = os.path.isfile(neu_full)
        expr_ok = os.path.isfile(expr_full)
        print(f"  neutral={neu_ok} expr={expr_ok} | {p['neutral_path'][:60]}...")


if __name__ == "__main__":
    main()
