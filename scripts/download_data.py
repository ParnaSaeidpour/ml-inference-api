"""Download the NEU-DET steel surface defect dataset in YOLO format.

Source: KeenForgeAI/NEU-DET-corrected on Hugging Face, a cleaned copy of the
NEU Surface Defect Database (He et al., 2020) with duplicates removed and a
fixed, class-stratified 80/10/10 train/val/test split.

Usage:
    python scripts/download_data.py
"""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "neu-det"
REPO_ID = "KeenForgeAI/NEU-DET-corrected"

# Order matters: YOLO class ids 0..5 follow this list.
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def main() -> None:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DATA_DIR,
        allow_patterns=["images/**", "labels/**", "classes.txt"],
    )

    upstream = (DATA_DIR / "classes.txt").read_text().split()
    if upstream != CLASSES:
        raise SystemExit(f"class list changed upstream: {upstream}")

    # Write our own dataset config rather than trusting the downloaded one.
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (DATA_DIR / "data.yaml").write_text(
        f"path: {DATA_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n"
    )

    for split in ("train", "val", "test"):
        n = len(list((DATA_DIR / "images" / split).glob("*.jpg")))
        print(f"{split:5s} {n} images")


if __name__ == "__main__":
    main()
