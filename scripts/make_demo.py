"""Render docs/predictions.jpg: the model's detections on the test fixtures.

Uses images from the held-out test split (tests/fixtures), one per class.

Usage:
    python -m scripts.make_demo
"""

import io
from pathlib import Path

from PIL import Image

from app import model as model_module

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
OUTPUT = ROOT / "docs" / "predictions.jpg"
TILE = 320


def main() -> None:
    model = model_module.load()
    paths = sorted(FIXTURES.glob("*.jpg"))
    cols = 3
    rows = -(-len(paths) // cols)
    grid = Image.new("RGB", (cols * TILE, rows * TILE), "white")

    for i, path in enumerate(paths):
        image = model_module.decode_image(path.read_bytes())
        jpeg = model_module.annotate(model, image.resize((TILE, TILE)))
        tile = Image.open(io.BytesIO(jpeg))
        grid.paste(tile, ((i % cols) * TILE, (i // cols) * TILE))

    OUTPUT.parent.mkdir(exist_ok=True)
    grid.save(OUTPUT, quality=88)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
