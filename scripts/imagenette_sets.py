"""Make the calibration and evaluation picture sets used for the real-model results in the README.

    python scripts/imagenette_sets.py data/imagenette2-160 data

Copies a seeded, class-balanced sample of 20 pictures per class from ``train`` into ``data/calib`` and 30 per
class from ``val`` into ``data/val``. The two splits never share a picture, so INT8 is not evaluated on the
pictures it was calibrated on.
"""

import random
import shutil
import sys
from pathlib import Path

SETS = (("train", "calib", 20), ("val", "val", 30))


def main(source: Path, target: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    for split, name, per_class in SETS:
        out = target / name
        out.mkdir(parents=True, exist_ok=True)
        for cls in sorted(p for p in (source / split).iterdir() if p.is_dir()):  # skips stray files such as .DS_Store
            for picture in rng.sample(sorted(cls.glob("*.JPEG")), per_class):
                shutil.copy2(picture, out / picture.name)
        print(f"{out}: {len(list(out.glob('*.JPEG')))} pictures")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
