"""
Prepare BF images from merged BF+GFP JPGs for Morgana GUI.

Your JPGs are RGB composites where:
  - Red channel = BF (clean, no GFP)
  - Green channel = BF + GFP overlay

This script extracts the BF channel and saves as TIF files
in the folder structure Morgana expects.

Output:
  benchmarking_segment_tools/morgana_input/
    control_initial/
      033.tif, 043.tif
    control_elongated/
      033.tif, 043.tif
    treated_initial/
      049.tif, 055.tif, ...
    treated_elongation/
      049.tif, 055.tif, ...

Then point Morgana GUI to each of these folders.
"""

import sys
from pathlib import Path
import numpy as np
from skimage import io as skio
import tifffile

_SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = r"Z:\Megha_Kattimani\Imaging\Zeiss\Benchmarking_Segmentation tools\JPG"
OUTPUT_DIR = _SCRIPT_DIR / 'morgana_input'


def main():
    base = Path(INPUT_DIR)
    folder_map = {
        ('control', 'initial'): 'control_initial',
        ('control', 'elongated'): 'control_elongated',
        ('treated', 'initial'): 'treated_initial',
        ('treated', 'elongation'): 'treated_elongation',
    }

    for (cond, tp), out_name in folder_map.items():
        src = base / cond / tp
        if not src.exists():
            continue

        dst = OUTPUT_DIR / out_name
        dst.mkdir(parents=True, exist_ok=True)

        jpgs = sorted(src.glob('*.jpg'))
        print(f"\n{out_name}/  ({len(jpgs)} images)")

        for jpg in jpgs:
            img_rgb = skio.imread(str(jpg))
            # BF = red channel (no GFP contamination)
            bf = img_rgb[:, :, 0]  # uint8

            # Extract sample ID from filename
            name = jpg.stem
            sample_id = name
            for part in name.split('#'):
                if part and part[0].isdigit():
                    sample_id = part.split('.')[0].split('_')[0]
                    break

            out_path = dst / f"{sample_id}.tif"
            tifffile.imwrite(str(out_path), bf)
            print(f"  {sample_id}.tif  ({bf.shape}, {bf.dtype})")

    print(f"\nDone! BF images saved to: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Open a terminal and run:  python -m morgana")
    print(f"  2. In the GUI, point to one folder at a time (e.g. {OUTPUT_DIR / 'control_initial'})")
    print(f"  3. Create training data (draw masks on 2-3 images)")
    print(f"  4. Train the classifier")
    print(f"  5. Generate masks for all images")
    print(f"  6. Repeat for each folder")


if __name__ == '__main__':
    main()
