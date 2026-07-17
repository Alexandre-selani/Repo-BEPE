import os

import numpy as np
from PIL import Image
import glob
from multiprocessing import Pool, cpu_count

# --- UPDATE THESE ---
INPUT_DIR  = "/home/alexandreselani/Desktop/Segmentacao/Validation/Validation/validation_annotation"       # folder containing the RGB masks
OUTPUT_DIR = "/home/alexandreselani/Desktop/Segmentacao/Validation/Validation/labels"      # where to write the uint8 label masks
# --------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map RGB triplets to your class IDs
COLOR2CLASS = {
     2:(  0, 128,   0),   # Ground  → 2
    1:(128,   0,   0),   # Grass   → 1
    3:(128, 128,   0)   # Panicum → 3
}

def process_mask(path):
    """Load one RGB mask, map colors→labels, save as uint8 single‐channel."""
    # derive output path
    name = os.path.splitext(os.path.basename(path))[0] + ".png"
    out  = os.path.join(OUTPUT_DIR, name)

    # load and convert
    img_arr = np.array(Image.open(path).convert("L"))

    h, w = img_arr.shape
    

    print(img_arr.shape)
    lbl = np.full((h,w,3), (0,0,0))
    print(lbl.shape)
    # for each color, set class
    for cls, rgb in COLOR2CLASS.items():
        for y in range(h):
            for x in range(w):
                if img_arr[y,x]==cls:
                    lbl[y,x]=rgb
                
        
    #print(lbl)
    # save
    Image.fromarray(lbl.astype(np.uint8)).save(out)
    print(f"wrote {out}")

if __name__ == "__main__":
    # find all PNGs in the input folder
    mask_files = glob.glob(os.path.join(INPUT_DIR, "*.png"))
    if not mask_files:
        print("No PNGs found in", INPUT_DIR)
        exit(1)

    # use a pool to convert in parallel
    with Pool(cpu_count()) as pool:
        pool.map(process_mask, mask_files)
   
