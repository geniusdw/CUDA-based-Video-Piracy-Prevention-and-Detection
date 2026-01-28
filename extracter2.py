import cv2
import os
import numpy as np

# -------- PATHS --------
INPUT_DIR  = r"E:\MiniProject\extracted_image"
OUTPUT_DIR = r"E:\MiniProject\output_image"
MOIRE_MASK = r"E:\MiniProject\moire_mask.npy"
# ----------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- PARAMETERS --------
MOIRE_STRENGTH = 6.0   # start 4–8 (do NOT exceed 10)
DRIFT_SPEED = 0.2      # Speed of the "rolling" effect
# ---------------------------

# Pre-compute meshgrid for speed
frames = sorted(os.listdir(INPUT_DIR))
if not frames:
    print("No frames found!")
    exit()

# Read first frame to get dimensions
first_img = cv2.imread(os.path.join(INPUT_DIR, frames[0]))
h, w = first_img.shape[:2]
x = np.arange(w)
y = np.arange(h)
X, Y = np.meshgrid(x, y)

# Spatial frequencies for Moiré
fx = 6.8
fy = 7.3

print(f"Processing {len(frames)} frames with Dynamic Moiré...")

for i, fname in enumerate(frames):
    img = cv2.imread(os.path.join(INPUT_DIR, fname))
    if img is None:
        continue

    # Calculate dynamic phase shift
    phase = i * DRIFT_SPEED

    # Generate Moiré Mask on the fly
    # sin(2pi * (X+Y)/fx + phase)
    mask = (
        np.sin(2 * np.pi * (X + Y) / fx + phase) +
        np.sin(2 * np.pi * (X - Y) / fy)
    )
    # Normalize to -1..1 range equivalent
    # (Actually it's roughly -2 to 2, so let's clip/normalize simpler)
    # We essentially want a consistent range to multiply by STRENGTH.
    # The original script normalized to -1,1. Let's approximate.
    
    # Fast normalize: Mask is roughly between -2 and 2
    mask = mask * 0.5 

    # Convert to YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y_channel, Cr, Cb = cv2.split(ycrcb)
    Y_channel = Y_channel.astype(np.float32)

    # ---- Apply Dynamic Moiré ----
    Y_channel = np.clip(Y_channel + MOIRE_STRENGTH * mask, 0, 255)

    # Back to image
    Y_channel = Y_channel.astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([Y_channel, Cr, Cb]), cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), out)
    
    if i % 10 == 0:
        print(f"Processed frame {i}/{len(frames)}", end='\r')

print(f"\n✔ Dynamic Moiré induction applied to all {len(frames)} frames")
