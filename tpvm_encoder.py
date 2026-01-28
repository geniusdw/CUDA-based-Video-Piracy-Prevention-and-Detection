import cv2
import numpy as np
import os

# -------- PARAMETERS --------
INPUT_DIR  = r"E:\MiniProject\extracted_image"
OUTPUT_DIR = r"E:\MiniProject\tpvm_output"
PATTERN_STRENGTH = 40  # 0-255. Higher = Stronger Flicker/Protection. Start around 30-50.
# ---------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

frames = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))])

if not frames:
    print("No frames found in input directory!")
    exit()

# Load first frame to detect size
sample = cv2.imread(os.path.join(INPUT_DIR, frames[0]))
h, w = sample.shape[:2]

# Generate the Carrier Pattern (Checkerboard / High-Freq Noise)
# A checkerboard is good for anti-aliasing attacks.
# 1s and -1s
pattern = np.indices((h, w)).sum(axis=0) % 2
pattern = pattern.astype(np.float32)
pattern[pattern == 0] = -1  # Now pattern is -1 and +1

# Alternatively: Random Noise Pattern (uncomment for randomness)
# pattern = np.random.choice([-1, 1], size=(h, w)).astype(np.float32)

print(f"Processing {len(frames)} frames into Counter-Phase Flicker pairs...")

count = 0
for fname in frames:
    img = cv2.imread(os.path.join(INPUT_DIR, fname))
    if img is None: continue
    
    img_float = img.astype(np.float32)
    
    # Calculate Modulation
    # Frame A = Image + (Pattern * Strength)
    # Frame B = Image - (Pattern * Strength)
    
    # Clipping is critical so we don't overflow
    mod = pattern * PATTERN_STRENGTH
    
    # We apply modulation to all channels or just Luminance?
    # Applying to all channels creates localized luminance flicker.
    # To make it "Chromatic", we could invert it per channel.
    # Let's stick to Luminance/Global flicker first as it's the "Killer" physics solution.
    
    # Expand mod to 3 channels
    mod_3c = np.dstack([mod] * 3)
    
    frame_a = np.clip(img_float + mod_3c, 0, 255).astype(np.uint8)
    frame_b = np.clip(img_float - mod_3c, 0, 255).astype(np.uint8)
    
    # Save Sequence: Frame 1A, Frame 1B, Frame 2A, Frame 2B...
    # Naming must ensure order.
    base_name = os.path.splitext(fname)[0]
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_0_A.jpg"), frame_a)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_1_B.jpg"), frame_b)
    
    count += 1
    if count % 10 == 0:
        print(f"Processed {count}/{len(frames)}", end='\r')

print(f"\n✔ Done! {count*2} frames generated in {OUTPUT_DIR}")
print("IMPORTANT: Assemble these frames at DOUBLE the original framerate (e.g., 60fps or 120fps).")
