import cv2
import os

FRAME_DIR = r"E:\MiniProject\tpvm_output"
OUTPUT_VIDEO = r"E:\MiniProject\final_output_tpvm.mp4"
FPS = 60  # Must be 2x original for TPVM to work effectively!

frames = sorted(os.listdir(FRAME_DIR))
if not frames:
    raise RuntimeError("No frames found")

first = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
h, w, _ = first.shape

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    FPS,
    (w, h)
)

for fname in frames:
    frame = cv2.imread(os.path.join(FRAME_DIR, fname))
    out.write(frame)

out.release()
print("✔ Final video reconstructed")
