import cv2
import os

VIDEO_PATH = r"E:\MiniProject\sample.mp4"
OUTPUT_DIR = r"E:\MiniProject\extracted_image"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"frame_{count:05d}.jpg"),
        frame
    )
    count += 1

cap.release()
print(f"✔ Extracted {count} frames")
