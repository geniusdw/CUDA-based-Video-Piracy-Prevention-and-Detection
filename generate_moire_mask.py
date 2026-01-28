import numpy as np
import cv2
import time

# Interactive Moiré Visualizer
# Adjust these keys to tune:
# [w/s] Change fx
# [a/d] Change fy
# [q/e] Change Speed
# [ESC] Quit

H, W = 720, 1280
x = np.arange(W)
y = np.arange(H)
X, Y = np.meshgrid(x, y)

fx = 6.8
fy = 7.3
speed = 0.2
frame_count = 0

print("Controls: 'w/s' (fx), 'a/d' (fy), 'q/e' (Speed), 'ESC' (Exit)")

while True:
    phase = frame_count * speed
    
    # Calculate Moiré
    moire = (
        np.sin(2 * np.pi * (X + Y) / fx + phase) +
        np.sin(2 * np.pi * (X - Y) / fy)
    )
    
    # Normalize to 0-1 for display
    disp = (moire * 0.5 + 0.5)
    
    cv2.imshow("Dynamic Moire Preview", disp)
    
    key = cv2.waitKey(16) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('w'): fx += 0.1
    elif key == ord('s'): fx -= 0.1
    elif key == ord('a'): fy -= 0.1
    elif key == ord('d'): fy += 0.1
    elif key == ord('q'): speed -= 0.05
    elif key == ord('e'): speed += 0.05
    
    frame_count += 1

cv2.destroyAllWindows()
print(f"Final params: fx={fx:.2f}, fy={fy:.2f}, speed={speed:.2f}")
