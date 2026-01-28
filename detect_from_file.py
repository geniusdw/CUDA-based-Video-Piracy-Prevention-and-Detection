import cv2
import sys
import os
import numpy as np
# Import the Watermarker class from our main script
# Ensure cpu_hybrid_security.py is in the same folder or python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cpu_hybrid_security import Watermarker

def detect_in_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Analyzing: {file_path}")
    
    # Initialize Watermarker with SAME key and alpha as embedding
    # (Alpha doesn't matter for detection, but Block Size and Key do)
    watermarker = Watermarker(key=42, block_size=8)
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Could not open video/image.")
        return

    frame_count = 0
    total_score = 0
    detected_count = 0
    
    print("-" * 40)
    print(f"{'Frame':<10} | {'Correlation Score':<20} | {'Verdict'}")
    print("-" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run Detection
        score = watermarker.detect(frame)
        
        # Threshold Logic:
        # > 3.0: Very Strong Match
        # > 2.0: Strong Match
        # > 1.0: Weak Match
        # < 1.0: No Match (Noise)
        
        verdict = "NO MATCH"
        if score > 2.0:
            verdict = "DETECTED"
            detected_count += 1
        elif score > 1.0:
            verdict = "POSSIBLE"
            
        print(f"{frame_count:<10} | {score:>.4f}{' '*14} | {verdict}")
        
        total_score += score
        frame_count += 1
        
        # Analyze first 10 frames only for quick check
        if frame_count >= 10:
            break
            
    cap.release()
    print("-" * 40)
    
    avg_score = total_score / frame_count if frame_count > 0 else 0
    print(f"\nSummary:")
    print(f"Analyzed {frame_count} frames.")
    print(f"Average Correlation Score: {avg_score:.4f}")
    if avg_score > 2.0:
        print(">>> RESULT: WATERMARK CONFIRMED (PIRATED COPY DETECTED) <<<")
    else:
        print(">>> RESULT: CLEAN / NO WATERMARK DETECTED <<<")

if __name__ == "__main__":
    # Default to scanning the output file we just made
    target_file = "hybrid_protected_output.mp4"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    detect_in_file(target_file)
