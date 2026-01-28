import cv2
import numpy as np
import os

class Watermarker:
    def __init__(self, key=42, block_size=8, alpha=5.0):
        """
        Initialize DCT Watermarker.
        key: Seed for random watermark generation.
        block_size: DCT block size (usually 8).
        alpha: Watermark embedding strength.
        """
        self.key = key
        self.block_size = block_size
        self.alpha = alpha
        self.watermark = None
        
        # We target mid-frequency coefficients for robustness and invisibility
        # Zig-zag order indices for 8x8 block (approximate mid-band)
        self.mid_band_indices = [
            (3,0), (2,1), (1,2), (0,3),
            (4,0), (3,1), (2,2), (1,3), (0,4), 
            (5,0), (4,1), (3,2), (2,3), (1,4), (0,5)
        ]

    def _generate_watermark(self, total_blocks):
        """Generate a pseudo-random sequence of length = total_blocks * len(mid_band)."""
        np.random.seed(self.key)
        # Sequence of +1 and -1
        return np.random.choice([1, -1], size=total_blocks * len(self.mid_band_indices))

    def embed(self, frame):
        """
        Embed watermark into the Y channel of the frame using Block-DCT.
        """
        h, w = frame.shape[:2]
        
        # Trim to multiple of block_size
        h_trim = (h // self.block_size) * self.block_size
        w_trim = (w // self.block_size) * self.block_size
        frame_trimmed = frame[:h_trim, :w_trim]
        
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(frame_trimmed, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = y.astype(np.float32)
        
        # Calculate number of blocks
        h_blocks = h_trim // self.block_size
        w_blocks = w_trim // self.block_size
        total_blocks = h_blocks * w_blocks
        
        # Generate watermark if not already (or reuse if same size)
        if self.watermark is None or len(self.watermark) != total_blocks * len(self.mid_band_indices):
            self.watermark = self._generate_watermark(total_blocks)
            
        print(f"Embedding Watermark... Blocks: {h_blocks}x{w_blocks} = {total_blocks}")

        # Block Processing
        idx = 0
        for r in range(0, h_trim, self.block_size):
            for c in range(0, w_trim, self.block_size):
                block = y[r:r+self.block_size, c:c+self.block_size]
                dct_block = cv2.dct(block)
                
                # Embed into mid-band
                for (u, v) in self.mid_band_indices:
                    w_bit = self.watermark[idx]
                    dct_block[u, v] += self.alpha * w_bit
                    idx += 1
                
                y[r:r+self.block_size, c:c+self.block_size] = cv2.idct(dct_block)
        
        # Merge and return
        y = np.clip(y, 0, 255).astype(np.uint8)
        watermarked_frame = cv2.merge([y, cr, cb])
        return cv2.cvtColor(watermarked_frame, cv2.COLOR_YCrCb2BGR)

    def detect(self, frame):
        """
        Detect watermark in a suspicious frame.
        Returns correlation score.
        """
        h, w = frame.shape[:2]
        h_trim = (h // self.block_size) * self.block_size
        w_trim = (w // self.block_size) * self.block_size
        frame_trimmed = frame[:h_trim, :w_trim]
        
        ycrcb = cv2.cvtColor(frame_trimmed, cv2.COLOR_BGR2YCrCb)
        y, _, _ = cv2.split(ycrcb)
        y = y.astype(np.float32)
        
        h_blocks = h_trim // self.block_size
        w_blocks = w_trim // self.block_size
        total_blocks = h_blocks * w_blocks
        
        if self.watermark is None:
            self.watermark = self._generate_watermark(total_blocks)
            
        extracted_signal = []
        
        for r in range(0, h_trim, self.block_size):
            for c in range(0, w_trim, self.block_size):
                block = y[r:r+self.block_size, c:c+self.block_size]
                dct_block = cv2.dct(block)
                
                # Extract from mid-band
                for (u, v) in self.mid_band_indices:
                    extracted_signal.append(dct_block[u, v])
                    
        extracted_signal = np.array(extracted_signal)
        
        # We don't have the original frame to subtract, so we correlate directly.
        # DCT coefficients are roughly 0-mean for AC components, so high correlation with W 
        # indicates presence.
        
        # Normalize
        correlation = np.dot(extracted_signal, self.watermark) / len(self.watermark)
        return correlation


class TpvmModulator:
    def __init__(self, strength=30):
        self.strength = strength
        
    def apply_tpvm(self, frame):
        """
        Takes a frame and returns Frame A and Frame B.
        Frame A = Frame + Pattern
        Frame B = Frame - Pattern
        """
        h, w = frame.shape[:2]
        
        # Generate simple checkerboard pattern for demo
        # (Ideally this should be high-freq noise or specific spectral pattern)
        rows = np.arange(h).reshape(-1, 1)
        cols = np.arange(w).reshape(1, -1)
        # 1 if (r+c) is even, -1 if odd
        pattern = ((rows + cols) % 2) * 2 - 1 
        pattern = pattern.astype(np.float32) * self.strength
        
        # Expand to 3 channels
        pattern_3c = np.dstack([pattern]*3)
        
        frame_float = frame.astype(np.float32)
        
        frame_a = np.clip(frame_float + pattern_3c, 0, 255).astype(np.uint8)
        frame_b = np.clip(frame_float - pattern_3c, 0, 255).astype(np.uint8)
        
        return frame_a, frame_b

def main():
    input_video = "sample.mp4"
    output_video = "hybrid_protected_output.mp4"
    
    if not os.path.exists(input_video):
        print("Detailed instruction: Please make sure 'sample.mp4' exists in e:/MiniProject/")
        # create a dummy sample if not exists for testing purposes
        print("Creating dummy sample.mp4 for demonstration...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Sample Video", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out = cv2.VideoWriter(input_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for _ in range(30): out.write(dummy_frame)
        out.release()
        
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # We will output at DOUBLE the framerate because TPVM splits 1 frame into 2
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps * 2, (width, height))
    
    watermarker = Watermarker(alpha=10.0)
    modulator = TpvmModulator(strength=40)
    
    frame_count = 0
    print("Starting Hybrid Processing (Watermarking + TPVM)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # ---------------------------
        # STEP 1: DETECTION LAYER
        # Embed Invisible Watermark
        # ---------------------------
        wm_frame = watermarker.embed(frame)
        
        # ---------------------------
        # STEP 2: PREVENTION LAYER
        # Apply Temporal Modulation
        # ---------------------------
        frame_a, frame_b = modulator.apply_tpvm(wm_frame)
        
        # Save both frames
        out.write(frame_a)
        out.write(frame_b)
        
        # ---------------------------
        # VALIDATION / TEST LOGGING
        # Check if we can detect watermark in Frame A (simulating a camera capture of just one phase)
        # ---------------------------
        if frame_count % 10 == 0:
            score = watermarker.detect(frame_a)
            print(f"Frame {frame_count}: WM Correlation in Frame A = {score:.2f} (Threshold > 2.0 usually)")
        
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"\nProcessing Complete. Saved to {output_video}")
    print("Play this video. It will look fast/flickery on standard players because they can't handle 60/120Hz blending perfectly,")
    print("but it demonstrates the frame splitting.")

if __name__ == "__main__":
    main()
