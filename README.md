# Video Piracy Prevention System

A CUDA-accelerated video protection system that makes screen recordings look distorted while keeping the original video perfectly watchable to the human eye. Built for educational research into anti-camcorder technology.

---

## How It Works

The system uses two layers running simultaneously:

**Layer 1 — Anti-Capture (Temporal Psychovisual Modulation)**

Every source frame is split into two output frames — Frame A and Frame B. Frame A adds a luma and chroma pattern to the original. Frame B subtracts the exact same pattern. When played at 120fps on a 120Hz monitor, the human eye fuses A and B together (above the eye's ~60Hz flicker fusion limit) and sees the original clean video. A camera recording at 30fps captures only one of the two frames and sees the full distortion — horizontal banding from the CMOS rolling shutter effect, and colour noise from Bayer demosaicing.

**Layer 2 — Forensic Watermark (DCT Embedding)**

A pseudo-random invisible watermark is embedded into the DCT mid-band coefficients of every output frame. It survives the A+B averaging the detector uses, and survives H.264 re-encoding at typical pirate quality (CRF 24–32). The detector can confirm whether a recorded video came from a protected source

```
Source video (25fps)
        │
        ▼
┌─────────────────────────────┐
│    security_gpu_f.cu        │
│                             │
│  processSignalInterference  │  ← anti-capture kernel
│  embedDctWatermark (×2)     │  ← forensic watermark kernel
│                             │
└─────────────────────────────┘
        │
        ▼
Protected output (120fps)
  A, B, A, B, A, B ...
        │
   ┌────┴────┐
   ▼         ▼
 Human      Camera
 eye fuses  captures
 A+B=clean  one frame
            = distorted
```

---

## Files

| File | Purpose |
|------|---------|
| `security_gpu_f.cu` | Encoder — reads source video, produces protected 120fps output |
| `detector_f.cu` | Detector — analyzes a recording and reports whether watermark is present |
| `watermark_common.h` | Shared header — watermark key, DCT mid-band table, bit generator |
| `quick_check_output.ps1` | Fast verification script — checks FPS, A/B diff, generates diff image |
| `setup_cuda.ps1` | Build script for Windows + CUDA + OpenCV |

---

## Requirements

- **OS:** Windows 10/11 (x64)
- **GPU:** Any NVIDIA GPU with CUDA support (compute capability 3.5+)
- **CUDA Toolkit:** 11.0 or newer
- **OpenCV:** 4.x built with CUDA support
- **Display:** 120Hz monitor (required for the anti-capture effect to work)
- **Target camera tested:** FRONTECH Webcam 2255 HD 720p, 30fps, CMOS rolling shutter

---

## Build

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_cuda.ps1
```

Or manually with nvcc:

```bash
nvcc security_gpu_f.cu -o security_gpu.exe \
  -I"C:/opencv/build/include" \
  -L"C:/opencv/build/x64/vc16/lib" \
  -lopencv_world4xx -std=c++17

nvcc detector_f.cu -o detector.exe \
  -I"C:/opencv/build/include" \
  -L"C:/opencv/build/x64/vc16/lib" \
  -lopencv_world4xx -std=c++17
```

---

## Usage

**Protect a video:**
```powershell
.\security_gpu.exe
# uses sample.mp4 → hybrid_protected_output.mp4 by default

.\security_gpu.exe input.mp4 output.mp4
# custom paths
```

**Detect watermark in a recording:**
```powershell
.\detector.exe
# analyzes hybrid_protected_output.mp4 by default

.\detector.exe recorded_pirate_copy.mp4
# analyze any file

.\detector.exe recorded_pirate_copy.mp4 100
# limit analysis to first 100 frame pairs
```

**Quick check (verify encoder is working):**
```powershell
powershell -ExecutionPolicy Bypass -File .\quick_check_output.ps1 `
  -Video .\hybrid_protected_output.mp4 -Gain 8.0
```

This extracts Frame A and Frame B, amplifies the difference, saves `diff_amplified.png`, and reports the output FPS. If the diff image shows a pattern, the encoder is working.

**Play the protected video correctly (must show at 120fps):**
```html
<!-- save as test.html and open in Chrome/Edge -->
<!DOCTYPE html>
<html>
<body style="background:black;margin:0">
  <video src="hybrid_protected_output.mp4" autoplay loop
         style="width:100vw;height:100vh;object-fit:contain">
  </video>
</body>
</html>
```

> VLC often silently caps playback at 60fps. Use a browser or mpv with `--vo=gpu --video-sync=display-resample` for correct 120fps playback.

---

## Key Parameters

All tunable parameters are at the top of each file.

**security_gpu_f.cu**

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `kOutputFpsMultiplier` | `4.8f` | Source FPS × this = output FPS. `4.8 × 25 = 120fps`. Match to your monitor refresh rate. |
| `kLumaAmplitude` | `38.0f` | Brightness swing of A/B pair in 8-bit units. Higher = stronger camera artifact. |
| `kLumaSpatialFreq` | `16.0f` | Vertical sine cycles per frame height. 16 cycles / 720px = 45px bands. |
| `kChromaAmplitude` | `24.0f` | Checkerboard R/B offset. Targets Bayer demosaicing. Spatially averages to zero for the eye. |
| `kRollingShutterRowTime` | `46e-6f` | Row readout time in seconds for the target camera. FRONTECH at 720p/30fps ≈ 46µs/row. |
| `kDctEmbedStrength` | `3.0f` | Forensic watermark strength. Raise to 4–5 for better compression survival. |

**detector_f.cu**

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `kDetectThreshold` | `0.25` | Correlation score above which a frame pair is marked DETECTED. |
| `kPossibleThreshold` | `0.10` | Score above which a frame pair is marked POSSIBLE. |
| `kMinRatioShort` | `0.10` | Min fraction of DETECTED pairs to confirm watermark (clips < 40 pairs). |
| `kMinRatioMedium` | `0.05` | Same for clips 40–99 pairs. |
| `kMinRatioLong` | `0.02` | Same for clips ≥ 100 pairs. |

---

## Physics Behind the Effect

**Why the eye sees nothing**

The A/B frame pair alternates at `output_fps / 2 = 60Hz`. The human eye's critical flicker fusion frequency (CFF) is approximately 60Hz at normal display brightness. At or above this rate, the eye temporally integrates successive frames and perceives their average: `(original + delta) + (original - delta)) / 2 = original`. The luma and chroma deltas cancel exactly.

**Why the camera sees distortion**

The FRONTECH uses a CMOS rolling shutter that reads rows sequentially from top to bottom. At 30fps, the full frame takes ~33ms to read. During this time the display shows approximately 4 A/B transitions (33ms ÷ 8.33ms per frame at 120fps = 4 transitions). Top rows are captured during Frame A, rows a quarter down during Frame B, mid-rows during Frame A again, and so on — producing alternating bright/dark horizontal bands in the recording.

The checkerboard chroma pattern (1-pixel period, alternating +R/−B and −R/+B) aligns with the 2×2 RGGB Bayer grid on the camera sensor. Bayer demosaicing interpolates colour from adjacent pixels of opposite sign, producing false colour noise that the eye never sees because it spatially averages over many pixels.

**Why the watermark survives recording**

The DCT watermark is embedded in mid-frequency coefficients (DCT indices 1–4 in 8×8 blocks). H.264 compression discards high-frequency coefficients aggressively but preserves mid-frequency ones even at high CRF values. The detector averages each A/B pair before analyzing: `(A + B) / 2 = original + watermark`, because the watermark was embedded with the same sign in both A and B, while the anti-capture delta cancels out.

---

## Detector Output Example

```
Using Device: NVIDIA GeForce RTX 3060
----------------------------------------
Pair       | Correlation Score    | Verdict
----------------------------------------
0          | 0.3142               | DETECTED
1          | 0.2876               | DETECTED
2          | 0.1203               | POSSIBLE
3          | 0.3401               | DETECTED
...
----------------------------------------
Analyzed 120 TPVM frame-pairs.
Average Correlation Score: 0.2891
Detected pairs (>0.25): 98/120
Detected ratio: 0.8167 (min ratio for this length: 0.0200)
Min detected pairs required for confirm: 3
>>> RESULT: WATERMARK CONFIRMED (PIRATED COPY DETECTED) <<<
```

---

## Limitations

- Requires a 120Hz display. On 60Hz displays the A/B alternation is at 30Hz and visible to the human eye as flicker.
- The spatial pattern is static (same every frame). A future improvement is a per-frame dynamic parameter schedule seeded from a session key.
- All exports use the same `kWatermarkKey = 42`. For real forensic use, generate a unique key per screening derived from timestamp + seat ID to identify the specific leaker.
- High spatial frequencies (above ~120 cycles/frame-height) are attenuated by budget webcam lenses before reaching the sensor. Keep `kLumaSpatialFreq` at 16–32 for reliable camera artifacts.

---

## References

1. D. Zhai and M. Wu — *Preventing Illegal Recording of Movies Using Temporal Psychovisual Modulation*, IEEE Trans. Information Forensics and Security, 2014
2. Thomson Licensing — US Patent 7,760,952 — *Method and apparatus for preventing the copying of movies shown in theaters*, 2010
3. I. J. Cox et al. — *Digital Watermarking and Steganography*, 2nd ed., Morgan Kaufmann, 2008
4. NVIDIA Corporation — *CUDA C++ Programming Guide*, v12.x, 2024

---

## Disclaimer

This project is for **educational research purposes only**. It demonstrates temporal psychovisual modulation and DCT watermarking techniques as studied in academic literature on anti-piracy technology. Do not use for unauthorized protection of content you do not own.

---

*Built with CUDA, OpenCV, and a lot of rolling shutter physics.*
