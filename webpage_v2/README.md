# CUDA Video Shield Website (v2)

This folder contains a small local website with three pages:

- `index.html` - welcome page
- `protected-maker.html` - uploads a video and runs `security_gpu.exe`
- `piracy-detector.html` - uploads a video and runs `detector.exe`

## Run the website

From `E:\CUDA`, start the local server:

```powershell
python .\webpage_v2\server.py
```

Then open:

```text
http://127.0.0.1:8001
```

## Notes

- The server saves uploaded videos in `webpage_v2/runtime/uploads`.
- Generated protected videos are saved in `webpage_v2/runtime/generated`.
- The website expects `security_gpu.exe` and `detector.exe` to exist in the project root.
