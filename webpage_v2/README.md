# CUDA Video Shield Website

This folder contains a small local website with three pages:

- `index.html` - welcome page
- `protected-maker.html` - uploads a video and runs `security_gpu.exe`
- `piracy-detector.html` - uploads a video and runs `detector.exe`

## Run the website

From `E:\CUDA`, start the local server:

```powershell
python .\webpage\server.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Notes

- The server saves uploaded videos in `webpage/runtime/uploads`.
- Generated protected videos are saved in `webpage/runtime/generated`.
- The website expects `security_gpu.exe` and `detector.exe` to exist in the project root.
