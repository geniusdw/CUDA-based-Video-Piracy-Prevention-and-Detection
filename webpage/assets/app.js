(function () {
  const page = document.body.dataset.page;

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
      return "0 B";
    }

    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = bytes;
    let index = 0;

    while (value >= 1024 && index < units.length - 1) {
      value /= 1024;
      index += 1;
    }

    const precision = value >= 100 || index === 0 ? 0 : value >= 10 ? 1 : 2;
    return `${value.toFixed(precision)} ${units[index]}`;
  }

  function setStatus(element, message, tone) {
    element.textContent = message;
    element.dataset.tone = tone;
  }

  function setChip(element, label, tone) {
    element.textContent = label;
    element.className = `chip ${tone || "muted"}`;
  }

  function setVideo(preview, emptyState, source) {
    if (source) {
      preview.src = source;
      preview.style.display = "block";
      emptyState.style.display = "none";
      preview.load();
      return;
    }

    preview.pause();
    preview.removeAttribute("src");
    preview.load();
    preview.style.display = "none";
    emptyState.style.display = "grid";
  }

  function describeFile(file) {
    return [
      `Name: ${file.name}`,
      `Size: ${formatBytes(file.size)}`,
      `Type: ${file.type || "Unknown"}`,
    ].join(" | ");
  }

  function wirePreview(input, preview, emptyState, meta, onChange) {
    let objectUrl = null;

    input.addEventListener("change", () => {
      const [file] = input.files || [];

      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
        objectUrl = null;
      }

      if (!file) {
        meta.textContent = "No file selected yet.";
        setVideo(preview, emptyState, "");
        if (typeof onChange === "function") {
          onChange(null);
        }
        return;
      }

      objectUrl = URL.createObjectURL(file);
      meta.textContent = describeFile(file);
      setVideo(preview, emptyState, objectUrl);

      if (typeof onChange === "function") {
        onChange(file);
      }
    });
  }

  async function postVideo(endpoint, file) {
    const formData = new FormData();
    formData.append("video", file);

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    let data;
    try {
      data = await response.json();
    } catch (error) {
      throw new Error("The server returned an unreadable response.");
    }

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Processing failed.");
    }

    return data;
  }

  function initProtectPage() {
    const form = document.getElementById("protect-form");
    const fileInput = document.getElementById("protect-video");
    const fileMeta = document.getElementById("protect-file-meta");
    const inputPreview = document.getElementById("protect-input-preview");
    const inputEmpty = document.getElementById("protect-input-empty");
    const outputPreview = document.getElementById("protect-output-preview");
    const outputEmpty = document.getElementById("protect-output-empty");
    const submitButton = document.getElementById("protect-submit");
    const status = document.getElementById("protect-status");
    const pill = document.getElementById("protect-status-pill");
    const log = document.getElementById("protect-log");
    const downloadLink = document.getElementById("protect-download");

    let selectedFile = null;

    wirePreview(fileInput, inputPreview, inputEmpty, fileMeta, (file) => {
      selectedFile = file;
      setVideo(outputPreview, outputEmpty, "");
      downloadLink.href = "#";
      downloadLink.classList.add("disabled");
      log.textContent = "No processing has run yet.";
      setStatus(
        status,
        file ? "Ready to run protection." : "Upload a video, then run the protection process.",
        "info"
      );
      setChip(pill, file ? "Ready" : "Waiting", "muted");
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      if (!selectedFile) {
        setStatus(status, "Please choose a video before starting the protection process.", "warning");
        return;
      }

      submitButton.disabled = true;
      submitButton.textContent = "Processing on GPU...";
      setChip(pill, "Processing", "warning");
      setStatus(
        status,
        "Uploading video and running security_gpu.exe. This can take a little time for large videos.",
        "warning"
      );
      log.textContent = "Processing has started...";

      try {
        const data = await postVideo("/api/protect", selectedFile);
        const outputUrl = `${data.outputVideo}?t=${Date.now()}`;

        setVideo(outputPreview, outputEmpty, outputUrl);
        downloadLink.href = outputUrl;
        downloadLink.classList.remove("disabled");
        log.textContent = data.log || "Processing finished.";
        setStatus(status, `Protected video created successfully as ${data.outputFileName}.`, "success");
        setChip(pill, "Completed", "success");
      } catch (error) {
        log.textContent = error.message;
        setStatus(status, error.message, "danger");
        setChip(pill, "Failed", "danger");
      } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Generate Protected Video";
      }
    });
  }

  function initDetectorPage() {
    const form = document.getElementById("detect-form");
    const fileInput = document.getElementById("detect-video");
    const fileMeta = document.getElementById("detect-file-meta");
    const inputPreview = document.getElementById("detect-input-preview");
    const inputEmpty = document.getElementById("detect-input-empty");
    const submitButton = document.getElementById("detect-submit");
    const status = document.getElementById("detect-status");
    const pill = document.getElementById("detect-status-pill");
    const result = document.getElementById("detect-result");
    const analyzedPairs = document.getElementById("detect-pairs");
    const averageScore = document.getElementById("detect-score");
    const detectedPairs = document.getElementById("detect-detected-pairs");
    const requiredPairs = document.getElementById("detect-required-pairs");
    const log = document.getElementById("detect-log");

    let selectedFile = null;

    function resetResult() {
      result.textContent = "No result yet";
      result.className = "result-title neutral";
      analyzedPairs.textContent = "-";
      averageScore.textContent = "-";
      detectedPairs.textContent = "-";
      requiredPairs.textContent = "-";
      log.textContent = "No analysis has run yet.";
    }

    wirePreview(fileInput, inputPreview, inputEmpty, fileMeta, (file) => {
      selectedFile = file;
      resetResult();
      setStatus(
        status,
        file ? "Ready to run piracy detection." : "Upload a video, then run the detection process.",
        "info"
      );
      setChip(pill, file ? "Ready" : "Waiting", "muted");
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      if (!selectedFile) {
        setStatus(status, "Please choose a video before starting detection.", "warning");
        return;
      }

      submitButton.disabled = true;
      submitButton.textContent = "Analyzing on GPU...";
      setChip(pill, "Analyzing", "warning");
      setStatus(
        status,
        "Uploading video and running detector.exe. Large videos can take a while.",
        "warning"
      );
      log.textContent = "Analysis has started...";

      try {
        const data = await postVideo("/api/detect", selectedFile);

        result.textContent = data.verdict;
        result.className = `result-title ${data.isPirated ? "pirated" : "clean"}`;
        analyzedPairs.textContent = data.analyzedPairs ?? "-";
        averageScore.textContent = data.averageScore ?? "-";
        detectedPairs.textContent = data.detectedPairsSummary ?? "-";
        requiredPairs.textContent = data.minDetectedPairs ?? "-";
        log.textContent = data.log || "Analysis finished.";
        setStatus(
          status,
          `Detection complete. Final conclusion: ${data.verdict}.`,
          data.isPirated ? "danger" : "success"
        );
        setChip(pill, data.isPirated ? "Pirated" : "Not Pirated", data.isPirated ? "danger" : "success");
      } catch (error) {
        resetResult();
        log.textContent = error.message;
        setStatus(status, error.message, "danger");
        setChip(pill, "Failed", "danger");
      } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Run Piracy Detection";
      }
    });
  }

  if (page === "protect") {
    initProtectPage();
  }

  if (page === "detect") {
    initDetectorPage();
  }
})();
