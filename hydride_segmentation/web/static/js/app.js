/* Browser logic for the intranet segmentation app.
   Plain ES5-compatible JavaScript with no framework and no build step, so the
   deployed server needs nothing beyond Python. */

(function () {
  "use strict";

  var VIEW_CAPTIONS = {
    overlay_png_b64: "Detected features drawn over your image. Judge quality here first.",
    input_png_b64: "The image exactly as the server processed it, after any downscaling.",
    mask_png_b64: "Binary segmentation on its own. This is the image the download button saves.",
    fn_classification_png_b64: "Green outlines counted towards Fn, red not counted. Labels show the angle and length of the largest counted features.",
    fn_angle_threshold_png_b64: "Orientation histogram with the Fn threshold marked. Shows how sensitive Fn is to where you set the threshold.",
    orientation_map_png_b64: "Detected features coloured by their measured orientation angle.",
    size_histogram_png_b64: "Distribution of feature sizes across the image.",
    angle_histogram_png_b64: "Distribution of feature orientation angles across the image."
  };

  var METRIC_LABELS = {
    area_fraction: "Area fraction",
    fn_count: "Fn, count-based",
    fn_length_weighted: "Fn, length-weighted",
    fn_count_numerator: "Radial hydrides counted",
    fn_count_denominator: "Hydrides measured",
    fn_length_numerator_px: "Radial hydride length (px)",
    fn_length_denominator_px: "Total hydride length (px)",
    fn_angle_threshold_deg: "Angle threshold used (deg)",
    fn_excluded_small_features: "Features excluded as too small",
    hydride_area_fraction: "Hydride area fraction",
    hydride_area_fraction_percent: "Hydride area (%)",
    hydride_total_area_pixels: "Total hydride area (px)",
    hydride_count: "Hydride count",
    hydride_density_per_megapixel: "Hydride density (per megapixel)",
    mask_area_fraction: "Mask area fraction",
    min_feature_pixels: "Minimum feature size used (px)",
    excluded_small_features: "Features excluded as too small",
    orientation_mean_deg: "Mean orientation (deg)",
    orientation_median_deg: "Median orientation (deg)",
    orientation_std_deg: "Orientation spread (deg)",
    orientation_p10_deg: "Orientation 10th percentile (deg)",
    orientation_p90_deg: "Orientation 90th percentile (deg)",
    orientation_min_deg: "Minimum orientation (deg)",
    orientation_max_deg: "Maximum orientation (deg)",
    orientation_alignment_index: "Alignment index",
    orientation_entropy_bits: "Orientation entropy (bits)",
    size_mean_pixels: "Mean feature size (px)",
    size_median_pixels: "Median feature size (px)",
    size_std_pixels: "Feature size spread (px)",
    size_p10_pixels: "Feature size 10th percentile (px)",
    size_p90_pixels: "Feature size 90th percentile (px)",
    size_min_pixels: "Smallest feature (px)",
    size_max_pixels: "Largest feature (px)"
  };

  function $(id) { return document.getElementById(id); }

  function humanizeKey(key) {
    if (METRIC_LABELS[key]) { return METRIC_LABELS[key]; }
    return key.replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function formatValue(value) {
    if (value === null || value === undefined) { return "-"; }
    if (typeof value === "number") {
      if (!isFinite(value)) { return String(value); }
      if (Math.abs(value) > 0 && Math.abs(value) < 0.001) { return value.toExponential(3); }
      if (Number.isInteger(value)) { return String(value); }
      return value.toFixed(4);
    }
    return String(value);
  }

  /* Collapsible in-app help ------------------------------------------- */

  function bindHelpToggles() {
    var toggles = document.querySelectorAll(".help-toggle");
    for (var i = 0; i < toggles.length; i++) {
      (function (toggle) {
        toggle.addEventListener("click", function () {
          var target = $(toggle.getAttribute("data-help-target"));
          if (!target) { return; }
          var open = target.hasAttribute("hidden");
          if (open) { target.removeAttribute("hidden"); } else { target.setAttribute("hidden", ""); }
          toggle.setAttribute("aria-expanded", open ? "true" : "false");
        });
      })(toggles[i]);
    }
  }

  /* Server status chip ------------------------------------------------ */

  function renderStatus(chip, payload) {
    if (!payload || payload.ok !== true) {
      chip.className = "status-chip status-chip--error";
      chip.textContent = "Server unreachable";
      return false;
    }
    if (payload.trained_model_count === 0) {
      chip.className = "status-chip status-chip--pending";
      chip.textContent = "Conventional only - no trained model installed";
      return true;
    }
    if (!payload.preload_finished) {
      chip.className = "status-chip status-chip--pending";
      chip.textContent = "Loading models...";
      return false;
    }
    chip.className = "status-chip status-chip--ready";
    chip.textContent = "Ready - " + payload.ready_model_count + " model(s) loaded";
    return true;
  }

  function pollStatus(onUpdate) {
    var chip = $("server-status");
    if (!chip) { return; }
    var attempts = 0;

    function tick() {
      attempts += 1;
      fetch("api/status", { headers: { "Accept": "application/json" } })
        .then(function (res) { return res.json(); })
        .then(function (payload) {
          var settled = renderStatus(chip, payload);
          if (onUpdate) { onUpdate(payload); }
          if (!settled && attempts < 60) { window.setTimeout(tick, 2000); }
        })
        .catch(function () {
          chip.className = "status-chip status-chip--error";
          chip.textContent = "Server unreachable";
          if (attempts < 10) { window.setTimeout(tick, 5000); }
        });
    }
    tick();
  }

  /* Workspace --------------------------------------------------------- */

  function initWorkspace() {
    var dataNode = $("bootstrap-data");
    if (!dataNode) { return; }
    var data = JSON.parse(dataNode.textContent);

    var state = {
      file: null, sampleId: "", libraryId: "", result: null, view: "overlay_png_b64",
      running: false, lastEventSequence: 0, jobEvents: [], previewObjectUrl: "",
      libraryLoaded: false
    };

    var dropzone = $("dropzone");
    var fileInput = $("file-input");
    var fileName = $("file-name");
    var selectionPreview = $("selection-preview");
    var dropzonePlaceholder = $("dropzone-placeholder");
    var previewReplaceHint = $("preview-replace-hint");
    var modelSelect = $("model-select");
    var modelDescription = $("model-description");
    var modelWarning = $("model-warning");
    var conventionalFieldset = $("conventional-fieldset");
    var runBtn = $("run-btn");
    var runStatus = $("run-status");
    var errorBox = $("error-box");
    var progressPanel = $("progress-panel");
    var progressBar = $("progress-bar");
    var progressFill = $("progress-fill");
    var progressStage = $("progress-stage");
    var progressPercent = $("progress-percent");
    var progressMessage = $("progress-message");
    var jobLogOutput = $("job-log-output");
    var libraryOpenBtn = $("library-open");
    var libraryCloseBtn = $("library-close");
    var libraryDialog = $("library-dialog");
    var libraryGrid = $("library-grid");
    var libraryStatus = $("library-status");
    var fallbackSamples = $("fallback-samples");
    var emptyState = $("empty-state");
    var resultArea = $("result-area");
    var resultImage = $("result-image");
    var viewCaption = $("view-caption");
    var metricGroups = $("metric-groups");
    var runMeta = $("run-meta");
    var allControls = data.controls.concat(data.quantificationControls || []);

    /* -- models -- */

    function populateModels(models, defaultId) {
      modelSelect.innerHTML = "";
      for (var i = 0; i < models.length; i++) {
        var model = models[i];
        var option = document.createElement("option");
        option.value = model.model_id;
        option.textContent = model.available ? model.display_name : model.display_name + "  (unavailable)";
        option.disabled = !model.available;
        modelSelect.appendChild(option);
      }
      var wanted = defaultId || data.defaultModelId;
      for (var j = 0; j < modelSelect.options.length; j++) {
        if (modelSelect.options[j].value === wanted && !modelSelect.options[j].disabled) {
          modelSelect.selectedIndex = j;
          break;
        }
      }
      onModelChanged();
    }

    function currentModel() {
      var id = modelSelect.value;
      for (var i = 0; i < data.models.length; i++) {
        if (data.models[i].model_id === id) { return data.models[i]; }
      }
      return null;
    }

    function onModelChanged() {
      var model = currentModel();
      if (!model) { return; }
      modelDescription.textContent = model.description || "";
      if (model.is_conventional) {
        conventionalFieldset.removeAttribute("hidden");
      } else {
        conventionalFieldset.setAttribute("hidden", "");
      }
      if (!model.available) {
        modelWarning.textContent = model.availability_message ||
          "This model has no checkpoint installed on the server.";
        modelWarning.removeAttribute("hidden");
      } else if (model.warm_state === "loading") {
        modelWarning.textContent = "This model is still loading into memory. The first run may take longer.";
        modelWarning.removeAttribute("hidden");
      } else {
        modelWarning.setAttribute("hidden", "");
      }
      updateRunButton();
    }

    /* -- image selection -- */

    function clearPreview() {
      if (state.previewObjectUrl) {
        URL.revokeObjectURL(state.previewObjectUrl);
        state.previewObjectUrl = "";
      }
      selectionPreview.removeAttribute("src");
      selectionPreview.alt = "";
      selectionPreview.setAttribute("hidden", "");
      previewReplaceHint.setAttribute("hidden", "");
      dropzonePlaceholder.removeAttribute("hidden");
      dropzone.classList.remove("has-preview");
    }

    function showPreview(url, alt, objectUrl) {
      clearPreview();
      state.previewObjectUrl = objectUrl ? url : "";
      selectionPreview.src = url;
      selectionPreview.alt = alt;
      selectionPreview.removeAttribute("hidden");
      previewReplaceHint.removeAttribute("hidden");
      dropzonePlaceholder.setAttribute("hidden", "");
      dropzone.classList.add("has-preview");
    }

    selectionPreview.addEventListener("error", function () {
      clearPreview();
      fileName.textContent += " Preview is unavailable in this browser, but the image remains selected.";
    });

    function setFile(file) {
      var extension = file && file.name.indexOf(".") >= 0
        ? file.name.split(".").pop().toLowerCase() : "";
      var allowed = data.allowedExtensions || [];
      if (!file || allowed.indexOf(extension) < 0) {
        state.file = null;
        state.sampleId = "";
        state.libraryId = "";
        fileInput.value = "";
        fileName.textContent = "";
        clearPreview();
        showError("Choose a supported image: " + allowed.join(", ").toUpperCase() + ".");
        updateRunButton();
        return;
      }
      if (file.size > data.maxUploadMb * 1024 * 1024) {
        state.file = null;
        state.sampleId = "";
        state.libraryId = "";
        fileInput.value = "";
        fileName.textContent = "";
        clearPreview();
        showError("That image is larger than the " + data.maxUploadMb + " MB limit.");
        updateRunButton();
        return;
      }
      state.file = file;
      state.sampleId = "";
      state.libraryId = "";
      var objectUrl = URL.createObjectURL(file);
      showPreview(objectUrl, "Preview of " + file.name, true);
      fileName.textContent = "Validated filename and size: " + file.name + " (" +
        (file.size / (1024 * 1024)).toFixed(2) + " MB). Contents are checked before queuing.";
      clearError();
      updateRunButton();
    }

    function setSample(sampleId, label, sampleUrl) {
      state.file = null;
      state.sampleId = sampleId;
      state.libraryId = "";
      fileInput.value = "";
      showPreview(sampleUrl, "Preview of example image: " + label, false);
      fileName.textContent = "Example image: " + label;
      clearError();
      updateRunButton();
    }

    function setLibraryImage(imageId, label, imageUrl) {
      state.file = null;
      state.sampleId = "";
      state.libraryId = imageId;
      fileInput.value = "";
      showPreview(imageUrl, "Preview of library image: " + label, false);
      fileName.textContent = "Library image: " + label;
      clearError();
      updateRunButton();
    }

    function updateRunButton() {
      var model = currentModel();
      var hasImage = Boolean(state.file || state.sampleId || state.libraryId);
      runBtn.disabled = state.running || !hasImage || !model || !model.available;
      if (state.running) {
        runStatus.textContent = "Running...";
      } else if (!hasImage) {
        runStatus.textContent = "Choose an image to begin.";
      } else {
        runStatus.textContent = "";
      }
    }

    dropzone.addEventListener("click", function () { fileInput.click(); });
    dropzone.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); }
    });
    fileInput.addEventListener("change", function () {
      if (fileInput.files && fileInput.files.length) { setFile(fileInput.files[0]); }
    });
    ["dragenter", "dragover"].forEach(function (name) {
      dropzone.addEventListener(name, function (event) {
        event.preventDefault(); dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      dropzone.addEventListener(name, function (event) {
        event.preventDefault(); dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", function (event) {
      var files = event.dataTransfer && event.dataTransfer.files;
      if (files && files.length) { setFile(files[0]); }
    });

    var sampleButtons = document.querySelectorAll(".sample-btn");
    for (var s = 0; s < sampleButtons.length; s++) {
      (function (button) {
        button.addEventListener("click", function () {
          setSample(
            button.getAttribute("data-sample-id"),
            button.getAttribute("data-sample-label") || button.textContent.trim(),
            button.getAttribute("data-sample-url")
          );
        });
      })(sampleButtons[s]);
    }

    /* -- server image library -- */

    function renderLibrary(images) {
      libraryGrid.innerHTML = "";
      for (var i = 0; i < images.length; i++) {
        (function (image) {
          var button = document.createElement("button");
          button.type = "button";
          button.className = "library-item";
          button.setAttribute("role", "listitem");
          button.title = image.filename;

          var thumb = document.createElement("img");
          thumb.src = image.thumb_url;
          thumb.alt = "";
          thumb.loading = "lazy";
          button.appendChild(thumb);

          var caption = document.createElement("span");
          caption.textContent = image.label;
          button.appendChild(caption);

          button.addEventListener("click", function () {
            setLibraryImage(image.id, image.label, image.url);
            closeLibrary();
          });
          libraryGrid.appendChild(button);
        })(images[i]);
      }
    }

    function loadLibrary() {
      libraryStatus.textContent = "Loading the library...";
      libraryStatus.removeAttribute("hidden");
      return fetch(data.libraryUrl, { headers: { "Accept": "application/json" } })
        .then(function (response) { return response.json(); })
        .then(function (payload) {
          if (!payload.ok || !payload.available || !payload.images.length) {
            // The folder was removed or emptied since the page loaded. Send the
            // user back to the built-in examples rather than an empty dialog.
            libraryStatus.textContent =
              "No image library is available on this server. Use the example images instead.";
            libraryGrid.innerHTML = "";
            showFallbackSamples();
            return;
          }
          libraryStatus.setAttribute("hidden", "");
          renderLibrary(payload.images);
          state.libraryLoaded = true;
        })
        .catch(function () {
          libraryStatus.textContent = "The library could not be loaded. Please try again.";
          libraryStatus.removeAttribute("hidden");
        });
    }

    function showFallbackSamples() {
      if (libraryOpenBtn) { libraryOpenBtn.setAttribute("hidden", ""); }
      if (fallbackSamples) { fallbackSamples.removeAttribute("hidden"); }
    }

    function openLibrary() {
      if (!libraryDialog) { return; }
      if (typeof libraryDialog.showModal === "function") {
        libraryDialog.showModal();
      } else {
        // Very old browsers without <dialog>; the grid still works inline.
        libraryDialog.setAttribute("open", "");
      }
      // Reload each time so images copied onto the server mid-session appear.
      loadLibrary();
    }

    function closeLibrary() {
      if (!libraryDialog) { return; }
      if (typeof libraryDialog.close === "function") {
        libraryDialog.close();
      } else {
        libraryDialog.removeAttribute("open");
      }
    }

    if (libraryOpenBtn && libraryDialog) {
      if (data.libraryAvailable) {
        libraryOpenBtn.removeAttribute("hidden");
        if (fallbackSamples) { fallbackSamples.setAttribute("hidden", ""); }
      } else {
        showFallbackSamples();
      }
      libraryOpenBtn.addEventListener("click", openLibrary);
      libraryCloseBtn.addEventListener("click", closeLibrary);
      // A modal <dialog> closes itself on Escape, but the attribute fallback
      // above does not, so handle the key rather than trapping users in it.
      libraryDialog.addEventListener("keydown", function (event) {
        if (event.key === "Escape" || event.key === "Esc") {
          event.preventDefault();
          closeLibrary();
        }
      });
      // Clicking the backdrop closes the dialog, matching how users expect
      // modals to behave; clicks inside the panel must not bubble out to it.
      libraryDialog.addEventListener("click", function (event) {
        if (event.target === libraryDialog) { closeLibrary(); }
      });
    }

    modelSelect.addEventListener("change", onModelChanged);

    $("reset-params").addEventListener("click", function () {
      for (var i = 0; i < allControls.length; i++) {
        var control = allControls[i];
        var input = $("ctl-" + control.key);
        if (input) { input.value = control.default; }
      }
    });

    /* -- errors -- */

    function showError(message) {
      errorBox.textContent = message;
      errorBox.removeAttribute("hidden");
    }
    function clearError() {
      errorBox.textContent = "";
      errorBox.setAttribute("hidden", "");
    }

    function resetProgress() {
      state.lastEventSequence = 0;
      state.jobEvents = [];
      jobLogOutput.innerHTML = "";
      progressStage.textContent = "Validating";
      progressPercent.textContent = "0%";
      progressMessage.textContent = "Checking the image and request.";
      progressFill.style.width = "0%";
      progressBar.setAttribute("aria-valuenow", "0");
      progressPanel.removeAttribute("hidden");
    }

    function updateProgress(payload) {
      var percent = Math.max(0, Math.min(100, Number(payload.percent || 0)));
      progressStage.textContent = humanizeKey(payload.stage || payload.state || "working");
      progressPercent.textContent = Math.round(percent) + "%";
      progressMessage.textContent = payload.message || "Working.";
      progressFill.style.width = percent + "%";
      progressBar.setAttribute("aria-valuenow", String(Math.round(percent)));
      var events = payload.events || [];
      for (var i = 0; i < events.length; i++) {
        state.jobEvents.push(events[i]);
        var item = document.createElement("li");
        item.textContent = "[" + Math.round(events[i].percent) + "%] " + events[i].message;
        jobLogOutput.appendChild(item);
      }
      state.lastEventSequence = payload.last_event_sequence || state.lastEventSequence;
      jobLogOutput.scrollTop = jobLogOutput.scrollHeight;
    }

    /* -- results -- */

    function selectView(view) {
      state.view = view;
      var tabs = document.querySelectorAll(".tab");
      for (var i = 0; i < tabs.length; i++) {
        var active = tabs[i].getAttribute("data-view") === view;
        tabs[i].classList.toggle("active", active);
        tabs[i].setAttribute("aria-selected", active ? "true" : "false");
      }
      if (state.result && state.result.images[view]) {
        resultImage.src = "data:image/png;base64," + state.result.images[view];
        resultImage.alt = humanizeKey(view.replace("_png_b64", "")) + " view of the segmentation result";
        viewCaption.textContent = VIEW_CAPTIONS[view] || "";
      }
    }

    var tabButtons = document.querySelectorAll(".tab");
    for (var t = 0; t < tabButtons.length; t++) {
      (function (tab) {
        tab.addEventListener("click", function () { selectView(tab.getAttribute("data-view")); });
      })(tabButtons[t]);
    }

    function renderFnPanel(payload) {
      var panel = $("fn-panel");
      var fn = payload.fn || {};
      if (!fn.available) {
        panel.setAttribute("hidden", "");
        return;
      }
      panel.removeAttribute("hidden");
      $("fn-length").textContent = fn.fn_length_weighted.toFixed(3);
      $("fn-count").textContent = fn.fn_count.toFixed(3);
      $("fn-length-detail").textContent =
        Math.round(fn.length_numerator_px) + " of " + Math.round(fn.length_denominator_px) + " px of hydride length";
      $("fn-count-detail").textContent =
        fn.count_numerator + " of " + fn.count_denominator + " hydrides";

      var context = "Counted at an angle threshold of " + fn.angle_threshold_deg + " deg.";
      if (fn.excluded_small_features > 0) {
        context += " " + fn.excluded_small_features +
          " feature(s) were excluded as smaller than the minimum feature size.";
      }
      if (fn.count_denominator === 0) {
        context = "No hydrides were measured, so Fn cannot be computed. " +
          "Check the segmentation on the Overlay tab, or lower the minimum feature size.";
      }
      $("fn-context").textContent = context;
    }

    function renderMetricGroups(payload) {
      metricGroups.innerHTML = "";
      var groups = payload.metric_groups || [];
      if (!groups.length) {
        var empty = document.createElement("p");
        empty.className = "hint";
        empty.textContent = "No measurements were produced for this run.";
        metricGroups.appendChild(empty);
        return;
      }
      for (var g = 0; g < groups.length; g++) {
        var group = groups[g];
        var details = document.createElement("details");
        details.className = "metric-group";
        if (group.key === "fn") { details.open = true; }
        var summary = document.createElement("summary");
        summary.textContent = group.title;
        details.appendChild(summary);

        var table = document.createElement("table");
        table.className = "metrics-table";
        var tbody = document.createElement("tbody");
        for (var m = 0; m < group.metrics.length; m++) {
          var entry = group.metrics[m];
          var row = document.createElement("tr");
          var nameCell = document.createElement("td");
          var valueCell = document.createElement("td");
          nameCell.textContent = humanizeKey(entry.key);
          valueCell.textContent = formatValue(entry.value);
          row.appendChild(nameCell);
          row.appendChild(valueCell);
          tbody.appendChild(row);
        }
        table.appendChild(tbody);
        details.appendChild(table);
        metricGroups.appendChild(details);
      }
    }

    function renderResult(payload) {
      state.result = payload;

      var tabs = document.querySelectorAll(".tab");
      var firstAvailable = null;
      for (var i = 0; i < tabs.length; i++) {
        var view = tabs[i].getAttribute("data-view");
        if (payload.images[view]) {
          tabs[i].removeAttribute("hidden");
          if (!firstAvailable) { firstAvailable = view; }
        } else {
          tabs[i].setAttribute("hidden", "");
        }
      }

      renderFnPanel(payload);
      renderMetricGroups(payload);

      var image = (payload.manifest && payload.manifest.image) || {};
      var parts = [
        payload.model_display_name,
        payload.source_name,
        image.width + " x " + image.height + " px",
        (payload.timing && payload.timing.total_seconds !== undefined)
          ? payload.timing.total_seconds + " s" : ""
      ];
      if (image.downscaled) {
        parts.push("downscaled from " + image.original_width + " x " + image.original_height + " px for speed");
      }
      runMeta.textContent = parts.filter(Boolean).join("  |  ");

      emptyState.setAttribute("hidden", "");
      resultArea.removeAttribute("hidden");
      $("download-mask").disabled = !payload.images.mask_png_b64;
      $("download-overlay").disabled = !payload.images.overlay_png_b64;

      selectView(payload.images[state.view] ? state.view : (firstAvailable || "overlay_png_b64"));
    }

    function download(view, suffix) {
      if (!state.result || !state.result.images[view]) { return; }
      var link = document.createElement("a");
      var base = (state.result.source_name || "image").replace(/\.[^.]+$/, "");
      link.href = "data:image/png;base64," + state.result.images[view];
      link.download = base + "_" + suffix + ".png";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    $("download-mask").addEventListener("click", function () { download("mask_png_b64", "mask"); });
    $("download-overlay").addEventListener("click", function () { download("overlay_png_b64", "overlay"); });

    $("download-metrics").addEventListener("click", function () {
      if (!state.result) { return; }
      var report = {
        source: state.result.source_name,
        model: state.result.model_display_name,
        model_id: state.result.model_id,
        fn: state.result.fn,
        metrics: state.result.metrics,
        quantification: state.result.manifest && state.result.manifest.quantification,
        image: state.result.manifest && state.result.manifest.image,
        privacy: state.result.privacy || (state.result.manifest && state.result.manifest.privacy),
        timing: state.result.timing,
        processing_log: state.jobEvents
      };
      var base = (state.result.source_name || "image").replace(/\.[^.]+$/, "");
      var link = document.createElement("a");
      link.href = "data:application/json;charset=utf-8," +
        encodeURIComponent(JSON.stringify(report, null, 2));
      link.download = base + "_measurements.json";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });

    /* -- run -- */

    function run() {
      var model = currentModel();
      if (!model || state.running) { return; }

      if (state.file && state.file.size > data.maxUploadMb * 1024 * 1024) {
        showError("That image is larger than the " + data.maxUploadMb + " MB limit for this server.");
        return;
      }

      clearError();
      state.running = true;
      updateRunButton();

      var form = new FormData();
      form.append("model_id", model.model_id);
      if (state.libraryId) {
        form.append("library_id", state.libraryId);
      } else if (state.sampleId) {
        form.append("sample_id", state.sampleId);
      } else if (state.file) {
        form.append("image", state.file, state.file.name);
      }
      if (model.is_conventional) {
        for (var i = 0; i < data.controls.length; i++) {
          var control = data.controls[i];
          var input = $("ctl-" + control.key);
          if (input) { form.append(control.key, input.value); }
        }
      }
      // Quantification acts on the mask, so it applies to every model.
      var quantControls = data.quantificationControls || [];
      for (var q = 0; q < quantControls.length; q++) {
        var qInput = $("ctl-" + quantControls[q].key);
        if (qInput) { form.append(quantControls[q].key, qInput.value); }
      }
      form.append("include_fn_classification", $("chk-fn-classification").checked ? "true" : "false");

      var started = Date.now();
      resetProgress();
      fetch("api/jobs", { method: "POST", body: form })
        .then(function (res) {
          return res.json().then(function (payload) { return { status: res.status, payload: payload }; });
        })
        .then(function (result) {
          if (!result.payload || result.payload.ok !== true) {
            var detail = (result.payload && result.payload.error && result.payload.error.detail) ||
              ("The server returned status " + result.status + ".");
            showError(detail);
            throw new Error("__handled__");
          }
          return pollJob(result.payload.status_url);
        })
        .catch(function (error) {
          if (error.message !== "__handled__") {
            showError("The request could not be completed: " + error.message +
              ". Check that the server is still running, then try again.");
          }
        })
        .then(function () {
          state.running = false;
          updateRunButton();
          var seconds = ((Date.now() - started) / 1000).toFixed(1);
          if (!errorBox.hasAttribute("hidden")) { runStatus.textContent = ""; }
          else if (state.result) { runStatus.textContent = "Completed in " + seconds + " s"; }
        });
    }

    function pollJob(statusUrl) {
      return new Promise(function (resolve, reject) {
        function check() {
          fetch(statusUrl + "?after=" + state.lastEventSequence, {
            headers: { "Accept": "application/json" }
          })
            .then(function (res) {
              return res.json().then(function (payload) {
                if (!res.ok) {
                  throw new Error((payload.error && payload.error.detail) || "Job status failed.");
                }
                return payload;
              });
            })
            .then(function (payload) {
              updateProgress(payload);
              if (payload.state === "completed") {
                renderResult(payload.result);
                resolve(payload.result);
              } else if (payload.state === "failed") {
                reject(new Error((payload.error && payload.error.detail) || payload.message));
              } else {
                window.setTimeout(check, 500);
              }
            })
            .catch(reject);
        }
        check();
      });
    }

    runBtn.addEventListener("click", run);

    populateModels(data.models, data.defaultModelId);
    updateRunButton();

    pollStatus(function (payload) {
      if (payload && payload.models) {
        data.models = payload.models;
        var selected = modelSelect.value;
        populateModels(payload.models, selected || data.defaultModelId);
      }
    });
  }

  window.MicroSeg = { initWorkspace: initWorkspace };

  document.addEventListener("DOMContentLoaded", function () {
    bindHelpToggles();
    if (!$("bootstrap-data")) { pollStatus(null); }
  });
})();
