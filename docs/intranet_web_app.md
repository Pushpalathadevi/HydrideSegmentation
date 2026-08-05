# Intranet Web App Deployment

This guide covers running the segmentation tool as a browser application that colleagues open from their own machines over the office intranet. It is written for an air-gapped network: nothing in the install or the running application reaches the internet.

One host runs the server. Everyone else just opens a URL. No one else needs Python, a checkpoint, or an install of any kind.

## What Users Get

- Upload a micrograph, or run a bundled example image without needing their own data.
- Preview a selected upload in the drop area before running; bundled examples are presented as
  clickable thumbnail cards and also populate the same preview.
- Choose between the trained model and the conventional pipeline.
- **Radial hydride fraction (Fn)** reported as the headline result, both length-weighted and count-based, with a user-controlled angle threshold.
- Tune conventional and quantification parameters, each with in-app help behind a `?` button.
- View overlay, mask, Fn classification, orientation map, and size/angle distributions.
- Download the mask and overlay as PNG, and all measurements as JSON.
- A dedicated in-app Help page covering method choice, Fn interpretation, parameter meaning, result interpretation, and troubleshooting.

Uploaded images, masks, and reports are held in process memory only and are never written to disk or
forwarded anywhere. Background reports expire automatically, so users should download the artifacts
they need before leaving the page.

The workspace validates the filename extension and 5 MB limit immediately. The server then verifies
that the decoded format matches the extension, rejects multi-frame content and unsafe decoded
dimensions, and only then queues the in-memory job. A staged progress bar and timestamped live log
make preprocessing, inference, postprocessing, analysis, and rendering visible.

## Fn Quantification

Fn is the share of hydrides oriented radially rather than circumferentially, and it is the number most users are after. It sits in a dedicated panel above the measurements, reported two ways:

| Value | Meaning |
| --- | --- |
| **Length-weighted Fn** | Radial hydride length divided by total hydride length. The value to report in most cases; far less sensitive to how many tiny features the segmentation found |
| **Count-based Fn** | Number of radial hydrides divided by hydrides measured. Simple, but easily distorted by noise specks |

Both are shown with their numerator and denominator, so a user can see the value came from, say, "77 of 582 hydrides" rather than an unexplained number.

Fn applies to **both** routes. It is computed from the segmented mask, so the conventional and trained models both produce it.

Two settings change it, and both are recorded in the run manifest and the downloadable JSON:

| Setting | Default | Effect |
| --- | --- | --- |
| `fn_angle_threshold_deg` | `45` | Angle from horizontal at which a hydride counts as radial |
| `min_feature_pixels` | `1` | Features smaller than this are excluded from Fn and the distributions |

Ticking **Show which hydrides were counted** adds two QA views: an annotated image outlining counted features in green and uncounted in red, and the orientation histogram with the threshold drawn on it so users can see whether Fn is sensitive to exactly where the threshold sits. These are opt-in because they cost roughly an extra 0.5 s.

The in-app Help page explains all of this at `/help#fn`, including the guidance to always report the threshold and minimum feature size alongside the value.

Measurements are grouped into Fn, coverage, orientation, and feature size, with the Fn group open by default.

## Requirements

| Item | Requirement |
| --- | --- |
| Host OS | Windows 10/11 or Ubuntu (any systemd Linux) |
| Python | 3.10 or newer |
| CPU | Any x86-64. No GPU needed; CPU is the default and supported path |
| RAM | 4 GB minimum, 8 GB comfortable |
| Network | Intranet only. The host needs no internet access at run time |
| Client machines | A browser. Nothing installed |

## Install On An Air-Gapped Host

### Step 1: Build a wheelhouse on a machine that has internet

On any machine with the same OS and Python version as the target host:

```bash
pip download -r requirements-web.txt -d wheelhouse
```

Copy the repository and the `wheelhouse/` folder to the air-gapped host by USB or your approved transfer route.

### Step 2: Install offline on the host

Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install --no-index --find-links ./wheelhouse -r requirements-web.txt
.venv/bin/pip install --no-index --no-deps -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\pip install --no-index --find-links .\wheelhouse -r requirements-web.txt
.venv\Scripts\pip install --no-index --no-deps -e .
```

`--no-index` guarantees pip never tries to reach the network.

### Step 3: Install a trained checkpoint

Checkpoint binaries are not tracked in git, so a fresh copy of the repository has no `.pt` file. Until one is installed the conventional method works and the trained models are shown as unavailable.

```bash
microseg-cli install-model --checkpoint path/to/best_checkpoint.pth --model-id site_unet_v1 --nickname site_unet_v1_optical
```

Details: [`gui_model_integration_guide.md`](gui_model_integration_guide.md).

### Step 4: Start the server

```bash
python scripts/run_web_server.py
```

The command prints the URLs to share:

```text
  Segmentation web app is starting.
    On this machine:      http://localhost:5005/
    On the intranet:      http://ws-lab-07:5005/
    On the intranet:      http://10.24.6.31:5005/
```

Send colleagues one of the intranet links.

### Step 5: Open the host firewall

Nothing else works until inbound TCP on the port is allowed.

Windows, in an elevated PowerShell:

```powershell
New-NetFirewallRule -DisplayName "MicroSeg Web" -Direction Inbound -Protocol TCP -LocalPort 5005 -Action Allow
```

Ubuntu:

```bash
sudo ufw allow 5005/tcp
```

## Run It As A Service

A service keeps the app running across reboots and sign-outs.

### Ubuntu

```bash
sudo cp deploy/microseg-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now microseg-web
```

Edit `User`, `WorkingDirectory`, and the virtual-environment path in the unit first. Check it with `systemctl status microseg-web` and follow logs with `journalctl -u microseg-web -f`.

### Windows

Use [`deploy/start_web_server.bat`](../deploy/start_web_server.bat), either by double-clicking it or through Task Scheduler:

1. Create a task, trigger `At startup`.
2. Action: start `deploy\start_web_server.bat`.
3. Select **Run whether user is logged on or not** so the server survives sign-out.

## Configuration

Defaults live in [`configs/app/web_server.default.yml`](../configs/app/web_server.default.yml), which documents every option inline. Point at your own file with `--config` or `MICROSEG_WEB_CONFIG`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `server.host` | `0.0.0.0` | `0.0.0.0` publishes to the intranet; `127.0.0.1` keeps it local |
| `server.port` | `5005` | Listening port |
| `server.threads` | `4` | Waitress worker threads; keep near the core count |
| `limits.max_upload_mb` | `5` | Exact image-byte ceiling, enforced before decoding |
| `limits.max_long_side_px` | `2048` | Larger images are downscaled for speed; `0` disables |
| `limits.max_image_pixels` | `40000000` | Decoded-pixel safety ceiling |
| `limits.max_concurrent_jobs` | `2` | Simultaneous segmentation jobs; extra requests queue |
| `limits.max_retained_jobs` | `32` | Bound on queued and completed in-memory records |
| `limits.job_retention_seconds` | `1800` | Time before terminal in-memory reports expire |
| `models.preload_on_startup` | `true` | Load checkpoints at startup so the first request is fast; when disabled, the status indicator reports that models will load on first use |
| `models.default_model_id` | `auto` | `auto` picks the first ready trained model |
| `analysis.include_analysis` | `true` | Include orientation and distribution figures |
| `demo.sample_images` | two images | Example images offered in the browser |

Any value can be overridden with an environment variable named `MICROSEG_WEB_<SECTION>__<KEY>`, plus short aliases for the common ones:

```bash
MICROSEG_WEB_PORT=8080 python scripts/run_web_server.py
MICROSEG_WEB_LIMITS__MAX_UPLOAD_MB=50 python scripts/run_web_server.py
```

Command-line flags take precedence over both: `--host`, `--port`, `--threads`, `--preload/--no-preload`, `--dev`.

## Performance

By default, trained checkpoints are warmed into a shared in-process cache at startup on a background thread. `/health` answers immediately while that happens, and the header status indicator turns green when it finishes. Every later request reuses the loaded model. With `--no-preload`, the indicator instead reports **Ready - models load on first use** and the selected model is loaded on demand.

Measured on a 1024 x 768 optical micrograph, CPU only:

| Route | Time per request |
| --- | --- |
| Conventional, with analysis figures and Fn | ~0.7 s |
| Trained UNet, with analysis figures and Fn | ~1.3 s |
| Either route, plus the Fn classification views | add ~0.5 s |
| Either route, analysis disabled | ~0.3-0.6 s |

If your host is slower or your images are larger, the effective levers are `limits.max_long_side_px` (downscaling has the largest effect on wall time), `analysis.include_analysis`, and `server.threads`.

`limits.max_concurrent_jobs` bounds how many segmentations run at once. Beyond that requests wait, and if they wait past `server.request_timeout_seconds` the user is told the server is busy rather than being left hanging.

## API

The browser UI is built on a small JSON API you can also call from scripts.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Workspace page |
| `GET` | `/help` | In-app help page |
| `GET` | `/health` | Liveness probe; never blocks on model loading |
| `GET` | `/api/status` | Model readiness, active jobs, configured limits |
| `GET` | `/api/models` | Selectable models with availability |
| `GET` | `/api/samples` | Example images offered |
| `GET` | `/api/samples/<id>` | One example image |
| `POST` | `/api/segment` | Run a segmentation |
| `POST` | `/api/jobs` | Validate and submit an asynchronous in-memory job |
| `GET` | `/api/jobs/<job_id>` | Poll new progress events and retrieve the terminal result |
| `POST` | `/api/warm` | Load a model into memory on demand |

Example:

```bash
curl -F "model_id=hydride_conventional" -F "image=@micrograph.png" http://ws-lab-07:5005/api/segment
```

`/api/segment` accepts `model_id`, either `image` (file upload) or `sample_id`, the conventional parameter fields, the quantification fields (`fn_angle_threshold_deg`, `min_feature_pixels`, `orientation_bins`, `size_bins`), and the flags `include_analysis` and `include_fn_classification`.

The browser uses `/api/jobs`; `/api/segment` remains available for compatible programmatic clients.
Pass `after=<last_event_sequence>` while polling to receive only new log events.

It returns `ok`, `fn` (the Fn summary), `metrics`, `metric_groups`, `images` (base64 PNG strings), `manifest` (including the `quantification` settings used), and `timing` split into inference and analysis. Errors return `ok: false` with an `error.code` and a plain-language `error.detail`.

Fn straight from the command line:

```bash
curl -s -F "model_id=hydride_conventional" -F "image=@micrograph.png" -F "fn_angle_threshold_deg=45" http://ws-lab-07:5005/api/segment | python -c "import json,sys; print(json.load(sys.stdin)['fn'])"
```

## Security Notes

This app is designed for a trusted intranet and has no authentication of its own.

- Bind to `127.0.0.1` and put a reverse proxy in front if you need access control.
- Uploads are capped, extension-checked, and content-checked by decoding before use.
- Nothing is persisted: no upload directory, no result database.
- The server makes no outbound network connections, and the pages load no external scripts, fonts, or styles.

## Troubleshooting

**Colleagues cannot reach the URL.** Almost always the host firewall. Confirm the server works locally first with `curl http://localhost:5005/health`, then open the port.

**The trained model is unavailable.** No checkpoint is installed. Run `microseg-cli models --details` on the host to see why, then install one.

**Requests are slow.** Check the status indicator is green. If it is, lower `limits.max_long_side_px` or set `analysis.include_analysis: false`.

**`waitress is not installed`.** The launcher fell back to the development server. Install waitress from your wheelhouse; the development server is single-threaded and not meant for shared use.

**The port is already in use.** Another process holds it. Pick a different port with `--port`.

## Related Docs

- [`phase35_intranet_web_app.md`](phase35_intranet_web_app.md) closeout record
- [`gui_model_integration_guide.md`](gui_model_integration_guide.md) installing a checkpoint
- [`gui_user_guide.md`](gui_user_guide.md) the desktop application
- [`conventional_segmentation_pipeline.md`](conventional_segmentation_pipeline.md) what the conventional parameters do scientifically
- `deploy/README.md` service files
