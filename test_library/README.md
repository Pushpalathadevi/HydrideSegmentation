# Test image library

Micrographs placed in this folder are offered to users of the web application as
a browsable gallery, so colleagues can try the tool without having an image of
their own on hand.

## How to use it

Copy image files directly into this folder on the server:

```
test_library/
    optical_q2_hydrided.png
    optical_q4_hydrided.png
    sem_reoriented_hydrides.png
```

The server scans the folder on every request, so images added or removed here
appear in the browser without restarting the web application.

## Rules

- **Flat folder only.** Subfolders are ignored.
- **Accepted types:** `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`.
  Everything else in the folder is skipped.
- **Filenames become labels.** The file stem is shown under each thumbnail with
  underscores and hyphens turned into spaces, so `q_2_hydrided.png` reads as
  "q 2 hydrided". Name files for the people who will read them.
- **Large images are fine.** The browser upload limit does not apply here
  because these files never travel over the network on their way in. They are
  still downscaled to the configured `limits.max_long_side_px` before inference.

## Why this folder is not in git

Like `pre_trained_weights/` and the frozen checkpoint binaries, the library is
deployment data rather than source. Only this README and `.gitkeep` are tracked;
the images themselves are ignored by `.gitignore`. Supply them as part of
deploying to a given host.

## Configuration

The folder location and the cap on how many images are listed are set under
`demo` in `configs/app/web_server.default.yml`:

```yaml
demo:
  library_dir: "test_library"
  library_max_images: 200
```

If this folder is missing, empty, or unreadable, the web application quietly
falls back to the two example images listed under `demo.sample_images`, so the
server still works on a host where no library was supplied.
