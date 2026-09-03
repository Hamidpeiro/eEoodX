# Timber Scanner — starter pipeline

Static setup: one board placed under a fixed overhead camera (Arducam B0477,
IMX283, 16mm C-mount), photographed, measured, and exported to Rhino.

This is a complete pixel → millimetre measurement and segmentation pipeline running on your
actual rig.

## Install

```bash
pip install -r requirements.txt
```

## Physical setup, in order

### 1. Mount the camera
At ~4.3m overhead (to fit a 3.5m field of view), lens locked at a fixed
focus and aperture. Level it — perpendicularity to the table matters more
than anything else for accuracy over that distance. **Once focus/aperture
are set, don't touch them again** — every script after this assumes a fixed
lens state.

### 2. Build the table
A flat surface, ~3.5m long, covered in a uniform, light or dark mat / surface. Diffuse lighting along its full
length (not a single overhead point light — you'll get a hot spot in the
middle and dark corners).

### 3. Generate and place the corner markers
```bash
python 0_generate_markers.py
```
Print `calibration_data/marker_0.png` .. `marker_3.png` at exactly 100mm
per side (check with a ruler — printer scaling isn't always exact). Stick
them down at the 4 corners of your working area. **Measure their real-world
positions with a tape measure** and update `config.MARKER_WORLD_POSITIONS_MM`
to match — this is the ground truth everything else is scaled against.

### 4. Calibrate the lens
Print a checkerboard (any "9x6 squares" PDF works, giving 8x5 inner
corners — default in `config.py` or 13x8 inner corners for larger boards). Tape it to something rigid.

Take 15–25 photos of it at the camera's actual working distance/focus,
covering different parts of the frame (corners, edges, tilted). Save into
`calibration_data/checkerboard_photos/`. Then:

```bash
python 1_calibrate_camera.py
```

Check the printed RMS reprojection error — aim for well under 1px. If it's
high, you probably don't have enough photos or enough variation between
them.

**Re-run this any time you touch focus or aperture.**

### 5. Sanity-check the pixel-to-mm mapping
With the empty table and all 4 markers visible, take one photo, then:

```bash
python 2_setup_workspace.py path/to/workspace_photo.jpg
```

Check the per-marker reprojection error it prints — should be near 0mm.
Anything over a few mm means a mismeasured marker position or a marker
that's not flat.

### 6. Timber Segmentation Pipeline
The timber segmentation inside `3_measure_board.py` uses an adaptive multi-channel pipeline:
- **Workspace-Constrained Otsu Thresholding**: Calculates dynamic Otsu threshold on the Lightness channel (`L`) in LAB space specifically inside the ArUco workspace polygon, separating dark timber from bright table background.
- **LAB & HSV Warmth Filtering**: Uses chromaticity (`b* > 132`, `a* > 129`, `S > 35`) to strictly isolate timber wood grain from neutral gray shadows and white table surface.
- **ArUco Marker Masking**: Automatically masks out circular regions around the 4 ArUco markers before thresholding to prevent tape edges, marker borders, and reflections from interfering with timber bounding boxes.

### 7. Interactive Live Measurement & Capture
Run the live measurement camera loop:

```bash
python 3_measure_board.py
```

**Controls:**
- **`SPACE`**: Capture current frame, run homography & timber measurement, increment timber number, and save outputs.
- **`ENTER`**: Exit the camera loop.

All output files are automatically saved to `sample_output/captures/`:
- `timber_XX.jpg`: Original captured image frame
- `timber_XX_mask.png`: Binary segmentation mask of detected timber
- `timber_XX_overlay.png`: Visual overlay displaying detected ArUco workspace (red) and measured timber outline (green)
- `timber_XX_measurement.json`: Real-world measurements including length (mm), width (mm), corner coordinates (mm), and average LAB color

### 8. Export to Rhino
```bash
python 4_export_rhino.py sample_output/captures/timber_01_measurement.json sample_output/captures/timber_02_measurement.json
```

Produces `sample_output/boards.3dm` — closed polyline outlines plus text
labels, in real-world mm. Pass multiple JSON files at once to combine several
boards into one file. Open directly in Rhino, or feed into Rhino.Inside /
Grasshopper for a live pipeline.

## What's stubbed out, deliberately

- **Defect detection/labeling** (knots, cracks, species, grading) — this
  needs a model trained on your own timber images (YOLOv8-seg is a
  reasonable starting point). The `defects` field in the JSON output and
  the defect-marker loop in the Rhino export are there waiting for it —
  map your model's pixel boxes through the same `pixel_to_mm()` function in
  `3_measure_board.py` so defects land in the same coordinate frame as the
  board outline.
- **Color chart correction** — `config.USE_COLOR_CHART_CORRECTION` and the
  stub in `sample_color_lab()` are there for when you add an in-frame
  ColorChecker. Without it, LAB readings will drift with your lighting
  setup rather than being a truly calibrated color measurement.

## Known limitations of this approach

- Single 2D image only measures length/width/color — no thickness, warp,
  bow, or twist. That needs a second sensor (structured light, stereo, or a
  laser line) if you need it.
- At ~4.3m working distance you're at roughly 0.6mm/pixel resolution.
  Board-level length/width accuracy after sub-pixel edge fitting should
  land around 0.1–0.2mm, but small defects (a few pixels across) won't be
  measured that precisely — fine for detecting them, not for precise
  defect sizing. If you need that, plan a second closer-range pass over
  flagged regions.
