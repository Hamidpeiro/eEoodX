"""
Central config for the timber scanning rig.
Fill in the CAPS variables for your physical setup. Everything else
reads from here so you only edit numbers in one place.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, "calibration_data")
CAMERA_CALIB_FILE = os.path.join(CALIB_DIR, "camera_intrinsics.npz")
HOMOGRAPHY_FILE = os.path.join(CALIB_DIR, "homography.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "sample_output")
# All captured timber images, debug images and measurement JSON files # are stored here. 
CAPTURE_DIR = os.path.join( OUTPUT_DIR, "captures" )

# ---------------------------------------------------------------------------
# Camera (Arducam B0477 / IMX283)
# ---------------------------------------------------------------------------
# Native still resolution. If you capture at a lower resolution for speed,
# change this and re-run calibration at that same resolution.
IMAGE_WIDTH = 5472
IMAGE_HEIGHT = 3648
PIXEL_SIZE_UM = 2.4  # sensor physical pixel pitch, from datasheet

# ---------------------------------------------------------------------------
# Checkerboard used for intrinsic (lens distortion) calibration
# ---------------------------------------------------------------------------
# INNER corners, not squares. A "14x9 squares" board has 13x8 inner corners.
CHECKERBOARD_INNER_CORNERS = (13, 8)
CHECKERBOARD_SQUARE_SIZE_MM = 20.0  # measure your printed squares with calipers

# ---------------------------------------------------------------------------
# Workspace / homography (ArUco markers at the 4 known table corners)
# ---------------------------------------------------------------------------
ARUCO_DICT = "DICT_4X4_50"
ARUCO_MARKER_SIZE_MM = 100.0  # physical printed size of each marker's black square

# Marker ID -> real-world (X, Y) position in millimetres on your table plane.
# Standard CAD / Rhino / Grasshopper coordinate system:
# - Origin (0,0) at Bottom-Left (Marker 1)
# - +X axis along table length towards Bottom-Right (Marker 0)
# - +Y axis along table width towards Top-Left (Marker 2)
# - (+X, +Y) at Top-Right (Marker 3)
MARKER_WORLD_POSITIONS_MM = {
    1: (0.0, 0.0),        # bottom-left (Origin)
    0: (776.0, 0.0),      # bottom-right (+X)
    2: (0.0, 440.0),      # top-left (+Y)
    3: (776.0, 440.0),    # top-right (+X, +Y)
}

# ---------------------------------------------------------------------------
# Camera Setup & Parallax Compensation
# ---------------------------------------------------------------------------
# Height from camera lens to table surface in mm.
# Update this value if you adjust camera mounting height.
CAMERA_HEIGHT_MM = 1750.0

# Timber thickness in mm (set to 0.0 to measure directly on table plane without requiring thickness input)
TIMBER_THICKNESS_MM = 0.0

# ---------------------------------------------------------------------------
# Segmentation (board vs background)
# ---------------------------------------------------------------------------
# Simple approach: background is a known, roughly uniform color (e.g. a dark
# mat). We threshold in HSV. Tune these ranges to your actual mat color by
# running measure_board.py with --debug and looking at the mask.
BACKGROUND_HSV_LOWER = (0, 0, 0)
BACKGROUND_HSV_UPPER = (180, 15, 255)  # tune this: covers dark, low-saturation mat

MIN_BOARD_CONTOUR_AREA_PX = 5000  # ignore small noise blobs after thresholding

# ---------------------------------------------------------------------------
# Color reference chart (optional but recommended)
# ---------------------------------------------------------------------------
# If you place a color checker in frame, set this True and fill in the pixel
# ROI of one known-reference patch once you know where it sits in your shots.
USE_COLOR_CHART_CORRECTION = False
