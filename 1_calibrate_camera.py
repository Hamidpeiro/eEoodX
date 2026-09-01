"""
Step 1: Intrinsic camera calibration (lens distortion correction).

Do this ONCE per (camera + lens + focus + aperture) setting. If you touch
the focus ring or aperture ring after this, re-run it - both physically
change the lens's optical characteristics.

How to shoot the calibration photos:
    1. Print a checkerboard (config.CHECKERBOARD_INNER_CORNERS = 8x5 inner
       corners by default). Any "9x6 squares" checkerboard PDF online works -
       just measure your printed square size with calipers and update
       config.CHECKERBOARD_SQUARE_SIZE_MM to match exactly.
    2. Tape it FLAT to something rigid (a board, not paper that can bow).
    3. With the camera mounted in its final rig position, take 15-25 photos
       of the checkerboard at that same distance/focus, tilted and shifted
       to different parts of the frame each time (corners, edges, center,
       some angle). Save them into calibration_data/checkerboard_photos/.
    4. Run this script.

Usage:
    python 1_calibrate_camera.py
"""

import cv2
import numpy as np
import glob
import os
import config

PHOTOS_DIR = os.path.join(config.CALIB_DIR, "checkerboard_photos")
os.makedirs(PHOTOS_DIR, exist_ok=True)

cols, rows = config.CHECKERBOARD_INNER_CORNERS
square_mm = config.CHECKERBOARD_SQUARE_SIZE_MM

# 3D points of checkerboard corners in the board's own coordinate frame
# (Z=0 plane), scaled to real millimetres.
objp = np.zeros((cols * rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

objpoints = []  # 3D points, one set per accepted photo
imgpoints = []  # matching 2D detected corner points

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

image_paths = sorted(glob.glob(os.path.join(PHOTOS_DIR, "*.jpg")) +
                      glob.glob(os.path.join(PHOTOS_DIR, "*.png")))

if not image_paths:
    print(f"No photos found in {PHOTOS_DIR}")
    print("Add 15-25 checkerboard photos there first (see the docstring "
          "at the top of this file), then re-run.")
    raise SystemExit(1)

image_size = None
accepted = 0

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"  skip (unreadable): {path}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    if not found:
        print(f"  skip (no checkerboard found): {os.path.basename(path)}")
        continue

    corners_refined = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )
    objpoints.append(objp)
    imgpoints.append(corners_refined)
    accepted += 1
    print(f"  ok: {os.path.basename(path)}")

print(f"\nAccepted {accepted}/{len(image_paths)} photos.")
if accepted < 10:
    print("Fewer than 10 good photos - calibration will be unreliable. "
          "Add more photos covering different areas of the frame.")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

print(f"\nRMS reprojection error: {ret:.4f} px "
      "(lower is better; under ~0.5px is good for this use case)")
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.ravel())

np.savez(
    config.CAMERA_CALIB_FILE,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    image_size=image_size,
    rms_error=ret,
)
print(f"\nSaved to {config.CAMERA_CALIB_FILE}")
