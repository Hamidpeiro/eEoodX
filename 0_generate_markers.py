"""
Generate printable ArUco markers for the fixed table-corner positions.

Run once. Print the output PNGs at the exact size defined in config.py
(ARUCO_MARKER_SIZE_MM) - use a ruler on the printed page to confirm, printer
scaling is not always 100% accurate. Stick them down permanently at the
positions you'll later measure and enter into config.py.

Usage:
    python 0_generate_markers.py
"""

import cv2
import os
import config

os.makedirs(config.CALIB_DIR, exist_ok=True)

aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))

# Render at a decent pixel density so printing stays crisp. 10 px/mm is fine.
px_per_mm = 10
marker_px = int(config.ARUCO_MARKER_SIZE_MM * px_per_mm)

for marker_id in config.MARKER_WORLD_POSITIONS_MM:
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
    out_path = os.path.join(config.CALIB_DIR, f"marker_{marker_id}.png")
    cv2.imwrite(out_path, img)
    print(f"Wrote {out_path} ({config.ARUCO_MARKER_SIZE_MM}mm square at {px_per_mm}px/mm)")

print("\nPrint each marker so the black square measures exactly "
      f"{config.ARUCO_MARKER_SIZE_MM}mm per side (verify with a ruler).")
print("Stick them at the table positions you will measure into "
      "config.MARKER_WORLD_POSITIONS_MM.")
