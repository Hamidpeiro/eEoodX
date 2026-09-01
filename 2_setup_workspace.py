"""
Step 2: Compute the pixel -> real-world-millimetre mapping for your table.

Run this ONCE after the rig (camera height/angle) is physically fixed in
place, and re-run it any time the camera or the markers move.

What it does:
    1. Takes one photo of the empty table with the 4 corner ArUco markers
       visible (from 0_generate_markers.py, stuck at measured positions).
    2. Undistorts it using the lens calibration from step 1.
    3. Finds each marker's center in pixel coordinates.
    4. Computes a homography from pixel space -> the real-world mm
       coordinates you entered in config.MARKER_WORLD_POSITIONS_MM.
    5. Saves that homography so measure_board.py can convert any pixel
       point on the table into real-world millimetres.

Usage:
    python 2_setup_workspace.py path/to/workspace_photo.jpg
"""

import cv2
import numpy as np
import sys
import config


def undistort(img):
    calib = np.load(config.CAMERA_CALIB_FILE)
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]
    return cv2.undistort(img, camera_matrix, dist_coeffs), camera_matrix, dist_coeffs


def main():
    if len(sys.argv) != 2:
        print("Usage: python 2_setup_workspace.py path/to/workspace_photo.jpg")
        raise SystemExit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        raise SystemExit(1)

    img_undist, _, _ = undistort(img)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        print("No ArUco markers detected. Check lighting/focus and that "
              "the markers from 0_generate_markers.py are visible and flat.")
        raise SystemExit(1)

    ids = ids.flatten()
    detected = {int(i): c[0].mean(axis=0) for i, c in zip(ids, corners)}
    print(f"Detected marker IDs: {sorted(detected.keys())}")

    expected_ids = set(config.MARKER_WORLD_POSITIONS_MM.keys())
    missing = expected_ids - set(detected.keys())
    if missing:
        print(f"Missing expected markers: {missing}. "
              "Check they're stuck down, in frame, and not occluded.")
        raise SystemExit(1)

    pixel_pts = []
    world_pts = []
    for marker_id, world_xy in config.MARKER_WORLD_POSITIONS_MM.items():
        pixel_pts.append(detected[marker_id])
        world_pts.append(world_xy)

    pixel_pts = np.array(pixel_pts, dtype=np.float32)
    world_pts = np.array(world_pts, dtype=np.float32)

    # Homography: pixel (undistorted image) -> real-world mm on the table plane
    H, mask = cv2.findHomography(pixel_pts, world_pts, method=0)

    # Sanity check: reproject the marker pixels through H and compare to
    # the world positions you typed in, so typos or a moved marker show up.
    reprojected = cv2.perspectiveTransform(pixel_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors_mm = np.linalg.norm(reprojected - world_pts, axis=1)
    print("\nPer-marker reprojection error (should be near 0mm):")
    for marker_id, err in zip(config.MARKER_WORLD_POSITIONS_MM.keys(), errors_mm):
        print(f"  marker {marker_id}: {err:.2f} mm")
    if errors_mm.max() > 5.0:
        print("\nWarning: >5mm error on a marker - double check the measured "
              "position in config.py and that the marker is flat/not warped.")

    np.savez(config.HOMOGRAPHY_FILE, H=H)
    print(f"\nSaved homography to {config.HOMOGRAPHY_FILE}")


if __name__ == "__main__":
    main()
