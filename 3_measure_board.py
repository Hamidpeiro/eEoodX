"""
Step 3: Measure a single board from one photo.

Pipeline: undistort -> threshold against background -> find board contour
-> minAreaRect -> sub-pixel corner refinement -> map corners to real-world
mm via the homography from step 2 -> compute length/width -> sample color
-> write a JSON result.

Usage:
    python 3_measure_board.py path/to/board_photo.jpg
    python 3_measure_board.py path/to/board_photo.jpg --debug   # saves mask/overlay for tuning
"""

import cv2
import numpy as np
import json
import sys
import os
import argparse
import config


def undistort(img):
    calib = np.load(config.CAMERA_CALIB_FILE)
    return cv2.undistort(img, calib["camera_matrix"], calib["dist_coeffs"])


def segment_board(img):
    """Threshold out the board from a known-uniform background. Returns the
    largest contour above the area cutoff, or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg_mask = cv2.inRange(hsv, config.BACKGROUND_HSV_LOWER, config.BACKGROUND_HSV_UPPER)
    board_mask = cv2.bitwise_not(bg_mask)

    # Clean up small noise / fill small holes
    kernel = np.ones((15, 15), np.uint8)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, kernel)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, board_mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < config.MIN_BOARD_CONTOUR_AREA_PX:
        return None, board_mask

    return largest, board_mask


# def refine_corners_subpixel(img_gray, corners):
#     """cv2.cornerSubPix expects float32 points and a small search window."""
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
#     pts = corners.reshape(-1, 1, 2).astype(np.float32)
#     refined = cv2.cornerSubPix(img_gray, pts, (5, 5), (-1, -1), criteria)
#     return refined.reshape(-1, 2)
def refine_corners_subpixel(img_gray, corners):
    """Refine rectangle corners to sub-pixel accuracy.

    cornerSubPix requires all initial points to be inside the image,
    so we clamp the points before refinement.
    """
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01
    )

    pts = corners.reshape(-1, 1, 2).astype(np.float32)

    height, width = img_gray.shape[:2]

    # Keep points inside the image.
    pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, width - 1)
    pts[:, 0, 1] = np.clip(pts[:, 0, 1], 0, height - 1)

    refined = cv2.cornerSubPix(
        img_gray,
        pts,
        (5, 5),
        (-1, -1),
        criteria
    )

    return refined.reshape(-1, 2)

def pixel_to_mm(H, pts_px):
    pts_px = np.array(pts_px, dtype=np.float32).reshape(-1, 1, 2)
    pts_mm = cv2.perspectiveTransform(pts_px, H).reshape(-1, 2)
    return pts_mm


def compute_live_homography(img_undist):
    """Detect the 4 corner ArUco markers in THIS photo (same frame as the
    board) and compute a fresh homography from them, rather than trusting a
    cached one from setup day. This catches camera drift/bumps between
    shots - if a marker has moved, the reprojection error below will jump.

    Returns (H, reprojection_errors_mm) or (None, None) if markers weren't
    all found.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None

    ids = ids.flatten()
    detected = {int(i): c[0].mean(axis=0) for i, c in zip(ids, corners)}

    expected_ids = set(config.MARKER_WORLD_POSITIONS_MM.keys())
    if not expected_ids.issubset(detected.keys()):
        missing = expected_ids - set(detected.keys())
        print(f"  warning: markers not all visible in this shot (missing {missing})")
        return None, None

    pixel_pts, world_pts = [], []
    for marker_id, world_xy in config.MARKER_WORLD_POSITIONS_MM.items():
        pixel_pts.append(detected[marker_id])
        world_pts.append(world_xy)
    pixel_pts = np.array(pixel_pts, dtype=np.float32)
    world_pts = np.array(world_pts, dtype=np.float32)

    H, _ = cv2.findHomography(pixel_pts, world_pts, method=0)

    reprojected = cv2.perspectiveTransform(pixel_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors_mm = np.linalg.norm(reprojected - world_pts, axis=1)

    return H, errors_mm


def sample_color_lab(img_bgr, contour):
    """Average color from the interior of the board (eroded mask so we
    avoid edge/background bleed), returned in LAB."""
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mask = cv2.erode(mask, np.ones((25, 25), np.uint8))  # pull in from edges

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    mean_lab = cv2.mean(lab, mask=mask)[:3]  # (L, a, b) in OpenCV's 0-255 LAB scale

    if config.USE_COLOR_CHART_CORRECTION:
        # Plug your color-chart correction in here once you have a reference
        # patch ROI defined - e.g. a linear correction matrix fit against
        # known chart values. Left as a stub since it depends on your chart
        # placement.
        pass

    return {"L": round(mean_lab[0], 2), "a": round(mean_lab[1], 2), "b": round(mean_lab[2], 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--debug", action="store_true", help="save mask/overlay images for tuning")
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Could not read image: {args.image_path}")
        raise SystemExit(1)

    if not os.path.exists(config.CAMERA_CALIB_FILE):
        print(f"Missing {config.CAMERA_CALIB_FILE} - run 1_calibrate_camera.py first.")
        raise SystemExit(1)

    img_undist = undistort(img)

    # Prefer a homography computed live from THIS photo's markers - catches
    # any camera movement since the last shot. Falls back to the cached one
    # from 2_setup_workspace.py (with a loud warning) only if markers aren't
    # all visible right now, e.g. temporarily blocked by the board itself.
    H, errors_mm = compute_live_homography(img_undist)
    if H is not None:
        max_err = errors_mm.max()
        print(f"  live homography ok, max marker reprojection error: {max_err:.2f}mm")
        if max_err > 2.0:
            print(f"  warning: {max_err:.2f}mm marker error is high - check nothing "
                  "bumped the rig, or re-run 2_setup_workspace.py")
    else:
        if not os.path.exists(config.HOMOGRAPHY_FILE):
            print("Could not detect all corner markers in this photo, and no "
                  f"cached homography at {config.HOMOGRAPHY_FILE}. "
                  "Run 2_setup_workspace.py first, or make sure all 4 markers "
                  "are visible and unobstructed.")
            raise SystemExit(1)
        print("  warning: could not see all markers in this shot - falling back "
              "to the cached homography from 2_setup_workspace.py. Move the "
              "board so it doesn't block the markers if this keeps happening.")
        H = np.load(config.HOMOGRAPHY_FILE)["H"]

    contour, mask = segment_board(img_undist)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]

    if args.debug:
        cv2.imwrite(os.path.join(config.OUTPUT_DIR, f"{base_name}_mask.png"), mask)

    if contour is None:
        print("No board detected. Run with --debug and check the saved mask - "
              "tune config.BACKGROUND_HSV_LOWER/UPPER to your actual background color.")
        raise SystemExit(1)

    # Minimum-area rectangle around the board -> 4 corner points in pixels
    rect = cv2.minAreaRect(contour)
    box_px = cv2.boxPoints(rect)  # 4x2 float32

    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    box_px_refined = refine_corners_subpixel(gray, box_px)

    box_mm = pixel_to_mm(H, box_px_refined)

    # Side lengths of the rectangle in mm
    side_lengths = []
    for i in range(4):
        p1 = box_mm[i]
        p2 = box_mm[(i + 1) % 4]
        side_lengths.append(float(np.linalg.norm(p2 - p1)))
    # Opposite sides should roughly match; length = longer pair, width = shorter pair
    length_mm = (side_lengths[0] + side_lengths[2]) / 2
    width_mm = (side_lengths[1] + side_lengths[3]) / 2
    if width_mm > length_mm:
        length_mm, width_mm = width_mm, length_mm

    color = sample_color_lab(img_undist, contour)

    result = {
        "source_image": os.path.abspath(args.image_path),
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "corners_mm": [[round(float(x), 2), round(float(y), 2)] for x, y in box_mm],
        "color_lab": color,
        # Placeholder for the defect-detection model's output - wire in
        # your YOLO/segmentation results here, in the same mm coordinate
        # frame (map defect pixel boxes through pixel_to_mm() the same way).
        "defects": [],
    }

    out_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_measurement.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")

    if args.debug:
        overlay = img_undist.copy()
        cv2.drawContours(overlay, [np.int32(box_px_refined)], -1, (0, 255, 0), 8)
        cv2.imwrite(os.path.join(config.OUTPUT_DIR, f"{base_name}_overlay.png"), overlay)


if __name__ == "__main__":
    main()
