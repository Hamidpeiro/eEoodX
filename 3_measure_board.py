"""
Step 3: Measure a single board from one photo.

Pipeline:
    undistort
    -> detect 4 ArUco markers
    -> calculate live homography
    -> define workspace from ArUco markers
    -> threshold timber inside workspace
    -> remove ArUco markers
    -> find timber contour
    -> minAreaRect
    -> sub-pixel corner refinement
    -> map timber corners to real-world mm
    -> compute length/width
    -> sample color
    -> write JSON result.

Usage:
    python 3_measure_board.py path/to/board_photo.jpg

Debug:
    python 3_measure_board.py path/to/board_photo.jpg --debug
"""

import cv2
import numpy as np
import json
import sys
import os
import argparse
import config


# ============================================================
# CAMERA UNDISTORTION
# ============================================================

def undistort(img):
    """Remove lens distortion using the saved camera calibration."""

    calib = np.load(config.CAMERA_CALIB_FILE)

    return cv2.undistort(
        img,
        calib["camera_matrix"],
        calib["dist_coeffs"]
    )


# ============================================================
# TIMBER SEGMENTATION
# ============================================================

def segment_board(img, workspace_px):
    """Detect timber only inside the workspace defined by
    the four ArUco marker centers.

    workspace_px order:

        0 ---------------- 1
        |                  |
        |      TIMBER      |
        |                  |
        3 ---------------- 2

    Returns:
        contour, board_mask
    """

    # --------------------------------------------------------
    # 1. Create workspace mask
    # --------------------------------------------------------

    workspace_mask = np.zeros(
        img.shape[:2],
        dtype=np.uint8
    )

    workspace_pts = np.array(
        workspace_px,
        dtype=np.int32
    )

    cv2.fillPoly(
        workspace_mask,
        [workspace_pts],
        255
    )

    # --------------------------------------------------------
    # 2. Convert image to HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # 3. Detect background
    # --------------------------------------------------------

    bg_mask = cv2.inRange(
        hsv,
        config.BACKGROUND_HSV_LOWER,
        config.BACKGROUND_HSV_UPPER
    )

    # Everything that is NOT background becomes
    # a possible timber region.
    board_mask = cv2.bitwise_not(bg_mask)

    # --------------------------------------------------------
    # 4. Keep ONLY the ArUco workspace
    # --------------------------------------------------------

    board_mask = cv2.bitwise_and(
        board_mask,
        workspace_mask
    )

    # --------------------------------------------------------
    # 5. Remove ArUco markers
    # --------------------------------------------------------
    #
    # We remove a circular region around each marker center.
    #
    # The marker itself must not become the timber contour.
    #
    # 70 px is an initial value and can be adjusted later
    # depending on the camera resolution and marker size.
    # --------------------------------------------------------

    marker_radius = 70

    for point in workspace_px:

        x = int(round(point[0]))
        y = int(round(point[1]))

        cv2.circle(
            board_mask,
            (x, y),
            marker_radius,
            0,
            -1
        )

    # --------------------------------------------------------
    # 6. Morphological cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (15, 15),
        dtype=np.uint8
    )

    board_mask = cv2.morphologyEx(
        board_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    board_mask = cv2.morphologyEx(
        board_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # 7. Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        board_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, board_mask

    # --------------------------------------------------------
    # 8. Remove very small contours
    # --------------------------------------------------------

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area >= config.MIN_BOARD_CONTOUR_AREA_PX:
            valid_contours.append(contour)

    if not valid_contours:
        return None, board_mask

    # --------------------------------------------------------
    # 9. Select largest valid contour
    # --------------------------------------------------------

    largest = max(
        valid_contours,
        key=cv2.contourArea
    )

    return largest, board_mask


# ============================================================
# SUB-PIXEL CORNER REFINEMENT
# ============================================================

def refine_corners_subpixel(img_gray, corners):
    """Refine rectangle corners to sub-pixel accuracy."""

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01
    )

    pts = corners.reshape(
        -1,
        1,
        2
    ).astype(np.float32)

    height, width = img_gray.shape[:2]

    # Keep points inside the image.
    pts[:, 0, 0] = np.clip(
        pts[:, 0, 0],
        0,
        width - 1
    )

    pts[:, 0, 1] = np.clip(
        pts[:, 0, 1],
        0,
        height - 1
    )

    refined = cv2.cornerSubPix(
        img_gray,
        pts,
        (5, 5),
        (-1, -1),
        criteria
    )

    return refined.reshape(
        -1,
        2
    )


# ============================================================
# PIXEL -> REAL WORLD MM
# ============================================================

def pixel_to_mm(H, pts_px):
    """Transform image pixel coordinates into world coordinates."""

    pts_px = np.array(
        pts_px,
        dtype=np.float32
    ).reshape(
        -1,
        1,
        2
    )

    pts_mm = cv2.perspectiveTransform(
        pts_px,
        H
    )

    return pts_mm.reshape(
        -1,
        2
    )


# ============================================================
# LIVE ARUCO HOMOGRAPHY
# ============================================================

def compute_live_homography(img_undist):
    """Detect the four ArUco markers and calculate the
    pixel -> millimetre homography.

    Marker arrangement:

        0 ---------------- 1
        |                  |
        |                  |
        |                  |
        3 ---------------- 2

    The world coordinates come from config.py.

    Returns:

        H
        errors_mm
        detected

    or:

        None, None, None
    """

    # --------------------------------------------------------
    # 1. Create ArUco detector
    # --------------------------------------------------------

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(
            cv2.aruco,
            config.ARUCO_DICT
        )
    )

    aruco_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        aruco_params
    )

    # --------------------------------------------------------
    # 2. Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        img_undist,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # 3. Detect markers
    # --------------------------------------------------------

    corners, ids, _ = detector.detectMarkers(
        gray
    )

    if ids is None:

        print(
            "  warning: no ArUco markers detected"
        )

        return None, None, None

    ids = ids.flatten()

    # --------------------------------------------------------
    # 4. Calculate center of every detected marker
    # --------------------------------------------------------

    detected = {}

    outer_corner_index = {
        0: 0,  # top-left marker → top-left corner
        1: 1,  # top-right marker → top-right corner
        2: 2,  # bottom-right marker → bottom-right corner
        3: 3,  # bottom-left marker → bottom-left corner
    }

    for marker_id, marker_corners in zip(
        ids,
        corners
    ):
        corner_index = outer_corner_index[int(marker_id)]

        detected[int(marker_id)] = (
            marker_corners[0][corner_index]
        )

    # --------------------------------------------------------
    # 5. Check required marker IDs
    # --------------------------------------------------------

    expected_ids = set(
        config.MARKER_WORLD_POSITIONS_MM.keys()
    )

    detected_ids = set(
        detected.keys()
    )

    if not expected_ids.issubset(
        detected_ids
    ):

        missing = (
            expected_ids -
            detected_ids
        )

        print(
            f"  warning: markers not all visible "
            f"(missing {missing})"
        )

        return None, None, None

    # --------------------------------------------------------
    # 6. Build pixel/world correspondences
    # --------------------------------------------------------

    pixel_pts = []
    world_pts = []

    for marker_id, world_xy in (
        config.MARKER_WORLD_POSITIONS_MM.items()
    ):

        pixel_pts.append(
            detected[marker_id]
        )

        world_pts.append(
            world_xy
        )

    pixel_pts = np.array(
        pixel_pts,
        dtype=np.float32
    )

    world_pts = np.array(
        world_pts,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 7. Calculate homography
    # --------------------------------------------------------

    H, _ = cv2.findHomography(
        pixel_pts,
        world_pts,
        method=0
    )

    if H is None:

        print(
            "  warning: could not calculate homography"
        )

        return None, None, None

    # --------------------------------------------------------
    # 8. Check marker reprojection
    # --------------------------------------------------------

    reprojected = cv2.perspectiveTransform(
        pixel_pts.reshape(
            -1,
            1,
            2
        ),
        H
    ).reshape(
        -1,
        2
    )

    errors_mm = np.linalg.norm(
        reprojected - world_pts,
        axis=1
    )

    # --------------------------------------------------------
    # 9. Print marker coordinates
    # --------------------------------------------------------

    print(
        "\nArUco marker positions in world coordinates:"
    )

    for marker_id, point in zip(
        config.MARKER_WORLD_POSITIONS_MM.keys(),
        reprojected
    ):

        print(
            f"  Marker {marker_id}: "
            f"X={point[0]:.2f} mm, "
            f"Y={point[1]:.2f} mm"
        )

    print(
        f"\n  Maximum marker reprojection error: "
        f"{errors_mm.max():.4f} mm"
    )

    return (
        H,
        errors_mm,
        detected
    )


# ============================================================
# COLOR SAMPLING
# ============================================================

def sample_color_lab(img_bgr, contour):
    """Calculate average LAB color from inside the timber."""

    mask = np.zeros(
        img_bgr.shape[:2],
        np.uint8
    )

    cv2.drawContours(
        mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Pull away from edges to avoid background contamination.
    mask = cv2.erode(
        mask,
        np.ones(
            (25, 25),
            np.uint8
        )
    )

    lab = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2LAB
    )

    mean_lab = cv2.mean(
        lab,
        mask=mask
    )[:3]

    if config.USE_COLOR_CHART_CORRECTION:

        # Placeholder for future color-chart correction.
        pass

    return {
        "L": round(mean_lab[0], 2),
        "a": round(mean_lab[1], 2),
        "b": round(mean_lab[2], 2)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_path"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="save mask and overlay images"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    img = cv2.imread(
        args.image_path
    )

    if img is None:

        print(
            f"Could not read image: "
            f"{args.image_path}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Check camera calibration
    # --------------------------------------------------------

    if not os.path.exists(
        config.CAMERA_CALIB_FILE
    ):

        print(
            f"Missing {config.CAMERA_CALIB_FILE} "
            "- run 1_calibrate_camera.py first."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Undistort image
    # --------------------------------------------------------

    img_undist = undistort(
        img
    )

    # --------------------------------------------------------
    # Calculate live homography
    # --------------------------------------------------------

    H, errors_mm, detected = (
        compute_live_homography(
            img_undist
        )
    )

    # --------------------------------------------------------
    # If live markers are not available,
    # use cached homography.
    # --------------------------------------------------------

    if H is not None:

        max_err = errors_mm.max()

        print(
            f"  live homography ok, "
            f"max marker reprojection error: "
            f"{max_err:.2f}mm"
        )

        if max_err > 2.0:

            print(
                f"  warning: {max_err:.2f}mm marker error "
                "- check whether the rig moved."
            )

    else:

        if not os.path.exists(
            config.HOMOGRAPHY_FILE
        ):

            print(
                "Could not detect all corner markers "
                "and no cached homography exists."
            )

            raise SystemExit(1)

        print(
            "  warning: using cached homography."
        )

        H = np.load(
            config.HOMOGRAPHY_FILE
        )["H"]

        # We cannot define the live workspace without
        # detected marker centers.
        print(
            "ERROR: timber detection requires all "
            "four ArUco markers to be visible."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Build workspace polygon
    # --------------------------------------------------------

    workspace_px = np.array(
        [
            detected[0],
            detected[1],
            detected[2],
            detected[3]
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Detect timber ONLY inside workspace
    # --------------------------------------------------------

    contour, mask = segment_board(
        img_undist,
        workspace_px
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        config.OUTPUT_DIR,
        exist_ok=True
    )

    base_name = os.path.splitext(
        os.path.basename(
            args.image_path
        )
    )[0]

    # --------------------------------------------------------
    # Save mask
    # --------------------------------------------------------

    if args.debug:

        cv2.imwrite(
            os.path.join(
                config.OUTPUT_DIR,
                f"{base_name}_mask.png"
            ),
            mask
        )

    # --------------------------------------------------------
    # Check timber detection
    # --------------------------------------------------------

    if contour is None:

        print(
            "No timber detected."
        )

        print(
            "Run with --debug and inspect the mask."
        )

        raise SystemExit(1)

    # ========================================================
    # TIMBER RECTANGLE
    # ========================================================

    rect = cv2.minAreaRect(
        contour
    )

    box_px = cv2.boxPoints(
        rect
    )

    # --------------------------------------------------------
    # Refine timber corners
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        img_undist,
        cv2.COLOR_BGR2GRAY
    )

    box_px_refined = (
        refine_corners_subpixel(
            gray,
            box_px
        )
    )

    # --------------------------------------------------------
    # Convert timber corners to mm
    # --------------------------------------------------------

    box_mm = pixel_to_mm(
        H,
        box_px_refined
    )

    # --------------------------------------------------------
    # Calculate timber side lengths
    # --------------------------------------------------------

    side_lengths = []

    for i in range(4):

        p1 = box_mm[i]

        p2 = box_mm[
            (i + 1) % 4
        ]

        side_lengths.append(
            float(
                np.linalg.norm(
                    p2 - p1
                )
            )
        )

    # Average opposite sides.
    length_mm = (
        side_lengths[0] +
        side_lengths[2]
    ) / 2

    width_mm = (
        side_lengths[1] +
        side_lengths[3]
    ) / 2

    # Ensure length is always the larger dimension.
    if width_mm > length_mm:

        length_mm, width_mm = (
            width_mm,
            length_mm
        )

    # --------------------------------------------------------
    # Sample timber color
    # --------------------------------------------------------

    color = sample_color_lab(
        img_undist,
        contour
    )

    # ========================================================
    # JSON RESULT
    # ========================================================

    result = {

        "source_image": os.path.abspath(
            args.image_path
        ),

        "length_mm": round(
            length_mm,
            2
        ),

        "width_mm": round(
            width_mm,
            2
        ),

        "corners_mm": [
            [
                round(float(x), 2),
                round(float(y), 2)
            ]
            for x, y in box_mm
        ],

        "color_lab": color,

        "defects": []
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    out_path = os.path.join(
        config.OUTPUT_DIR,
        f"{base_name}_measurement.json"
    )

    with open(
        out_path,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print(
        f"\nSaved to {out_path}"
    )

    # ========================================================
    # DEBUG OVERLAY
    # ========================================================

    if args.debug:

        overlay = img_undist.copy()

        # ----------------------------------------------------
        # Draw ArUco workspace boundary
        # RED = workspace
        # ----------------------------------------------------

        cv2.polylines(
            overlay,
            [
                np.int32(
                    workspace_px
                )
            ],
            isClosed=True,
            color=(0, 0, 255),
            thickness=6
        )

        # ----------------------------------------------------
        # Draw ArUco marker centers
        # BLUE = marker centers
        # ----------------------------------------------------

        for marker_id in [0, 1, 2, 3]:

            point = detected[
                marker_id
            ]

            x = int(
                round(point[0])
            )

            y = int(
                round(point[1])
            )

            cv2.circle(
                overlay,
                (x, y),
                10,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                overlay,
                str(marker_id),
                (x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

        # ----------------------------------------------------
        # Draw detected timber rectangle
        # GREEN = timber
        # ----------------------------------------------------

        cv2.polylines(
            overlay,
            [
                np.int32(
                    box_px_refined
                )
            ],
            isClosed=True,
            color=(0, 255, 0),
            thickness=6
        )

        # ----------------------------------------------------
        # Add labels
        # ----------------------------------------------------

        cv2.putText(
            overlay,
            "RED = ArUco workspace",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            overlay,
            "GREEN = detected timber",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Save overlay
        # ----------------------------------------------------

        overlay_path = os.path.join(
            config.OUTPUT_DIR,
            f"{base_name}_overlay.png"
        )

        cv2.imwrite(
            overlay_path,
            overlay
        )

        print(
            f"Debug overlay saved to {overlay_path}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()