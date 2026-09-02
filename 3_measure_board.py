"""
Step 3: Capture and measure timbers one by one.

Pipeline:
    camera
    -> SPACE to capture timber
    -> save original image
    -> undistort
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
    -> save mask
    -> save debug overlay
    -> write JSON
    -> wait for next timber

Usage:
    python 3_measure_board.py

Controls:
    SPACE = capture and measure current timber
    ENTER = exit

All captured images, debug images and JSON files are saved to:
    config.CAPTURE_DIR
"""

import cv2
import numpy as np
import json
import os
import config


# ============================================================
# CAMERA UNDISTORTION
# ============================================================

def undistort(img):
    """Remove lens distortion using the saved camera calibration."""

    calib = np.load(
        config.CAMERA_CALIB_FILE
    )

    return cv2.undistort(
        img,
        calib["camera_matrix"],
        calib["dist_coeffs"]
    )


# ============================================================
# TIMBER SEGMENTATION
# ============================================================

def segment_board(img, workspace_px):
    """Detect one timber inside the ArUco workspace."""

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
    # HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    bg_mask = cv2.inRange(
        hsv,
        config.BACKGROUND_HSV_LOWER,
        config.BACKGROUND_HSV_UPPER
    )

    board_mask = cv2.bitwise_not(
        bg_mask
    )

    # --------------------------------------------------------
    # Workspace only
    # --------------------------------------------------------

    board_mask = cv2.bitwise_and(
        board_mask,
        workspace_mask
    )

    # --------------------------------------------------------
    # Remove ArUco markers
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
    # Morphological cleanup
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
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        board_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, board_mask

    # --------------------------------------------------------
    # Filter small contours
    # --------------------------------------------------------

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area >= config.MIN_BOARD_CONTOUR_AREA_PX:

            valid_contours.append(
                contour
            )

    if not valid_contours:
        return None, board_mask

    # --------------------------------------------------------
    # One timber at a time:
    # select largest contour
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
    """Detect four ArUco markers and calculate pixel -> mm
    homography using the outer corner of each marker.
    """

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

    gray = cv2.cvtColor(
        img_undist,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, _ = detector.detectMarkers(
        gray
    )

    if ids is None:

        print(
            "  warning: no ArUco markers detected"
        )

        return None, None, None

    ids = ids.flatten()

    # OpenCV ArUco corner order:
    #
    # [0] -------- [1]
    #  |            |
    #  |            |
    # [3] -------- [2]

    outer_corner_index = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }

    detected = {}

    for marker_id, marker_corners in zip(
        ids,
        corners
    ):

        marker_id = int(
            marker_id
        )

        if marker_id not in outer_corner_index:
            continue

        corner_index = (
            outer_corner_index[
                marker_id
            ]
        )

        detected[marker_id] = (
            marker_corners[0][corner_index]
        )

    # --------------------------------------------------------
    # Check all markers
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
    # Pixel/world correspondences
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
    # Homography
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
    # Reprojection error
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
    # Print marker positions
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
        pass

    return {
        "L": round(mean_lab[0], 2),
        "a": round(mean_lab[1], 2),
        "b": round(mean_lab[2], 2)
    }


# ============================================================
# MEASURE ONE TIMBER
# ============================================================

def measure_timber(
    img,
    image_path,
    timber_number
):
    """Measure one captured timber."""

    print("\n" + "=" * 60)
    print(
        f"MEASURING TIMBER {timber_number:02d}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Undistort
    # --------------------------------------------------------

    img_undist = undistort(
        img
    )

    # --------------------------------------------------------
    # Live homography
    # --------------------------------------------------------

    H, errors_mm, detected = (
        compute_live_homography(
            img_undist
        )
    )

    if H is None:

        print(
            "\nERROR: Could not detect all four "
            "ArUco markers."
        )

        return False

    # --------------------------------------------------------
    # Homography quality
    # --------------------------------------------------------

    max_err = errors_mm.max()

    print(
        f"\nLive homography OK."
    )

    print(
        f"Maximum marker reprojection error: "
        f"{max_err:.2f} mm"
    )

    if max_err > 2.0:

        print(
            f"WARNING: marker error is "
            f"{max_err:.2f} mm."
        )

    # --------------------------------------------------------
    # Workspace
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
    # Detect timber
    # --------------------------------------------------------

    contour, mask = segment_board(
        img_undist,
        workspace_px
    )

    base_name = (
        f"timber_{timber_number:02d}"
    )

    # --------------------------------------------------------
    # ALWAYS save mask
    # --------------------------------------------------------

    mask_path = os.path.join(
        config.CAPTURE_DIR,
        f"{base_name}_mask.png"
    )

    cv2.imwrite(
        mask_path,
        mask
    )

    # --------------------------------------------------------
    # Check timber
    # --------------------------------------------------------

    if contour is None:

        print(
            "\nERROR: No timber detected."
        )

        print(
            f"Mask saved to:"
        )

        print(
            mask_path
        )

        return False

    # ========================================================
    # MIN AREA RECTANGLE
    # ========================================================

    rect = cv2.minAreaRect(
        contour
    )

    box_px = cv2.boxPoints(
        rect
    )

    # --------------------------------------------------------
    # Sub-pixel refinement
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
    # Pixel -> mm
    # --------------------------------------------------------

    box_mm = pixel_to_mm(
        H,
        box_px_refined
    )

    # --------------------------------------------------------
    # Side lengths
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

    length_mm = (
        side_lengths[0] +
        side_lengths[2]
    ) / 2

    width_mm = (
        side_lengths[1] +
        side_lengths[3]
    ) / 2

    if width_mm > length_mm:

        length_mm, width_mm = (
            width_mm,
            length_mm
        )

    # --------------------------------------------------------
    # Color
    # --------------------------------------------------------

    color = sample_color_lab(
        img_undist,
        contour
    )

    # ========================================================
    # JSON
    # ========================================================

    result = {

        "timber_id": timber_number,

        "source_image": os.path.abspath(
            image_path
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
        config.CAPTURE_DIR,
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

    # ========================================================
    # DEBUG OVERLAY
    # ========================================================

    overlay = img_undist.copy()

    # Workspace
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

    # ArUco reference points
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

    # Timber rectangle
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

    # Labels
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

    cv2.putText(
        overlay,
        f"TIMBER {timber_number:02d}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    overlay_path = os.path.join(
        config.CAPTURE_DIR,
        f"{base_name}_overlay.png"
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\n" + "-" * 60)
    print(
        f"TIMBER {timber_number:02d} MEASUREMENT"
    )
    print("-" * 60)

    print(
        f"Length: {length_mm:.2f} mm"
    )

    print(
        f"Width:  {width_mm:.2f} mm"
    )

    print(
        f"Color LAB: "
        f"L={color['L']:.2f}, "
        f"a={color['a']:.2f}, "
        f"b={color['b']:.2f}"
    )

    print(
        f"\nMeasurement saved:"
    )

    print(
        out_path
    )

    print(
        f"Overlay saved:"
    )

    print(
        overlay_path
    )

    return True


# ============================================================
# FIND NEXT TIMBER NUMBER
# ============================================================

def get_next_timber_number():
    """Find the next available timber number."""

    os.makedirs(
        config.CAPTURE_DIR,
        exist_ok=True
    )

    number = 1

    while True:

        json_path = os.path.join(
            config.CAPTURE_DIR,
            f"timber_{number:02d}_measurement.json"
        )

        if not os.path.exists(
            json_path
        ):
            return number

        number += 1


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

def main():

    # --------------------------------------------------------
    # Create capture directory
    # --------------------------------------------------------

    os.makedirs(
        config.CAPTURE_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check calibration
    # --------------------------------------------------------

    if not os.path.exists(
        config.CAMERA_CALIB_FILE
    ):

        print(
            f"ERROR: Missing "
            f"{config.CAMERA_CALIB_FILE}"
        )

        print(
            "Run 1_calibrate_camera.py first."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    CAMERA_INDEX = 1

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open Arducam B0477."
        )

        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("TIMBER MEASUREMENT CAMERA")
    print("=" * 60)

    print(
        "\nCamera opened successfully."
    )

    print(
        "Place ONE timber inside the workspace."
    )

    print(
        "SPACE = capture and measure"
    )

    print(
        "ENTER = exit"
    )

    print(
        f"\nOutput folder:"
    )

    print(
        config.CAPTURE_DIR
    )

    # --------------------------------------------------------
    # Timber number
    # --------------------------------------------------------

    timber_number = (
        get_next_timber_number()
    )

    # --------------------------------------------------------
    # Camera loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Failed to read camera frame."
            )

            break

        # ----------------------------------------------------
        # Live preview
        # ----------------------------------------------------

        display = frame.copy()

        cv2.putText(
            display,
            "SPACE = Capture / Measure",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "ENTER = Exit",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Next: TIMBER {timber_number:02d}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Arducam B0477 - Timber Measurement",
            display
        )

        # IMPORTANT:
        # waitKey must be called after imshow.
        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # SPACE = CAPTURE
        # ----------------------------------------------------

        if key == 32:

            print(
                f"\nCapturing timber "
                f"{timber_number:02d}..."
            )

            # ------------------------------------------------
            # Save original image
            # ------------------------------------------------

            image_path = os.path.join(
                config.CAPTURE_DIR,
                f"timber_{timber_number:02d}.jpg"
            )

            success = cv2.imwrite(
                image_path,
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    100
                ]
            )

            if not success:

                print(
                    "ERROR: Could not save "
                    "captured image."
                )

                continue

            print(
                f"Captured:"
            )

            print(
                image_path
            )

            # ------------------------------------------------
            # Measure
            # ------------------------------------------------

            success = measure_timber(
                frame,
                image_path,
                timber_number
            )

            if success:

                print(
                    "\n" + "=" * 60
                )

                print(
                    f"TIMBER {timber_number:02d} COMPLETE"
                )

                print(
                    "=" * 60
                )

                timber_number += 1

                print(
                    "\nPlace the next timber."
                )

                print(
                    f"Press SPACE to measure "
                    f"timber {timber_number:02d}."
                )

            else:

                print(
                    "\nMeasurement failed."
                )

                print(
                    "Timber number was NOT increased."
                )

                print(
                    "Correct the timber position "
                    "and press SPACE to try again."
                )

        # ----------------------------------------------------
        # ENTER = EXIT
        # ----------------------------------------------------

        elif key in (10, 13):

            print(
                "\nEnter pressed. Exiting..."
            )

            break

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    print(
        "\nCamera released."
    )

    print(
        "Measurement session finished."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
