"""
Quick test: check whether the connected Arducam camera (e.g. Arducam B0498 8.3MP
at 3840x2160 or B0477 20MP at 5472x3648) can deliver its full high resolution.

Run this standalone, separately from your measurement pipeline.

Usage:
    python test_resolution.py

Controls:
    SPACE = capture a still frame and save it
    ESC   = quit
"""

import cv2
import os
import time
import argparse

# Default working resolution
CAMERA_INDEX = 1
DEFAULT_WIDTH = 3840
DEFAULT_HEIGHT = 2160
DEFAULT_FPS = 10

OUTPUT_DIR = "test_capture"


def try_open_camera(index):
    # On Windows, cv2.CAP_DSHOW (DirectShow) is strongly recommended for USB UVC cameras
    print(f"Opening camera index {index} with CAP_DSHOW...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("CAP_DSHOW failed. Trying CAP_MSMF...")
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {index}.")
        raise SystemExit(1)

    return cap



def report_resolution(cap, label):
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    print(f"[{label}] Resolution: {w:.0f}x{h:.0f}  FPS: {fps:.1f}  FOURCC: {fourcc_str}")
    return w, h


def main():
    parser = argparse.ArgumentParser(description="Test camera resolution and capture.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help=f"Camera index (default: {CAMERA_INDEX})")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Target frame width (default: {DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Target frame height (default: {DEFAULT_HEIGHT})")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"Target FPS (default: {DEFAULT_FPS})")
    args = parser.parse_args()

    target_w = args.width
    target_h = args.height
    target_fps = args.fps

    cap = try_open_camera(args.camera)

    print("\n--- BEFORE setting anything ---")
    report_resolution(cap, "default")

    print(f"\n--- Requesting {target_w}x{target_h} @ {target_fps}fps ---")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
    if target_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, target_fps)

    time.sleep(0.5)
    w, h = report_resolution(cap, "after MJPG request")

    if (w, h) != (target_w, target_h):
        print("\nMJPG didn't take target resolution. Trying YUY2 instead...")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        time.sleep(0.5)
        w, h = report_resolution(cap, "after YUY2 request")

    # Discard initial warm-up frames to allow sensor auto-exposure to settle
    print("\nWarming up sensor...")
    frame = None
    for _ in range(8):
        ret, frame = cap.read()

    if frame is None or not ret:
        print("ERROR: Failed to read frame from camera.")
        cap.release()
        return

    actual_h, actual_w = frame.shape[:2]
    print(f"Captured frame shape: {actual_w}x{actual_h}")
    print(f"Pixel values: min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")

    if frame.max() == 0:
        print("\n" + "!" * 60)
        print("WARNING: Captured frame is pitch black (all 0s)!")
        print("This usually happens at high resolutions (1080p/4K) when:")
        print("  1. The camera is plugged into a USB 2.0 port or hub")
        print("  2. The USB cable is limited to USB 2.0 speeds")
        print("  Try switching back to 1280x720: python test_resolution.py")
        print("!" * 60 + "\n")
    else:
        print(f"\nSUCCESS: Live video feed active ({actual_w}x{actual_h}).")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "test_high_res_frame.jpg")
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"Saved test frame to: {out_path}")

    print("\nPress SPACE in preview window to grab another test frame, or ESC to quit.")

    frame_num = 2
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Lost camera feed.")
            break

        # Downscale for on-screen preview
        preview_width = 960
        preview_height = int(preview_width * frame.shape[0] / frame.shape[1])
        preview = cv2.resize(frame, (preview_width, preview_height))

        color = (0, 255, 0) if frame.max() > 0 else (0, 0, 255)
        status = f"{actual_w}x{actual_h}" if frame.max() > 0 else f"{actual_w}x{actual_h} [BLACK FRAME]"
        cv2.putText(preview, f"{status}  SPACE=capture  ESC=quit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Camera Test", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            out_path = os.path.join(OUTPUT_DIR, f"test_frame_{frame_num:02d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"Saved: {out_path}")
            frame_num += 1
        elif key in (10, 13, 27):  # ENTER or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
