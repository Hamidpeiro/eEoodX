import cv2

for index in range(6):

    print("\n" + "=" * 50)
    print(f"Testing camera index {index}")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("NOT AVAILABLE")
        continue

    print("Camera opened")

    # Try to request 1920x1080 first
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()

    if ret and frame is not None:
        h, w = frame.shape[:2]

        print(f"Actual frame: {w} x {h}")

        filename = f"camera_{index}.jpg"
        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")
    else:
        print("Could not capture frame")

    cap.release()