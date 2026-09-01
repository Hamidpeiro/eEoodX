import cv2
import os
import config

# Folder for checkerboard photos
PHOTOS_DIR = os.path.join(config.CALIB_DIR, "checkerboard_photos")
os.makedirs(PHOTOS_DIR, exist_ok=True)

# Open Arducam B0477
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open Arducam B0477")
    exit()

print("Camera opened.")
print("Click on the camera window first.")
print("SPACE = capture photo")
print("ENTER = exit")

photo_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Arducam B0477", frame)

    # IMPORTANT: waitKey must be called after imshow
    key = cv2.waitKey(1)

    # SPACE
    if key == 32:
        filename = os.path.join(
            PHOTOS_DIR,
            f"checkerboard_{photo_count + 1:02d}.jpg"
        )

        success = cv2.imwrite(
            filename,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )

        if success:
            photo_count += 1
            print(f"Captured photo {photo_count}: {filename}")
        else:
            print("ERROR: Could not save image")

    # ENTER
    elif key in (10, 13):
        print("Enter pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

print(f"Finished. Captured {photo_count} photos.")