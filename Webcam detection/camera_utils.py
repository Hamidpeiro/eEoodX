"""
Camera utilities for timber scanning rig:
- Hardware parameter controls via duvc_ctl and OpenCV (focus, zoom, exposure, light, white balance)
- Digital Sensor Zoom & Crop support
- VideoCapture setup
- Lens distortion removal (supporting OpenCV Fisheye and Standard Pinhole models with map caching)
"""

import os
import cv2
import numpy as np
import config

try:
    import duvc_ctl as duvc
    HAS_DUVC = True
except ImportError:
    duvc = None
    HAS_DUVC = False


# ===========================================================================
# Digital Zoom & Center Crop Helper
# ===========================================================================

def apply_digital_zoom(img, zoom_factor=1.0):
    """
    Apply high-quality center digital crop and resize.
    zoom_factor: float multiplier (1.0 = no zoom, 1.5 = 1.5x zoom, 2.0 = 2.0x zoom, etc.)
    """
    if zoom_factor is None or zoom_factor <= 1.001:
        return img

    h, w = img.shape[:2]
    crop_w = max(10, int(w / zoom_factor))
    crop_h = max(10, int(h / zoom_factor))

    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ===========================================================================
# Hardware Camera Control (duvc_ctl + OpenCV)
# ===========================================================================

def _clamp_property_value(controller, prop_name, value):
    """Safely clamp property value within device's supported min and max."""
    try:
        r = controller.get_property_range(prop_name)
        if isinstance(r, dict):
            min_val = r.get("min", None)
            max_val = r.get("max", None)
            if min_val is not None and value < min_val:
                return min_val
            if max_val is not None and value > max_val:
                return max_val
    except Exception:
        pass
    return value


def apply_camera_settings(cap=None, device_index=None):
    """
    Apply focus, zoom, exposure, brightness, contrast, and white balance settings
    from config.py to the camera using duvc_ctl and OpenCV properties.
    """
    idx = device_index if device_index is not None else getattr(config, "CAMERA_INDEX", 1)
    controller = None

    if HAS_DUVC:
        try:
            controller = duvc.CameraController(device_index=idx)
            print(f"[Camera Control] Connected to camera index {idx} ({getattr(controller, 'device_name', 'Webcam')}).")

            # Focus mode & value
            focus_mode = getattr(config, "CAMERA_FOCUS_MODE", "manual").lower()
            if focus_mode in ("manual", "auto"):
                try:
                    controller.focus_mode = focus_mode
                    print(f"  -> Focus Mode: {focus_mode}")
                except Exception as e:
                    print(f"  -> Could not set focus_mode: {e}")

            if focus_mode == "manual":
                focus_val = getattr(config, "CAMERA_FOCUS_VALUE", None)
                if focus_val is not None:
                    try:
                        clamped = _clamp_property_value(controller, "focus", int(focus_val))
                        controller.focus = clamped
                        print(f"  -> Focus: {controller.focus}")
                    except Exception as e:
                        print(f"  -> Could not set focus value: {e}")

            # Exposure mode & value
            exp_mode = getattr(config, "CAMERA_EXPOSURE_MODE", "auto").lower()
            if exp_mode == "auto":
                try:
                    if hasattr(controller, "exposure_mode"):
                        controller.exposure_mode = "auto"
                    elif hasattr(controller, "_set_property_auto"):
                        controller._set_property_auto("exposure")
                    print("  -> Exposure Mode: auto")
                except Exception as e:
                    print(f"  -> Could not set auto exposure: {e}")
            elif exp_mode == "manual":
                exp_val = getattr(config, "CAMERA_EXPOSURE_VALUE", None)
                if exp_val is not None:
                    try:
                        clamped = _clamp_property_value(controller, "exposure", int(exp_val))
                        controller.exposure = clamped
                        print(f"  -> Exposure (Manual): {controller.exposure}")
                    except Exception as e:
                        print(f"  -> Could not set exposure value: {e}")

            # Brightness, Contrast, Saturation, Sharpness, Gain, Backlight
            for prop_name, conf_attr in [
                ("brightness", "CAMERA_BRIGHTNESS"),
                ("contrast", "CAMERA_CONTRAST"),
                ("saturation", "CAMERA_SATURATION"),
                ("sharpness", "CAMERA_SHARPNESS"),
                ("gain", "CAMERA_GAIN"),
                ("backlight_compensation", "CAMERA_BACKLIGHT_COMPENSATION"),
            ]:
                val = getattr(config, conf_attr, None)
                if val is not None and hasattr(controller, prop_name):
                    try:
                        clamped = _clamp_property_value(controller, prop_name, int(val))
                        setattr(controller, prop_name, clamped)
                        print(f"  -> {prop_name.title()}: {getattr(controller, prop_name)}")
                    except Exception as e:
                        print(f"  -> Could not set {prop_name}: {e}")

            # White balance
            wb_mode = getattr(config, "CAMERA_WHITE_BALANCE_MODE", "auto").lower()
            if wb_mode == "auto":
                try:
                    if hasattr(controller, "white_balance_mode"):
                        controller.white_balance_mode = "auto"
                    elif hasattr(controller, "_set_property_auto"):
                        controller._set_property_auto("white_balance")
                    print("  -> White Balance: auto")
                except Exception as e:
                    print(f"  -> Could not set white balance mode: {e}")
            elif wb_mode == "manual":
                wb_val = getattr(config, "CAMERA_WHITE_BALANCE_TEMPERATURE", None)
                if wb_val is not None and hasattr(controller, "white_balance"):
                    try:
                        clamped = _clamp_property_value(controller, "white_balance", int(wb_val))
                        controller.white_balance = clamped
                        print(f"  -> White Balance (Manual): {controller.white_balance}K")
                    except Exception as e:
                        print(f"  -> Could not set white balance temperature: {e}")

        except Exception as e:
            print(f"[Camera Control] Note: duvc_ctl controller init: {e}")
            if controller is not None:
                try:
                    controller.close()
                except Exception:
                    pass
                controller = None

    # OpenCV DirectShow property fallbacks if cap is available
    if cap is not None and cap.isOpened():
        try:
            focus_mode = getattr(config, "CAMERA_FOCUS_MODE", "manual").lower()
            if focus_mode == "auto":
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            else:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                focus_val = getattr(config, "CAMERA_FOCUS_VALUE", None)
                if focus_val is not None:
                    cap.set(cv2.CAP_PROP_FOCUS, float(focus_val))

            for prop, conf_attr in [
                (cv2.CAP_PROP_BRIGHTNESS, "CAMERA_BRIGHTNESS"),
                (cv2.CAP_PROP_CONTRAST, "CAMERA_CONTRAST"),
                (cv2.CAP_PROP_SATURATION, "CAMERA_SATURATION"),
                (cv2.CAP_PROP_SHARPNESS, "CAMERA_SHARPNESS"),
                (cv2.CAP_PROP_GAIN, "CAMERA_GAIN"),
            ]:
                val = getattr(config, conf_attr, None)
                if val is not None:
                    cap.set(prop, float(val))

        except Exception as e:
            print(f"[Camera Control] OpenCV property fallback warning: {e}")

    return controller


def open_configured_camera(device_index=None, width=None, height=None, fps=None):
    """
    Open cv2.VideoCapture with DirectShow, configure stream dimensions and codec,
    and apply hardware parameters from config.py.
    
    Returns:
        cap: cv2.VideoCapture instance
        controller: duvc_ctl.CameraController instance (or None)
    """
    idx = device_index if device_index is not None else getattr(config, "CAMERA_INDEX", 1)
    target_w = width if width is not None else getattr(config, "IMAGE_WIDTH", 3840)
    target_h = height if height is not None else getattr(config, "IMAGE_HEIGHT", 2160)
    target_fps = fps if fps is not None else getattr(config, "CAMERA_FPS", 30)
    fourcc_str = getattr(config, "CAMERA_FOURCC", "MJPG")

    # Connect controller first for manual focus/exposure stability if supported
    controller = apply_camera_settings(device_index=idx)

    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened() and idx != 0:
        print(f"[Camera] Index {idx} failed to open, trying index 0...")
        idx = 0
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

    if not cap.isOpened():
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass
        raise RuntimeError(f"Could not open camera at index {idx} with cv2.CAP_DSHOW.")

    # Configure stream format
    if fourcc_str and len(fourcc_str) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    # Discard warm-up frames
    for _ in range(5):
        cap.read()

    # Re-apply properties to live capture device
    apply_camera_settings(cap=cap, device_index=idx)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[Camera] Stream ready: {actual_w:.0f}x{actual_h:.0f} @ {target_fps}fps (Index: {idx})")

    return cap, controller


# ===========================================================================
# Lens Distortion Removal (Fisheye & Standard Models with Map Caching)
# ===========================================================================

class Undistorter:
    """
    High-performance undistorter that supports both cv2.fisheye and standard
    OpenCV calibration models with precomputed remap grids.
    """
    def __init__(self, calib_file=None):
        self.calib_file = calib_file or config.CAMERA_CALIB_FILE
        self.is_loaded = False
        self.is_fisheye = False
        self.K = None
        self.D = None
        self.calib_size = None
        self.cached_maps = {}  # (w, h, balance, fov_scale) -> (map1, map2, new_K)
        self.load()

    def load(self):
        if not os.path.exists(self.calib_file):
            self.is_loaded = False
            return False

        try:
            data = np.load(self.calib_file)
            self.K = np.array(data["camera_matrix"], dtype=np.float64)
            self.D = np.array(data["dist_coeffs"], dtype=np.float64)
            self.is_fisheye = bool(data.get("is_fisheye", getattr(config, "CALIBRATION_MODEL", "fisheye") == "fisheye"))
            if "image_size" in data:
                size = data["image_size"]
                self.calib_size = (int(size[0]), int(size[1]))
            else:
                self.calib_size = None
            self.is_loaded = True
            self.cached_maps.clear()
            return True
        except Exception as e:
            print(f"[Undistorter] Failed to load calibration file {self.calib_file}: {e}")
            self.is_loaded = False
            return False

    def _get_maps(self, img_w, img_h, balance=None, fov_scale=None):
        bal = balance if balance is not None else getattr(config, "FISHEYE_BALANCE", 0.0)
        fov = fov_scale if fov_scale is not None else getattr(config, "FISHEYE_FOV_SCALE", 1.0)
        cache_key = (img_w, img_h, float(bal), float(fov), self.is_fisheye)

        if cache_key in self.cached_maps:
            return self.cached_maps[cache_key]

        K_scaled = self.K.copy()
        if self.calib_size is not None:
            calib_w, calib_h = self.calib_size
            if (img_w, img_h) != (calib_w, calib_h):
                sx = img_w / float(calib_w)
                sy = img_h / float(calib_h)
                K_scaled[0, 0] *= sx
                K_scaled[1, 1] *= sy
                K_scaled[0, 2] *= sx
                K_scaled[1, 2] *= sy

        image_size = (img_w, img_h)

        if self.is_fisheye:
            # OpenCV Fisheye Rectification
            # Ensure D has 4 parameters
            D_vec = self.D.reshape(4, 1) if self.D.size == 4 else self.D
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K_scaled,
                D_vec,
                image_size,
                np.eye(3),
                balance=bal,
                new_size=image_size,
                fov_scale=fov
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K_scaled,
                D_vec,
                np.eye(3),
                new_K,
                image_size,
                cv2.CV_16SC2
            )
        else:
            # Standard pinhole model
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                K_scaled,
                self.D,
                image_size,
                alpha=bal,
                newImgSize=image_size
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                K_scaled,
                self.D,
                None,
                new_K,
                image_size,
                cv2.CV_16SC2
            )

        self.cached_maps[cache_key] = (map1, map2, new_K)
        return map1, map2, new_K

    def undistort(self, img, balance=None, fov_scale=None):
        """
        Undistort an image using fast cv2.remap.
        """
        if not self.is_loaded:
            if not self.load():
                return img

        img_h, img_w = img.shape[:2]
        map1, map2, _ = self._get_maps(img_w, img_h, balance=balance, fov_scale=fov_scale)
        undistorted = cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        return undistorted


# Global singleton instance
_GLOBAL_UNDISTORTER = None

def get_undistorter(calib_file=None):
    global _GLOBAL_UNDISTORTER
    if _GLOBAL_UNDISTORTER is None or calib_file is not None:
        _GLOBAL_UNDISTORTER = Undistorter(calib_file)
    return _GLOBAL_UNDISTORTER

def undistort(img, balance=None, fov_scale=None):
    """Convenience function to undistort an image."""
    return get_undistorter().undistort(img, balance=balance, fov_scale=fov_scale)
