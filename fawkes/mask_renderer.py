import cv2
import json
import numpy as np
from enum import Enum
from pathlib import Path


class MaskType(Enum):
    FACE_MESH = "face_mesh"
    GUY_FAWKES = "guy_fawkes"
    MAGIC_MIRROR = "magic_mirror"
    TEDDY_BEAR = "teddy_bear"


class MaskRenderer:
    """Renders mask overlays onto detected faces using affine transformation."""

    # Key MediaPipe landmark indices for face alignment
    LANDMARK_INDICES = {
        'left_eye_outer': 263,
        'right_eye_outer': 33,
        'nose_tip': 1,
        'chin': 152,
    }

    def __init__(self):
        self.masks = {}
        self.mask_configs = {}
        self._load_masks()

    def _get_assets_path(self) -> Path:
        """Get path to assets directory, works for both dev and PyInstaller builds."""
        # Try relative to this module first (development)
        module_dir = Path(__file__).parent
        assets_path = module_dir / "assets" / "masks"

        if assets_path.exists():
            return assets_path

        # PyInstaller bundles files in _MEIPASS
        import sys
        if hasattr(sys, '_MEIPASS'):
            return Path(sys._MEIPASS) / "fawkes" / "assets" / "masks"

        return assets_path

    def _load_masks(self):
        """Load mask images and configurations from assets directory."""
        assets_path = self._get_assets_path()
        config_path = assets_path / "mask_config.json"

        if not config_path.exists():
            return

        with open(config_path, 'r') as f:
            self.mask_configs = json.load(f)

        for mask_name, config in self.mask_configs.items():
            image_path = assets_path / config['image']
            if image_path.exists():
                # Load with alpha channel (IMREAD_UNCHANGED)
                mask_img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if mask_img is not None:
                    self.masks[mask_name] = mask_img

    def get_face_landmarks(self, face_landmarks, frame_shape) -> dict:
        """Extract key landmark coordinates from MediaPipe face landmarks."""
        h, w = frame_shape[:2]
        landmarks = {}

        for name, idx in self.LANDMARK_INDICES.items():
            lm = face_landmarks.landmark[idx]
            landmarks[name] = np.array([lm.x * w, lm.y * h], dtype=np.float32)

        return landmarks

    def render_mask(self, mask_type: MaskType, face_landmarks, frame_shape) -> np.ndarray:
        """Render the specified mask onto a black background aligned to face landmarks.

        Args:
            mask_type: The type of mask to render
            face_landmarks: MediaPipe face landmarks object
            frame_shape: Shape of the output frame (height, width, channels)

        Returns:
            Frame with mask rendered on black background
        """
        mask_name = mask_type.value

        if mask_name not in self.masks or mask_name not in self.mask_configs:
            # Return black frame if mask not found
            return np.zeros(frame_shape, dtype=np.uint8)

        mask_img = self.masks[mask_name]
        config = self.mask_configs[mask_name]

        # Get face landmarks in frame coordinates
        face_pts = self.get_face_landmarks(face_landmarks, frame_shape)

        # Get mask anchor points from config
        anchors = config['anchors']

        # Source points from mask image (use 3 points for affine transform)
        src_pts = np.array([
            anchors['left_eye'],
            anchors['right_eye'],
            anchors['chin']
        ], dtype=np.float32)

        # Destination points from detected face
        dst_pts = np.array([
            face_pts['left_eye_outer'],
            face_pts['right_eye_outer'],
            face_pts['chin']
        ], dtype=np.float32)

        # Calculate affine transformation matrix
        transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)

        # Warp the mask to fit the face
        h, w = frame_shape[:2]
        warped_mask = cv2.warpAffine(
            mask_img,
            transform_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Create output on black background
        output = np.zeros((h, w, 3), dtype=np.uint8)

        # Alpha blend the warped mask onto the black background
        if warped_mask.shape[2] == 4:
            # Extract alpha channel and normalize to 0-1
            alpha = warped_mask[:, :, 3:4].astype(np.float32) / 255.0
            # Extract BGR channels
            mask_bgr = warped_mask[:, :, :3]
            # Blend: output = mask * alpha + background * (1 - alpha)
            output = (mask_bgr * alpha + output * (1 - alpha)).astype(np.uint8)
        else:
            # No alpha channel, just overlay non-black pixels
            mask_gray = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
            mask_binary = mask_gray > 0
            output[mask_binary] = warped_mask[mask_binary]

        return output

    @staticmethod
    def get_available_masks() -> list:
        """Return list of available mask types for UI dropdown."""
        return [
            (MaskType.FACE_MESH, "Face Mesh"),
            (MaskType.GUY_FAWKES, "Guy Fawkes"),
            (MaskType.MAGIC_MIRROR, "Magic Mirror"),
            (MaskType.TEDDY_BEAR, "Teddy Bear"),
        ]
