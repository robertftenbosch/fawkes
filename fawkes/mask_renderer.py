import cv2
import json
import numpy as np
import shutil
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
        'left_eye': 263,       # Left eye outer corner
        'right_eye': 33,       # Right eye outer corner
        'nose_tip': 1,         # Nose tip
        'left_mouth': 61,      # Left mouth corner
        'right_mouth': 291,    # Right mouth corner
        'upper_lip': 13,       # Upper lip center
        'lower_lip': 14,       # Lower lip center
        'chin': 152,           # Chin
        # Keep old names for backward compatibility
        'left_eye_outer': 263,
        'right_eye_outer': 33,
    }

    def __init__(self):
        self.masks = {}
        self.mask_configs = {}
        self.custom_masks = {}  # Custom masks loaded from user directory
        self.custom_mask_configs = {}
        self._load_masks()
        self._load_user_masks()

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

    def _get_user_masks_path(self) -> Path:
        """Get path to user's custom masks directory (~/.fawkes/masks)."""
        user_dir = Path.home() / ".fawkes" / "masks"
        return user_dir

    def _get_user_config_path(self) -> Path:
        """Get path to user's mask config file (~/.fawkes/mask_config.json)."""
        return Path.home() / ".fawkes" / "mask_config.json"

    def _load_user_masks(self):
        """Load custom masks from user directory."""
        config_path = self._get_user_config_path()
        masks_path = self._get_user_masks_path()

        if not config_path.exists():
            return

        try:
            with open(config_path, 'r') as f:
                self.custom_mask_configs = json.load(f)

            for mask_name, config in self.custom_mask_configs.items():
                image_path = masks_path / config['image']
                if image_path.exists():
                    mask_img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                    if mask_img is not None:
                        self.custom_masks[mask_name] = mask_img
        except (json.JSONDecodeError, IOError):
            pass

    def reload_masks(self):
        """Reload all masks including custom ones."""
        self.masks = {}
        self.mask_configs = {}
        self.custom_masks = {}
        self.custom_mask_configs = {}
        self._load_masks()
        self._load_user_masks()

    def add_custom_mask(self, name: str, image_path: str, anchors: dict, display_name: str = None) -> bool:
        """Add a custom mask to the user directory.

        Args:
            name: Internal name for the mask (alphanumeric)
            image_path: Path to the source image file
            anchors: Dict with 'left_eye', 'right_eye', 'chin' as (x, y) tuples
            display_name: Human-readable name for display

        Returns:
            True if successful, False otherwise
        """
        masks_path = self._get_user_masks_path()
        config_path = self._get_user_config_path()

        # Create directories if they don't exist
        masks_path.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        src_path = Path(image_path)
        ext = src_path.suffix.lower()
        if ext not in ['.png', '.jpg', '.jpeg']:
            ext = '.png'

        dest_filename = f"{name}{ext}"
        dest_path = masks_path / dest_filename

        try:
            # Copy image to user masks directory
            shutil.copy2(image_path, dest_path)

            # Load existing config or create new one
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}

            # Add new mask entry with all provided anchors
            anchors_config = {}
            for anchor_name, point in anchors.items():
                anchors_config[anchor_name] = list(point)

            config[name] = {
                'image': dest_filename,
                'anchors': anchors_config,
                'description': display_name or name,
                'display_name': display_name or name
            }

            # Save config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            # Reload masks
            self.reload_masks()
            return True

        except (IOError, OSError) as e:
            print(f"Error saving custom mask: {e}")
            return False

    def get_face_landmarks(self, face_landmarks, frame_shape) -> dict:
        """Extract key landmark coordinates from MediaPipe face landmarks."""
        h, w = frame_shape[:2]
        landmarks = {}

        for name, idx in self.LANDMARK_INDICES.items():
            lm = face_landmarks.landmark[idx]
            landmarks[name] = np.array([lm.x * w, lm.y * h], dtype=np.float32)

        return landmarks

    def render_mask(self, mask_type, face_landmarks, frame_shape) -> np.ndarray:
        """Render the specified mask onto a black background aligned to face landmarks.

        Args:
            mask_type: The type of mask to render (MaskType enum or string for custom masks)
            face_landmarks: MediaPipe face landmarks object
            frame_shape: Shape of the output frame (height, width, channels)

        Returns:
            Frame with mask rendered on black background
        """
        # Handle both MaskType enum and string names for custom masks
        if isinstance(mask_type, MaskType):
            mask_name = mask_type.value
        else:
            mask_name = str(mask_type)

        # Check built-in masks first, then custom masks
        if mask_name in self.masks and mask_name in self.mask_configs:
            mask_img = self.masks[mask_name]
            config = self.mask_configs[mask_name]
        elif mask_name in self.custom_masks and mask_name in self.custom_mask_configs:
            mask_img = self.custom_masks[mask_name]
            config = self.custom_mask_configs[mask_name]
        else:
            # Return black frame if mask not found
            return np.zeros(frame_shape, dtype=np.uint8)

        # Get face landmarks in frame coordinates
        face_pts = self.get_face_landmarks(face_landmarks, frame_shape)

        # Get mask anchor points from config
        anchors = config['anchors']

        # Map anchor names to face landmark names (for backward compatibility)
        anchor_to_landmark = {
            'left_eye': 'left_eye',
            'right_eye': 'right_eye',
            'nose_tip': 'nose_tip',
            'left_mouth': 'left_mouth',
            'right_mouth': 'right_mouth',
            'upper_lip': 'upper_lip',
            'lower_lip': 'lower_lip',
            'chin': 'chin',
            # Backward compatibility for old configs
            'left_eye_outer': 'left_eye',
            'right_eye_outer': 'right_eye',
        }

        # Build source and destination point arrays dynamically
        src_list = []
        dst_list = []

        for anchor_name, anchor_point in anchors.items():
            landmark_name = anchor_to_landmark.get(anchor_name, anchor_name)
            if landmark_name in face_pts:
                src_list.append(anchor_point)
                dst_list.append(face_pts[landmark_name])

        # Need at least 3 points for affine transform
        if len(src_list) < 3:
            # Fall back to basic 3-point if we have the minimum
            if 'left_eye' in anchors and 'right_eye' in anchors and 'chin' in anchors:
                src_list = [anchors['left_eye'], anchors['right_eye'], anchors['chin']]
                dst_list = [face_pts.get('left_eye', face_pts.get('left_eye_outer')),
                           face_pts.get('right_eye', face_pts.get('right_eye_outer')),
                           face_pts['chin']]
            else:
                return np.zeros(frame_shape, dtype=np.uint8)

        src_pts = np.array(src_list, dtype=np.float32)
        dst_pts = np.array(dst_list, dtype=np.float32)

        # Calculate affine transformation matrix
        if len(src_pts) == 3:
            # Exact 3-point affine transform
            transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)
        else:
            # Use least-squares estimation for more than 3 points
            transform_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts)

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
    def get_builtin_masks() -> list:
        """Return list of built-in mask types for UI dropdown."""
        return [
            (MaskType.FACE_MESH, "Face Mesh"),
            (MaskType.GUY_FAWKES, "Guy Fawkes"),
            (MaskType.MAGIC_MIRROR, "Magic Mirror"),
            (MaskType.TEDDY_BEAR, "Teddy Bear"),
        ]

    def get_available_masks(self) -> list:
        """Return list of all available masks including custom ones."""
        masks = self.get_builtin_masks()

        # Add custom masks
        for name, config in self.custom_mask_configs.items():
            display_name = config.get('display_name', name)
            masks.append((name, display_name))

        return masks
