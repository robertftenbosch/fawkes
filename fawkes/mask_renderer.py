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
    """Renders mask overlays onto detected faces using triangulation-based warping."""

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

    def _warp_triangle(self, src_img: np.ndarray, dst_img: np.ndarray,
                       src_tri: np.ndarray, dst_tri: np.ndarray) -> None:
        """Warp a single triangle from source to destination image.

        Args:
            src_img: Source image (mask)
            dst_img: Destination image (output, modified in-place)
            src_tri: Source triangle vertices (3x2 array)
            dst_tri: Destination triangle vertices (3x2 array)
        """
        # Get bounding rectangles
        src_rect = cv2.boundingRect(np.float32([src_tri]))
        dst_rect = cv2.boundingRect(np.float32([dst_tri]))

        # Clip to image bounds
        src_h, src_w = src_img.shape[:2]
        dst_h, dst_w = dst_img.shape[:2]

        src_rect = (
            max(0, src_rect[0]),
            max(0, src_rect[1]),
            min(src_rect[2], src_w - src_rect[0]),
            min(src_rect[3], src_h - src_rect[1])
        )
        dst_rect = (
            max(0, dst_rect[0]),
            max(0, dst_rect[1]),
            min(dst_rect[2], dst_w - dst_rect[0]),
            min(dst_rect[3], dst_h - dst_rect[1])
        )

        if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
            return

        # Offset triangles to their bounding rectangles
        src_tri_rect = src_tri - np.array([src_rect[0], src_rect[1]], dtype=np.float32)
        dst_tri_rect = dst_tri - np.array([dst_rect[0], dst_rect[1]], dtype=np.float32)

        # Get the affine transform for this triangle
        warp_mat = cv2.getAffineTransform(
            np.float32(src_tri_rect),
            np.float32(dst_tri_rect)
        )

        # Extract source region
        x, y, w, h = src_rect
        if y + h > src_h or x + w > src_w:
            return
        src_crop = src_img[y:y+h, x:x+w]

        # Warp the source triangle region
        dst_w_rect, dst_h_rect = dst_rect[2], dst_rect[3]
        warped = cv2.warpAffine(
            src_crop,
            warp_mat,
            (dst_w_rect, dst_h_rect),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Create mask for the destination triangle
        mask = np.zeros((dst_h_rect, dst_w_rect), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_rect), 255)

        # Apply the warped triangle to the destination
        dx, dy = dst_rect[0], dst_rect[1]
        if dy + dst_h_rect > dst_h or dx + dst_w_rect > dst_w:
            return

        # Handle alpha channel
        if warped.shape[2] == 4:
            # Combine triangle mask with alpha
            alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
            tri_mask = (mask[:, :, np.newaxis] / 255.0).astype(np.float32)
            combined_alpha = alpha * tri_mask

            dst_region = dst_img[dy:dy+dst_h_rect, dx:dx+dst_w_rect]
            warped_bgr = warped[:, :, :3]

            blended = (warped_bgr * combined_alpha + dst_region * (1 - combined_alpha))
            dst_img[dy:dy+dst_h_rect, dx:dx+dst_w_rect] = blended.astype(np.uint8)
        else:
            mask_3ch = mask[:, :, np.newaxis] / 255.0
            dst_region = dst_img[dy:dy+dst_h_rect, dx:dx+dst_w_rect]
            blended = warped * mask_3ch + dst_region * (1 - mask_3ch)
            dst_img[dy:dy+dst_h_rect, dx:dx+dst_w_rect] = blended.astype(np.uint8)

    def _triangulate_warp(self, src_img: np.ndarray, src_pts: np.ndarray,
                          dst_pts: np.ndarray, output_shape: tuple) -> np.ndarray:
        """Warp source image using Delaunay triangulation.

        Args:
            src_img: Source mask image
            src_pts: Source anchor points (Nx2 array)
            dst_pts: Destination face points (Nx2 array)
            output_shape: Shape of output image (h, w, c)

        Returns:
            Warped image
        """
        src_h, src_w = src_img.shape[:2]
        dst_h, dst_w = output_shape[:2]

        # Add corner points to ensure full coverage
        src_corners = np.array([
            [0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1],
            [src_w // 2, 0], [src_w - 1, src_h // 2],
            [src_w // 2, src_h - 1], [0, src_h // 2]
        ], dtype=np.float32)

        dst_corners = np.array([
            [0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1],
            [dst_w // 2, 0], [dst_w - 1, dst_h // 2],
            [dst_w // 2, dst_h - 1], [0, dst_h // 2]
        ], dtype=np.float32)

        # Combine anchor points with corners
        all_src_pts = np.vstack([src_pts, src_corners])
        all_dst_pts = np.vstack([dst_pts, dst_corners])

        # Create Delaunay triangulation on destination points
        rect = (0, 0, dst_w, dst_h)
        subdiv = cv2.Subdiv2D(rect)

        # Insert points (need to be tuples)
        pt_indices = {}
        for i, pt in enumerate(all_dst_pts):
            x, y = float(pt[0]), float(pt[1])
            # Clamp to rect bounds
            x = max(0, min(dst_w - 1, x))
            y = max(0, min(dst_h - 1, y))
            try:
                subdiv.insert((x, y))
                pt_indices[(int(x), int(y))] = i
            except cv2.error:
                pass

        # Get triangles
        triangles = subdiv.getTriangleList()

        # Create output image
        output = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

        # Warp each triangle
        for tri in triangles:
            # Get triangle vertices
            pt1 = (tri[0], tri[1])
            pt2 = (tri[2], tri[3])
            pt3 = (tri[4], tri[5])

            # Check if triangle is within bounds
            if not (0 <= pt1[0] < dst_w and 0 <= pt1[1] < dst_h and
                    0 <= pt2[0] < dst_w and 0 <= pt2[1] < dst_h and
                    0 <= pt3[0] < dst_w and 0 <= pt3[1] < dst_h):
                continue

            # Find corresponding source triangle
            dst_tri = np.array([pt1, pt2, pt3], dtype=np.float32)

            # Find indices of these points
            idx1 = self._find_point_index(all_dst_pts, pt1)
            idx2 = self._find_point_index(all_dst_pts, pt2)
            idx3 = self._find_point_index(all_dst_pts, pt3)

            if idx1 is None or idx2 is None or idx3 is None:
                continue

            src_tri = np.array([
                all_src_pts[idx1],
                all_src_pts[idx2],
                all_src_pts[idx3]
            ], dtype=np.float32)

            # Warp this triangle
            self._warp_triangle(src_img, output, src_tri, dst_tri)

        return output

    def _find_point_index(self, points: np.ndarray, target: tuple, threshold: float = 2.0) -> int:
        """Find index of point closest to target within threshold."""
        target_arr = np.array(target, dtype=np.float32)
        distances = np.linalg.norm(points - target_arr, axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx] < threshold:
            return min_idx
        return None

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

        h, w = frame_shape[:2]

        # Use triangulation-based warping for better deformation
        if len(src_pts) >= 4:
            # Use Delaunay triangulation for smooth warping
            output = self._triangulate_warp(mask_img, src_pts, dst_pts, frame_shape)
        else:
            # Fall back to simple affine transform for 3 points
            transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)

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
                alpha = warped_mask[:, :, 3:4].astype(np.float32) / 255.0
                mask_bgr = warped_mask[:, :, :3]
                output = (mask_bgr * alpha + output * (1 - alpha)).astype(np.uint8)
            else:
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
