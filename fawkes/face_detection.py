import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fawkes.mask_renderer import MaskRenderer, MaskType


# Face mesh connections for drawing (subset of key connections)
FACEMESH_TESSELATION = [
    (127, 34), (34, 139), (139, 127), (11, 0), (0, 37), (37, 11),
    (232, 231), (231, 120), (120, 232), (72, 37), (37, 39), (39, 72),
    (128, 121), (121, 47), (47, 128), (232, 121), (121, 128), (128, 232),
    (104, 69), (69, 67), (67, 104), (175, 171), (171, 148), (148, 175),
    (118, 50), (50, 101), (101, 118), (73, 39), (39, 40), (40, 73),
    (9, 151), (151, 108), (108, 9), (48, 115), (115, 131), (131, 48),
    (194, 211), (211, 204), (204, 194), (74, 40), (40, 185), (185, 74),
    (80, 81), (81, 82), (82, 80), (191, 80), (80, 81), (81, 191),
    (78, 95), (95, 88), (88, 78), (308, 324), (324, 318), (318, 308),
    (402, 318), (318, 324), (324, 402), (13, 14), (14, 17), (17, 13),
    (0, 267), (267, 269), (269, 0), (269, 270), (270, 409), (409, 269),
    (78, 191), (191, 80), (80, 78), (61, 185), (185, 40), (40, 61),
    (146, 91), (91, 181), (181, 146), (375, 321), (321, 405), (405, 375),
    (314, 17), (17, 84), (84, 314), (37, 72), (72, 38), (38, 37),
    (267, 37), (37, 0), (0, 267), (269, 270), (270, 267), (267, 269),
]

# Contour connections for face outline
FACEMESH_CONTOURS = [
    # Face oval
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
    # Lips outer
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (291, 61),
    # Lips inner
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
    (312, 311), (311, 310), (310, 415), (415, 308), (308, 78),
    # Left eye
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
    # Right eye
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263),
    # Left eyebrow
    (46, 53), (53, 52), (52, 65), (65, 55), (55, 70), (70, 63), (63, 105),
    (105, 66), (66, 107),
    # Right eyebrow
    (276, 283), (283, 282), (282, 295), (295, 285), (285, 300), (300, 293),
    (293, 334), (334, 296), (296, 336),
]


class FaceDetection:
    def __init__(self):
        model_path = self._get_model_path()

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        self.mask_renderer = MaskRenderer()
        self.current_mask = MaskType.FACE_MESH

    def _get_model_path(self) -> Path:
        """Get path to model file, works for both dev and PyInstaller builds."""
        module_dir = Path(__file__).parent
        model_path = module_dir / "assets" / "models" / "face_landmarker.task"

        if model_path.exists():
            return model_path

        # PyInstaller bundles files in _MEIPASS
        import sys
        if hasattr(sys, '_MEIPASS'):
            return Path(sys._MEIPASS) / "fawkes" / "assets" / "models" / "face_landmarker.task"

        return model_path

    def set_mask_type(self, mask_type: MaskType):
        """Set the current mask type for rendering."""
        self.current_mask = mask_type

    def process_frame(self, frame):
        """Process frame with current mask type.

        Args:
            frame: BGR frame from webcam

        Returns:
            Processed frame with face mesh or mask overlay on black background
        """
        if self.current_mask == MaskType.FACE_MESH:
            return self.get_face_mesh_on_black_background(frame)

        # Process frame to detect face landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.face_landmarker.detect(mp_image)

        # Create black background
        output = np.zeros(frame.shape, dtype=np.uint8)

        if results.face_landmarks:
            # Convert to format expected by mask_renderer
            face_landmarks = self._create_landmark_wrapper(results.face_landmarks[0], frame.shape)
            output = self.mask_renderer.render_mask(
                self.current_mask,
                face_landmarks,
                frame.shape
            )

        return output

    def _create_landmark_wrapper(self, landmarks, frame_shape):
        """Create a wrapper object that mimics the old mediapipe landmark format."""
        class LandmarkWrapper:
            def __init__(self, landmarks_list):
                self.landmark = landmarks_list

        # Convert normalized landmarks to wrapper format
        class NormalizedLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        landmark_list = [NormalizedLandmark(lm.x, lm.y, lm.z) for lm in landmarks]
        return LandmarkWrapper(landmark_list)

    def get_face_mesh_on_black_background(self, frame):
        """Detect face and render mesh on black background."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.face_landmarker.detect(mp_image)

        # Create black background
        h, w = frame.shape[:2]
        image = np.zeros((h, w, 3), dtype=np.uint8)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]

            # Convert normalized landmarks to pixel coordinates
            points = []
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))

            # Draw tesselation (mesh triangles)
            for connection in FACEMESH_TESSELATION:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    pt1 = points[start_idx]
                    pt2 = points[end_idx]
                    cv2.line(image, pt1, pt2, (128, 128, 128), 1)

            # Draw contours (face outline, eyes, lips, eyebrows)
            for connection in FACEMESH_CONTOURS:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    pt1 = points[start_idx]
                    pt2 = points[end_idx]
                    cv2.line(image, pt1, pt2, (0, 255, 0), 1)

            # Draw iris circles (approximate)
            # Left iris center (landmark 468)
            if len(points) > 468:
                cv2.circle(image, points[468], 5, (0, 255, 255), 1)
            # Right iris center (landmark 473)
            if len(points) > 473:
                cv2.circle(image, points[473], 5, (0, 255, 255), 1)

        return image
