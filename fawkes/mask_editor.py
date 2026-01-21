import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent
from PyQt6.QtCore import Qt, pyqtSignal


class ClickableLabel(QLabel):
    """QLabel that emits click coordinates."""
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.pos().x(), event.pos().y())


class MaskEditorDialog(QDialog):
    """Dialog for creating custom masks with anchor points."""

    ANCHOR_COLORS = {
        'left_eye': (255, 0, 0),       # Red (BGR for OpenCV)
        'right_eye': (0, 255, 0),      # Green
        'nose_tip': (255, 255, 0),     # Cyan
        'left_mouth': (255, 0, 255),   # Magenta
        'right_mouth': (0, 255, 255),  # Yellow
        'upper_lip': (128, 0, 255),    # Orange
        'lower_lip': (255, 128, 0),    # Light blue
        'chin': (0, 0, 255),           # Blue
    }
    ANCHOR_ORDER = [
        'left_eye', 'right_eye', 'nose_tip',
        'left_mouth', 'right_mouth',
        'upper_lip', 'lower_lip', 'chin'
    ]
    ANCHOR_LABELS = {
        'left_eye': 'Left Eye',
        'right_eye': 'Right Eye',
        'nose_tip': 'Nose Tip',
        'left_mouth': 'Left Mouth Corner',
        'right_mouth': 'Right Mouth Corner',
        'upper_lip': 'Upper Lip',
        'lower_lip': 'Lower Lip',
        'chin': 'Chin',
    }

    def __init__(self, parent=None, mask_renderer=None):
        super().__init__(parent)
        self.mask_renderer = mask_renderer
        self.current_image = None
        self.display_image = None
        self.image_path = None
        self.anchor_points = {}
        self.current_anchor_index = 0
        self.scale_factor = 1.0

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Mask Editor")
        self.setMinimumSize(500, 600)

        layout = QVBoxLayout()

        # Image preview area
        self.image_label = ClickableLabel(self)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("background-color: #333; border: 1px solid #666;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("No image loaded")
        self.image_label.clicked.connect(self.on_image_click)
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Instructions label
        self.instructions_label = QLabel(self)
        self.instructions_label.setWordWrap(True)
        self.update_instructions()
        layout.addWidget(self.instructions_label)

        # Load image button
        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        # Reset anchors button
        self.reset_button = QPushButton("Reset Anchor Points", self)
        self.reset_button.clicked.connect(self.reset_anchors)
        self.reset_button.setEnabled(False)
        layout.addWidget(self.reset_button)

        # Mask name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Mask Name:"))
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter a name for your mask")
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        # Save and Cancel buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Mask", self)
        self.save_button.clicked.connect(self.save_mask)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_instructions(self):
        """Update instruction text based on current state."""
        if self.current_image is None:
            text = "Instructions:\n1. Click 'Load Image' to select a PNG or JPG file"
        elif self.current_anchor_index < len(self.ANCHOR_ORDER):
            anchor = self.ANCHOR_ORDER[self.current_anchor_index]
            label = self.ANCHOR_LABELS.get(anchor, anchor)
            # Get color name based on BGR value
            color_names = {
                (255, 0, 0): 'RED',
                (0, 255, 0): 'GREEN',
                (255, 255, 0): 'CYAN',
                (255, 0, 255): 'MAGENTA',
                (0, 255, 255): 'YELLOW',
                (128, 0, 255): 'ORANGE',
                (255, 128, 0): 'LIGHT BLUE',
                (0, 0, 255): 'BLUE',
            }
            color = color_names.get(self.ANCHOR_COLORS[anchor], 'colored')
            placed = self.current_anchor_index
            total = len(self.ANCHOR_ORDER)
            text = f"Click to place anchor points ({placed}/{total}):\n"
            text += f"Next: {label} ({color} marker)"
        else:
            text = "All anchor points placed!\nEnter a name and click 'Save Mask'."
        self.instructions_label.setText(text)

    def load_image(self):
        """Open file dialog to load an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mask Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image_path = file_path
            # Load with alpha channel if present
            self.current_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.current_image is None:
                QMessageBox.warning(self, "Error", "Failed to load image")
                return

            self.reset_anchors()
            self.update_display()

    def reset_anchors(self):
        """Reset all anchor points."""
        self.anchor_points = {}
        self.current_anchor_index = 0
        self.save_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.update_display()
        self.update_instructions()

    def update_display(self):
        """Update the image display with current anchor points."""
        if self.current_image is None:
            return

        # Make a copy for display
        img = self.current_image.copy()

        # Convert to BGR if grayscale or BGRA
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # Create checkered background for transparency
            h, w = img.shape[:2]
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            checker_size = 10
            for y in range(0, h, checker_size):
                for x in range(0, w, checker_size):
                    if (x // checker_size + y // checker_size) % 2 == 0:
                        bg[y:y+checker_size, x:x+checker_size] = (80, 80, 80)
                    else:
                        bg[y:y+checker_size, x:x+checker_size] = (120, 120, 120)

            # Alpha blend
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            img_rgb = img[:, :, :3]
            img = (img_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)

        # Draw anchor points
        for anchor_name, point in self.anchor_points.items():
            color = self.ANCHOR_COLORS[anchor_name]
            # Convert BGR to RGB for display
            cv2.circle(img, point, 8, color, -1)
            cv2.circle(img, point, 10, (255, 255, 255), 2)

        # Scale image to fit display
        h, w = img.shape[:2]
        max_size = 400
        if w > h:
            self.scale_factor = max_size / w
        else:
            self.scale_factor = max_size / h

        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        scaled_img = cv2.resize(img, (new_w, new_h))

        # Convert to QPixmap
        rgb_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.display_image = scaled_img

    def on_image_click(self, x: int, y: int):
        """Handle click on image to place anchor point."""
        if self.current_image is None:
            return

        if self.current_anchor_index >= len(self.ANCHOR_ORDER):
            # All points placed, reset and start over
            self.reset_anchors()
            return

        # Convert display coordinates back to original image coordinates
        # Account for centering in the label
        label_w = self.image_label.width()
        label_h = self.image_label.height()

        if self.display_image is not None:
            disp_h, disp_w = self.display_image.shape[:2]
            # Calculate offset (image is centered)
            offset_x = (label_w - disp_w) // 2
            offset_y = (label_h - disp_h) // 2

            # Adjust click coordinates
            img_x = x - offset_x
            img_y = y - offset_y

            # Check if click is within the image
            if 0 <= img_x < disp_w and 0 <= img_y < disp_h:
                # Convert to original image coordinates
                orig_x = int(img_x / self.scale_factor)
                orig_y = int(img_y / self.scale_factor)

                anchor_name = self.ANCHOR_ORDER[self.current_anchor_index]
                self.anchor_points[anchor_name] = (orig_x, orig_y)
                self.current_anchor_index += 1
                self.reset_button.setEnabled(True)

                if self.current_anchor_index >= len(self.ANCHOR_ORDER):
                    self.save_button.setEnabled(True)

                self.update_display()
                self.update_instructions()

    def save_mask(self):
        """Save the custom mask to user directory."""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a name for the mask")
            return

        # Validate name (alphanumeric and underscores only)
        safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name.lower())
        if not safe_name:
            QMessageBox.warning(self, "Error", "Invalid mask name")
            return

        if len(self.anchor_points) < len(self.ANCHOR_ORDER):
            QMessageBox.warning(self, "Error", f"Please place all {len(self.ANCHOR_ORDER)} anchor points")
            return

        if self.mask_renderer is None:
            QMessageBox.warning(self, "Error", "Mask renderer not available")
            return

        try:
            # Save the mask
            success = self.mask_renderer.add_custom_mask(
                safe_name,
                self.image_path,
                self.anchor_points,
                name  # Display name
            )

            if success:
                QMessageBox.information(self, "Success", f"Mask '{name}' saved successfully!")
                self.accept()
            else:
                QMessageBox.warning(self, "Error", "Failed to save mask")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save mask: {str(e)}")
