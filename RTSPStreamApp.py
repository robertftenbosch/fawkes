import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QHBoxLayout, QLineEdit
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt


class RTSPStreamApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.rtsp_stream = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_rtsp_frame)

    def init_ui(self):
        self.setWindowTitle("RTSP Video Stream Viewer")
        self.setGeometry(100, 100, 900, 600)

        # ========== Main Layout ==========
        main_layout = QHBoxLayout()

        # ========== Left Side: Video Streams ==========
        video_layout = QVBoxLayout()

        # RTSP Video Feed (Main Display)
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        video_layout.addWidget(self.video_label)

        # Input fields for RTSP URL & Port
        self.rtsp_input = QLineEdit(self)
        self.rtsp_input.setPlaceholderText("Enter RTSP URL (e.g., rtsp://username:password@192.168.1.10:554/stream)")
        video_layout.addWidget(self.rtsp_input)

        self.port_input = QLineEdit(self)
        self.port_input.setPlaceholderText("Enter RTSP Port (Default: 554)")
        video_layout.addWidget(self.port_input)

        # Start and Stop buttons
        self.start_button = QPushButton("Start Stream", self)
        self.start_button.clicked.connect(self.start_rtsp_stream)
        video_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Stream", self)
        self.stop_button.clicked.connect(self.stop_rtsp_stream)
        self.stop_button.setEnabled(False)  # Disable until stream starts
        video_layout.addWidget(self.stop_button)

        # Quit Button
        self.quit_button = QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close_app)
        video_layout.addWidget(self.quit_button)

        main_layout.addLayout(video_layout, 2)  # Takes 2/3 width

        # ========== Right Side: Message Box ==========
        message_layout = QVBoxLayout()

        self.message_box = QTextEdit(self)
        self.message_box.setReadOnly(True)
        message_layout.addWidget(self.message_box)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Type a message...")
        message_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)
        message_layout.addWidget(self.send_button)

        main_layout.addLayout(message_layout, 1)  # Takes 1/3 width

        # Set Layout
        self.setLayout(main_layout)

    def start_rtsp_stream(self):
        """Start streaming the RTSP video based on user input."""
        rtsp_url = self.rtsp_input.text().strip()
        port = self.port_input.text().strip()

        if not rtsp_url:
            self.message_box.append("> Error: Please enter a valid RTSP URL!")
            return

        # Append port if not already included
        if port and ":" not in rtsp_url.split("/")[-1]:
            rtsp_url = f"{rtsp_url}:{port}"

        self.message_box.append(f"> Connecting to RTSP stream: {rtsp_url} ...")

        self.rtsp_stream = cv2.VideoCapture(rtsp_url)

        if not self.rtsp_stream.isOpened():
            self.message_box.append("> Error: Failed to connect to RTSP stream!")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.timer.start(30)  # Start frame update timer

    def update_rtsp_frame(self):
        """Update the RTSP video feed in the QLabel."""
        if self.rtsp_stream:
            ret, frame = self.rtsp_stream.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qimg))
            else:
                self.message_box.append("> Warning: Lost connection to RTSP stream.")

    def stop_rtsp_stream(self):
        """Stop the RTSP video stream."""
        if self.rtsp_stream:
            self.rtsp_stream.release()
            self.rtsp_stream = None

        self.video_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.timer.stop()
        self.message_box.append("> RTSP stream stopped.")

    def send_message(self):
        """Send a message to the message box."""
        text = self.input_field.text().strip()
        if text:
            self.message_box.append(f"> {text}")
            self.input_field.clear()

    def close_app(self):
        """Stop RTSP stream and close the app."""
        self.stop_rtsp_stream()
        self.close()

    def closeEvent(self, event):
        """Ensure all resources are released on close."""
        self.stop_rtsp_stream()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RTSPStreamApp()
    window.show()
    sys.exit(app.exec())
