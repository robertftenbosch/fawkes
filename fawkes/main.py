import sys
import cv2
import numpy as np
import socket
import threading
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QHBoxLayout, QLineEdit, QComboBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
import sounddevice as sd

from fawkes.face_detection import FaceDetection
from fawkes.mask_renderer import MaskRenderer
from fawkes.mask_editor import MaskEditorDialog

# Audio settings
CHUNK = 1024
CHANNELS = 1
RATE = 44100


class WebcamUDPApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize face detection first (needed for mask renderer reference)
        self.face_detection = FaceDetection()
        self.mask_renderer = self.face_detection.mask_renderer

        self.init_ui()

        # Initialize Webcam Capture
        self.cap = cv2.VideoCapture(0)

        # Start a timer to update the local webcam display
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_webcam)
        self.timer.start(30)  # Refresh rate in milliseconds

        # UDP Streaming Variables
        self.udp_streaming = False
        self.udp_thread = None
        self.audio_thread = None

        # UDP Receiver Variables
        self.udp_running = True
        self.udp_frame = None
        self.udp_thread = threading.Thread(target=self.receive_udp_stream, daemon=True)
        self.udp_thread.start()

        # Audio Streaming
        self.audio_running = True
        self.audio_receive_thread = threading.Thread(target=self.receive_audio_stream, daemon=True)
        self.audio_receive_thread.start()

    def init_ui(self):
        self.setWindowTitle("UDP Video & Audio Stream")
        self.setGeometry(100, 100, 900, 700)

        # ========== Main Layout ==========
        main_layout = QHBoxLayout()

        # ========== Left Side: Video Streams ==========
        video_layout = QVBoxLayout()

        # Webcam Feed (Top)
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 240)
        video_layout.addWidget(self.video_label)

        # UDP Received Stream Feed (Bottom)
        self.udp_label = QLabel(self)
        self.udp_label.setFixedSize(640, 240)
        video_layout.addWidget(self.udp_label)

        # Mask Selection Dropdown
        self.mask_combo = QComboBox(self)
        self._populate_mask_combo()
        self.mask_combo.currentIndexChanged.connect(self.on_mask_changed)
        video_layout.addWidget(self.mask_combo)

        # Edit Masks Button
        self.edit_masks_button = QPushButton("Edit Masks", self)
        self.edit_masks_button.clicked.connect(self.open_mask_editor)
        video_layout.addWidget(self.edit_masks_button)

        # UDP Stream Input Fields
        self.udp_ip_input = QLineEdit(self)
        self.udp_ip_input.setPlaceholderText("Enter UDP Target IP (e.g., 192.168.1.100)")
        video_layout.addWidget(self.udp_ip_input)

        self.udp_port_input = QLineEdit(self)
        self.udp_port_input.setPlaceholderText("Enter UDP Port (Default: 5005)")
        video_layout.addWidget(self.udp_port_input)

        # UDP Stream Start/Stop Buttons
        self.start_udp_button = QPushButton("Start UDP Stream", self)
        self.start_udp_button.clicked.connect(self.start_udp_stream)
        video_layout.addWidget(self.start_udp_button)

        self.stop_udp_button = QPushButton("Stop UDP Stream", self)
        self.stop_udp_button.clicked.connect(self.stop_udp_stream)
        self.stop_udp_button.setEnabled(False)
        video_layout.addWidget(self.stop_udp_button)

        # Quit Button
        self.quit_button = QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close_app)
        video_layout.addWidget(self.quit_button)

        main_layout.addLayout(video_layout, 2)

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

        main_layout.addLayout(message_layout, 1)

        self.setLayout(main_layout)

    def _populate_mask_combo(self):
        """Populate the mask dropdown with available masks."""
        self.mask_combo.blockSignals(True)
        current_data = self.mask_combo.currentData()
        self.mask_combo.clear()
        for mask_type, label in self.mask_renderer.get_available_masks():
            self.mask_combo.addItem(label, mask_type)
        # Restore selection if possible
        if current_data is not None:
            for i in range(self.mask_combo.count()):
                if self.mask_combo.itemData(i) == current_data:
                    self.mask_combo.setCurrentIndex(i)
                    break
        self.mask_combo.blockSignals(False)

    def on_mask_changed(self, index):
        """Handle mask selection change."""
        mask_type = self.mask_combo.itemData(index)
        if mask_type is not None:
            self.face_detection.set_mask_type(mask_type)
            self.message_box.append(f"> Mask changed to: {self.mask_combo.currentText()}")

    def open_mask_editor(self):
        """Open the mask editor dialog."""
        dialog = MaskEditorDialog(self, self.mask_renderer)
        if dialog.exec():
            # Refresh the mask dropdown after saving a new mask
            self._populate_mask_combo()
            # Select the last item (newly added mask)
            self.mask_combo.setCurrentIndex(self.mask_combo.count() - 1)
            self.message_box.append("> New mask added successfully!")

    def update_webcam(self):
        """Capture and display the webcam feed with face mesh/mask overlay."""
        ret, frame = self.cap.read()
        if ret:
            # Process frame with current mask type
            processed = self.face_detection.process_frame(frame)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            h, w, ch = processed.shape
            bytes_per_line = ch * w
            qimg = QImage(processed.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def start_udp_stream(self):
        """Start streaming the webcam feed over UDP."""
        udp_ip = self.udp_ip_input.text().strip()
        udp_port = self.udp_port_input.text().strip()

        if not udp_ip:
            self.message_box.append("> Error: Please enter a valid UDP IP!")
            return

        if not udp_port:
            udp_port = "5005"  # Default UDP port

        self.udp_address = (udp_ip, int(udp_port))
        self.audio_address = (udp_ip, int(udp_port) + 1)  # Audio port

        self.message_box.append(f"> Starting UDP Video Stream to {udp_ip}:{udp_port} ...")
        self.message_box.append(f"> Starting UDP Audio Stream to {udp_ip}:{int(udp_port) + 1} ...")

        # Start UDP Video Stream
        self.udp_streaming = True
        self.udp_thread = threading.Thread(target=self.stream_webcam_udp, daemon=True)
        self.udp_thread.start()

        # Start UDP Audio Stream
        self.audio_thread = threading.Thread(target=self.stream_audio_udp, daemon=True)
        self.audio_thread.start()

        self.start_udp_button.setEnabled(False)
        self.stop_udp_button.setEnabled(True)

    def stream_webcam_udp(self):
        """Stream the webcam feed over UDP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while self.udp_streaming:
            ret, frame = self.cap.read()
            if not ret:
                continue
            faces = self.face_detection.process_frame(frame)
            _, encoded = cv2.imencode(".jpg", faces, [cv2.IMWRITE_JPEG_QUALITY, 80])
            sock.sendto(encoded.tobytes(), self.udp_address)

        sock.close()

    def stream_audio_udp(self):
        """Stream the audio feed over UDP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16', blocksize=CHUNK) as stream:
            while self.udp_streaming:
                data, _ = stream.read(CHUNK)
                sock.sendto(data.tobytes(), self.audio_address)

        sock.close()

    def receive_audio_stream(self):
        """Receive and play the audio stream over UDP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 5006))  # Audio reception

        with sd.OutputStream(samplerate=RATE, channels=CHANNELS, dtype='int16', blocksize=CHUNK) as stream:
            while self.audio_running:
                data, _ = sock.recvfrom(CHUNK * 2)
                audio_data = np.frombuffer(data, dtype='int16').reshape(-1, CHANNELS)
                stream.write(audio_data)

        sock.close()

    def receive_udp_stream(self):
        """Receive and process video stream over UDP."""
        udp_ip = "0.0.0.0"  # Listen on all interfaces
        udp_port = 5005  # Port to receive data
        buffer_size = 65536  # Large buffer for incoming video packets

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((udp_ip, udp_port))

        while self.udp_running:
            packet, _ = sock.recvfrom(buffer_size)
            np_data = np.frombuffer(packet, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if frame is not None:
                self.udp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Update UDP video QLabel
                self.update_udp_frame()

        sock.close()

    def update_udp_frame(self):
        """Update the UDP video feed in the QLabel."""
        if self.udp_frame is not None:
            h, w, ch = self.udp_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(self.udp_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.udp_label.setPixmap(QPixmap.fromImage(qimg))

    def stop_udp_stream(self):
        """Stop the UDP stream."""
        if self.udp_streaming:
            self.udp_streaming = False
            self.message_box.append("> UDP Stream Stopped.")
            self.start_udp_button.setEnabled(True)
            self.stop_udp_button.setEnabled(False)

    def send_message(self):
        """Send a message to the message box."""
        text = self.input_field.text().strip()
        if text:
            self.message_box.append(f"> {text}")
            self.input_field.clear()

    def close_app(self):
        """Stop UDP stream and close the app."""
        self.stop_udp_stream()
        self.cap.release()
        self.close()

    def closeEvent(self, event):
        """Ensure all resources are released on close."""
        self.stop_udp_stream()
        self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = WebcamUDPApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
