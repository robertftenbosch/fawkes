import sys
import cv2
import numpy as np
import socket
import threading
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QHBoxLayout, QLineEdit, QSplitter, QSizePolicy
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, pyqtSignal, QThread, Qt
import pyaudio

from detection.face_detection import FaceDetection
from utils.thread.video_stream_thread import VideoStreamThread
from utils.ui.message_box import MessageTextEdit
from utils.ui.notification_center import NotificationCenter


class WebcamUDPApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

        # Initialize Webcam Capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.message_box.append("> Error: Unable to access the webcam!")
            return

        self.face_detection = FaceDetection()

        # Start a timer to update the local webcam display
        self.video_thread = VideoStreamThread(self.cap, self.face_detection)
        self.video_thread.frame_received.connect(self.update_webcam)
        self.video_thread.start()
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
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ========== Left Side: Video Streams ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Webcam Feed
        self.video_label = QLabel(self)
        # Instead of fixed size, we set a minimum size and an expanding size policy.
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.video_label)

        # UDP  UDP Received Stream Feed
        self.udp_label = QLabel(self)
        self.udp_label.setMinimumSize(320, 240)
        self.udp_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.udp_label)

        # UDP Stream Input Fields
        self.udp_ip_input = QLineEdit(self)
        self.udp_ip_input.setPlaceholderText("Enter UDP Target IP (e.g., 192.168.1.100)")
        left_layout.addWidget(self.udp_ip_input)

        self.udp_port_input = QLineEdit(self)
        self.udp_port_input.setPlaceholderText("Enter UDP Port (Default: 5005)")
        left_layout.addWidget(self.udp_port_input)

        # UDP Stream Start/Stop Buttons
        self.start_udp_button = QPushButton("Start UDP Stream", self)
        self.start_udp_button.clicked.connect(self.start_udp_stream)
        left_layout.addWidget(self.start_udp_button)

        self.stop_udp_button = QPushButton("Stop UDP Stream", self)
        self.stop_udp_button.clicked.connect(self.stop_udp_stream)
        self.stop_udp_button.setEnabled(False)
        left_layout.addWidget(self.stop_udp_button)

        # Quit Button
        self.quit_button = QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close_app)
        left_layout.addWidget(self.quit_button)

        main_splitter.addWidget(left_widget)

        # ========== Right Side: Message Box ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        # --- Message Area with Drag-and-Drop Upload ---
        message_area = QVBoxLayout()
        self.message_box = MessageTextEdit(self)
        self.message_box.setPlaceholderText("Type a message or drop a file here to upload...")
        self.message_box.file_dropped.connect(self.upload_file_over_udp)
        message_area.addWidget(self.message_box)

        # Text input and Send Button for chat messages
        send_layout = QHBoxLayout()
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Type a message...")
        send_layout.addWidget(self.input_field)
        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)
        send_layout.addWidget(self.send_button)
        message_area.addLayout(send_layout)
        right_layout.addLayout(message_area)

        # Notification Center (fixed height for notifications)
        self.notification_center = NotificationCenter(self)
        self.notification_center.setMinimumHeight(50)
        right_layout.addWidget(self.notification_center)

        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)

        # Main layout for the window
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

    def update_webcam(self, qimg):
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def start_udp_stream(self):
        """Start streaming the webcam feed over UDP."""
        udp_ip = self.udp_ip_input.text().strip()
        udp_port = self.udp_port_input.text().strip()

        if not udp_ip:
            self.notification_center.append("> Error: Please enter a valid UDP IP!")
            return

        if not udp_port:
            udp_port = "5005"  # Default UDP port

        self.udp_address = (udp_ip, int(udp_port))
        self.audio_address = (udp_ip, int(udp_port) + 1)  # Audio port

        self.notification_center.append(f"> Starting UDP Video Stream to {udp_ip}:{udp_port} ...")
        self.notification_center.append(f"> Starting UDP Audio Stream to {udp_ip}:{int(udp_port) + 1} ...")

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
            faces = self.face_detection.get_face_mesh_on_black_background(frame)
            _, encoded = cv2.imencode(".jpg", faces, [cv2.IMWRITE_JPEG_QUALITY, 80])
            sock.sendto(encoded.tobytes(), self.udp_address)

        sock.close()

    def stream_audio_udp(self):
        """Stream the audio feed over UDP."""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while self.udp_streaming:
            data = stream.read(CHUNK)
            sock.sendto(data, self.audio_address)

        stream.stop_stream()
        stream.close()
        audio.terminate()
        sock.close()

    def receive_audio_stream(self):
        """Receive and play the audio stream over UDP."""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 5006))  # Audio reception

        while self.audio_running:
            data, _ = sock.recvfrom(CHUNK * 2)
            stream.write(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()
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
            self.notification_center.append("> UDP Stream Stopped.")
            self.start_udp_button.setEnabled(True)
            self.stop_udp_button.setEnabled(False)

    def send_message(self):
        """Send a message to the message box."""
        text = self.input_field.text().strip()
        if text:
            self.message_box.append(f"> {text}")
            self.input_field.clear()

    def upload_file_over_udp(self, file_path):
        """Handle file drop: Upload the file over UDP in chunks."""
        self.notification_center.append(f"> Uploading file: {file_path}")
        if not hasattr(self, "udp_address"):
            self.notification_center.append("> Error: UDP address not configured!")
            return

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(1024)  # Chunk size (adjust as needed)
                    if not data:
                        break
                    sock.sendto(data, self.udp_address)
            sock.close()
            self.notification_center.append("> File upload completed.")
        except Exception as e:
            self.notification_center.append(f"> Error uploading file: {str(e)}")
    def close_app(self):
        """Stop UDP stream and close the app."""
        self.stop_udp_stream()
        self.cap.release()
        self.video_thread.stop()
        self.udp_running = False
        self.audio_running = False
        self.close()

    def closeEvent(self, event):
        """Ensure all resources are released on close."""
        self.close_app()
        event.accept()
def load_stylesheet(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading stylesheet: {e}")
        return ""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    stylesheet = load_stylesheet("style/style.qss")
    if stylesheet:
        app.setStyleSheet(stylesheet)
    window = WebcamUDPApp()
    window.show()
    sys.exit(app.exec())
