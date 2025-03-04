import sys
import cv2
import numpy as np
import socket
import threading
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QHBoxLayout, QLineEdit, QGridLayout, QSizePolicy, QSlider
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
import pyaudio

from face_detection import FaceDetection


class WebcamUDPApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

        # Initialize Webcam Capture
        self.cap = cv2.VideoCapture(0)

        self.face_detection = FaceDetection()

        # Start a timer to update the local webcam display
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_webcam)
        self.timer.start(30)  # Refresh rate in milliseconds

        # UDP Streaming Variables
        self.udp_streaming = False
        self.udp_thread = None
        self.audio_thread = None

        # Audio Control Variables
        self.input_volume = 1.0  # Default mic volume
        self.output_volume = 1.0  # Default speaker volume

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
        self.setWindowTitle("UDP Video & Audio Stream with Message Box on the Right")
        self.setGeometry(100, 100, 1000, 700)

        # ========== Main Layout ==========
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # ========== Left Side: Video & Controls ==========
        left_layout = QGridLayout()
        left_container = QWidget()
        left_container.setLayout(left_layout)
        main_layout.addWidget(left_container, 2)  # Takes 2/3 of the width

        # ========== Webcam Feed ==========
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(320, 180)
        left_layout.addWidget(self.video_label, 0, 0, 1, 2)

        # ========== UDP Received Stream ==========
        self.udp_label = QLabel(self)
        self.udp_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.udp_label.setMinimumSize(320, 180)
        left_layout.addWidget(self.udp_label, 1, 0, 1, 2)

        # ========== UDP IP & Port Inputs ==========
        self.udp_ip_input = QLineEdit(self)
        self.udp_ip_input.setText(self.get_local_ip())  # Set default IP to local machine
        self.udp_ip_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        left_layout.addWidget(self.udp_ip_input, 2, 0)

        self.udp_port_input = QLineEdit(self)
        self.udp_port_input.setPlaceholderText("Enter UDP Port (Default: 5005)")
        left_layout.addWidget(self.udp_port_input, 2, 1)

        # ========== Volume Controls ==========
        self.mic_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.mic_volume_slider.setMinimum(0)
        self.mic_volume_slider.setMaximum(100)
        self.mic_volume_slider.setValue(100)
        self.mic_volume_slider.valueChanged.connect(self.update_mic_volume)
        left_layout.addWidget(QLabel("Microphone Volume"), 3, 0)
        left_layout.addWidget(self.mic_volume_slider, 3, 1)

        self.speaker_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.speaker_volume_slider.setMinimum(0)
        self.speaker_volume_slider.setMaximum(100)
        self.speaker_volume_slider.setValue(100)
        self.speaker_volume_slider.valueChanged.connect(self.update_speaker_volume)
        left_layout.addWidget(QLabel("Speaker Volume"), 4, 0)
        left_layout.addWidget(self.speaker_volume_slider, 4, 1)

        # ========== Start/Stop Buttons ==========
        self.start_udp_button = QPushButton("Start UDP Stream")
        self.start_udp_button.clicked.connect(self.start_udp_stream)
        left_layout.addWidget(self.start_udp_button, 5, 0)

        self.stop_udp_button = QPushButton("Stop UDP Stream")
        self.stop_udp_button.clicked.connect(self.stop_udp_stream)
        self.stop_udp_button.setEnabled(False)
        left_layout.addWidget(self.stop_udp_button, 5, 1)

        # ========== Quit Button ==========
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close_app)
        left_layout.addWidget(self.quit_button, 6, 0, 1, 2)

        # ========== Right Side: Message Box ==========
        self.message_container = QWidget()
        self.message_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.message_layout = QVBoxLayout(self.message_container)

        self.message_box = QTextEdit(self)
        self.message_box.setReadOnly(True)
        self.message_layout.addWidget(self.message_box)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Type a message...")
        self.message_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.message_layout.addWidget(self.send_button)

        main_layout.addWidget(self.message_container, 1)


    def update_webcam(self):
        """Capture and display the webcam feed."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
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
            # adjusted_data = np.frombuffer(data, dtype=np.int16) * self.input_volume
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
            adjusted_data = np.frombuffer(data, dtype=np.int16) * self.output_volume
            stream.write(adjusted_data.astype(np.int16).tobytes())

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
            self.message_box.append("> UDP Stream Stopped.")
            self.start_udp_button.setEnabled(True)
            self.stop_udp_button.setEnabled(False)

    def send_message(self):
        """Send a message to the message box."""
        text = self.input_field.text().strip()
        if text:
            self.message_box.append(f"> {text}")
            self.input_field.clear()

    def update_mic_volume(self, value):
        """Adjust microphone volume based on slider input."""
        self.input_volume = value / 100.0

    def update_speaker_volume(self, value):
        """Adjust speaker volume based on slider input."""
        self.output_volume = value / 100.0

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

    def get_local_ip(self):
        """Retrieve the local machine's IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception as e:
            self.message_box.append(f"> Error: Unable to get local IP. {e}")
            return "127.0.0.1"  # Fallback to localhost


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamUDPApp()
    window.show()
    sys.exit(app.exec())
