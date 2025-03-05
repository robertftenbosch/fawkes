import sys
import json
import asyncio
import threading
import cv2
import numpy as np
import websockets
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from aiortc import RTCPeerConnection, MediaStreamTrack, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer
# pip install PyQt6 aiortc websockets fastapi uvicorn opencv-python numpy

SIGNALING_SERVER = "ws://localhost:8000/ws/"
CLIENT_ID = "peer1"
TARGET_ID = "peer2"
ice_server = RTCIceServer(urls="stun:stun.l.google.com:19302")
# ICE Servers for NAT Traversal
ice_servers = [
    ice_server  # Free STUN
]

class VideoStream(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)  # Open webcam
        self.running = True

    async def recv(self):
        if not self.running:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def stop(self):
        self.running = False
        self.cap.release()

class VideoChatApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("P2P Video Chat")
        self.setGeometry(100, 100, 800, 600)

        # UI Components
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        self.status_label = QLabel("Status: Ready")

        self.start_button = QPushButton("Start Call")
        self.start_button.clicked.connect(self.start_video_chat)

        self.mute_button = QPushButton("Mute Audio")
        self.mute_button.setEnabled(False)
        self.mute_button.clicked.connect(self.toggle_mute)

        self.end_button = QPushButton("End Call")
        self.end_button.setEnabled(False)
        self.end_button.clicked.connect(self.end_call)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.mute_button)
        button_layout.addWidget(self.end_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Video & WebRTC setup
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        config = RTCConfiguration()
        config.iceServers = ice_servers# ICE Servers for NAT Traversal
        self.peer_connection = RTCPeerConnection(config)
        self.video_track = None
        self.ws = None
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.start_event_loop, daemon=True).start()
        self.is_muted = False

    def start_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start_video_chat(self):
        self.start_button.setEnabled(False)
        self.mute_button.setEnabled(True)
        self.end_button.setEnabled(True)
        self.status_label.setText("Status: Connecting...")

        self.video_track = VideoStream()
        self.peer_connection.addTrack(self.video_track)

        asyncio.run_coroutine_threadsafe(self.connect_signaling(), self.loop)
        self.timer.start(30)

    async def connect_signaling(self):
        self.ws = await websockets.connect(SIGNALING_SERVER + CLIENT_ID)
        self.status_label.setText("Status: Connected")

        @self.peer_connection.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                await self.ws.send(json.dumps({
                    "type": "candidate",
                    "candidate": candidate.to_sdp(),
                    "target": TARGET_ID
                }))

        async def listen():
            async for message in self.ws:
                data = json.loads(message)

                if data["type"] == "offer":
                    await self.peer_connection.setRemoteDescription(RTCSessionDescription(data["offer"]["sdp"], data["offer"]["type"]))
                    answer = await self.peer_connection.createAnswer()
                    await self.peer_connection.setLocalDescription(answer)
                    await self.ws.send(json.dumps({
                        "type": "answer",
                        "answer": {
                            "sdp": self.peer_connection.localDescription.sdp,
                            "type": self.peer_connection.localDescription.type
                        },
                        "target": data["from"]
                    }))
                    self.status_label.setText("Status: Connected to Peer")

                elif data["type"] == "answer":
                    await self.peer_connection.setRemoteDescription(RTCSessionDescription(data["answer"]["sdp"], data["answer"]["type"]))
                    self.status_label.setText("Status: Connected to Peer")

                elif data["type"] == "candidate":
                    candidate = data["candidate"]
                    await self.peer_connection.addIceCandidate(candidate)

        asyncio.create_task(listen())

        offer = await self.peer_connection.createOffer()
        await self.peer_connection.setLocalDescription(offer)
        await self.ws.send(json.dumps({
            "type": "offer",
            "offer": {
                "sdp": self.peer_connection.localDescription.sdp,
                "type": self.peer_connection.localDescription.type
            },
            "target": TARGET_ID
        }))

    def toggle_mute(self):
        self.is_muted = not self.is_muted
        for sender in self.peer_connection.getSenders():
            if sender.track.kind == "audio":
                sender.track.enabled = not self.is_muted

        self.mute_button.setText("Unmute" if self.is_muted else "Mute")

    def end_call(self):
        self.start_button.setEnabled(True)
        self.mute_button.setEnabled(False)
        self.end_button.setEnabled(False)
        self.status_label.setText("Status: Call Ended")

        if self.video_track:
            self.video_track.stop()
            self.video_track = None

        if self.peer_connection:
            self.peer_connection.close()

        if self.ws:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)

        self.timer.stop()
        self.video_label.clear()

    def update_video_frame(self):
        if not self.video_track:
            return

        ret, frame = self.video_track.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoChatApp()
    window.show()
    sys.exit(app.exec())