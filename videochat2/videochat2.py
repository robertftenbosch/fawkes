import sys
import asyncio
import json
import os
import cv2
import aiohttp
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QLabel
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, VideoStreamTrack
from av import VideoFrame

from utils.constants import SIGNALING_SERVER


class VideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

    async def recv(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return VideoFrame.from_ndarray(frame, format="rgb24")

class VideoCallTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.peer_connection = RTCPeerConnection()
        self.local_video_track = VideoTrack()
        self.peer_connection.addTrack(self.local_video_track)

        self.init_ui()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.setup_webrtc())

    def init_ui(self):
        layout = QVBoxLayout()

        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        self.start_button = QPushButton("Start Video Call", self)
        self.start_button.clicked.connect(self.start_video)
        layout.addWidget(self.start_button)

        self.setLayout(layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)

    async def setup_webrtc(self):
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(f"{SIGNALING_SERVER}/peer1")
        self.ws_task = asyncio.create_task(self.listen_to_signaling())

    async def listen_to_signaling(self):
        async for msg in self.ws:
            data = json.loads(msg.data)
            if "sdp" in data:
                await self.peer_connection.setRemoteDescription(
                    RTCSessionDescription(data["sdp"], data["type"])
                )
                if data["type"] == "offer":
                    answer = await self.peer_connection.createAnswer()
                    await self.peer_connection.setLocalDescription(answer)
                    await self.ws.send_json({
                        "sdp": answer.sdp,
                        "type": answer.type,
                        "target": "peer1"
                    })

    def start_video(self):
        self.timer.start(30)

    def update_video_frame(self):
        ret, frame = self.local_video_track.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, ch = frame.shape
            bytes_per_line = ch * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))