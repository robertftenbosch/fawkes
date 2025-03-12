import asyncio
import json
import os

from PyQt6.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QTextEdit, QPushButton
from aiortc import RTCPeerConnection, RTCSessionDescription
import aiohttp

from utils.constants import SIGNALING_SERVER, TARGET_ID


class FileTransferTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.peer_connection = RTCPeerConnection()
        self.channel = self.peer_connection.createDataChannel("file_transfer")
        self.channel.on("open", self.on_channel_open)
        self.channel.on("message", self.on_channel_message)

        self.init_ui()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.setup_webrtc())

    def init_ui(self):
        layout = QVBoxLayout()

        self.log_box = QTextEdit(self)
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.send_button = QPushButton("Selecteer en verstuur bestand", self)
        self.send_button.clicked.connect(self.select_and_send_file)
        layout.addWidget(self.send_button)

        self.setLayout(layout)

    async def setup_webrtc(self):
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(f"{SIGNALING_SERVER}/{TARGET_ID}")
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
                        "target": TARGET_ID
                    })

    def on_channel_open(self):
        self.log("Kanaal geopend, klaar om bestanden te verzenden.")

    def on_channel_message(self, message):
        self.log(f"Ontvangen bericht: {message}")

    def select_and_send_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecteer bestand")
        if file_path:
            self.log(f"Bestand geselecteerd: {file_path}")
            self.send_file(file_path)

    def send_file(self, file_path):
        try:
            with open(file_path, "rb") as file:
                data = file.read()
                self.channel.send(data)
                self.log(f"Bestand {os.path.basename(file_path)} verzonden.")
        except Exception as e:
            self.log(f"Fout bij verzenden bestand: {e}")
    def log(self, message):
        self.log_box.append(message)