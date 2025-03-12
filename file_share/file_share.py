import asyncio
import json
import os

from PyQt6.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QTextEdit, QPushButton
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
import aiohttp

from utils.constants import SIGNALING_SERVER, TARGET_ID

CHUNK_SIZE = 16384
class FileTransferTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.peer_connection = RTCPeerConnection()
        self.channel = None  # Initialized later when connection is set up
        self.file_buffer = bytearray()  # Buffer to store received chunks
        self.receiving_file = False  # Flag to track file reception
        self.file_name = None  # Name of the file being received

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

        @self.peer_connection.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                await self.ws.send_json({
                    "candidate": candidate.to_sdp(),
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                    "target": TARGET_ID
                })

        @self.peer_connection.on("datachannel")
        def on_datachannel(channel):
            self.channel = channel
            self.channel.on("open", self.on_channel_open)
            self.channel.on("message", self.on_channel_message)

        self.channel = self.peer_connection.createDataChannel("file_transfer")
        self.channel.on("open", self.on_channel_open)
        self.channel.on("message", self.on_channel_message)

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

            elif "candidate" in data:
                candidate = RTCIceCandidate(data["candidate"], data["sdpMid"], data["sdpMLineIndex"])
                await self.peer_connection.addIceCandidate(candidate)

    def on_channel_open(self):
        self.log("Kanaal geopend, klaar om bestanden te verzenden.")

    def on_channel_message(self, message):
        """Handles receiving file chunks and reassembling them."""
        if isinstance(message, str):
            # Expecting metadata message (e.g., file name, size)
            if message.startswith("FILE_METADATA:"):
                self.start_receiving_file(message)
        elif isinstance(message, bytes):
            self.receive_file_chunk(message)

    def start_receiving_file(self, metadata_message):
        """Initialize file receiving process based on metadata."""
        parts = metadata_message.split(":")
        if len(parts) == 3:
            _, self.file_name, file_size = parts
            self.file_size = int(file_size)
            self.file_buffer = bytearray()  # Reset buffer
            self.receiving_file = True
            self.log(f"Bestand {self.file_name} ontvangen... Grootte: {self.file_size} bytes.")

    def receive_file_chunk(self, chunk):
        """Buffer received file chunk and save when complete."""
        if self.receiving_file:
            self.file_buffer.extend(chunk)
            self.log(f"Ontvangen {len(chunk)} bytes... ({len(self.file_buffer)}/{self.file_size})")

            if len(self.file_buffer) >= self.file_size:
                self.save_received_file()

    def save_received_file(self):
        """Saves the received file from buffer."""
        save_path = os.path.join(os.getcwd(), self.file_name)
        with open(save_path, "wb") as file:
            file.write(self.file_buffer)

        self.log(f"Bestand {self.file_name} volledig ontvangen en opgeslagen!")
        self.file_buffer.clear()
        self.receiving_file = False

    def select_and_send_file(self):
        """Opens a file dialog and selects a file to send."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecteer bestand")
        if file_path:
            self.log(f"Bestand geselecteerd: {file_path}")
            self.send_file(file_path)

    def send_file(self, file_path):
        """Sends a file in chunks via WebRTC DataChannel."""
        try:
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)

            # Send metadata first
            metadata = f"FILE_METADATA:{file_name}:{file_size}"
            self.channel.send(metadata)
            self.log(f"Verzenden gestart voor {file_name} ({file_size} bytes)...")

            with open(file_path, "rb") as file:
                while chunk := file.read(CHUNK_SIZE):
                    self.channel.send(chunk)
                    self.log(f"Verzonden {len(chunk)} bytes...")

            self.log(f"Bestand {file_name} succesvol verzonden!")
        except Exception as e:
            self.log(f"Fout bij verzenden bestand: {e}")

    def log(self, message):
        """Logs messages to the UI."""
        self.log_box.append(message)