import sys

from PyQt6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QMainWindow, QTabWidget

from browser.browser import BrowserApp
from file_share.file_share import FileTransferTab
from videochat.videochat import VideoChatApp
from videochat2.videochat2 import VideoCallTab


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Videochat en Browser")
        self.setGeometry(100, 100, 900, 600)  # Venstergrootte

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tabs toevoegen
        self.videochat_tab = VideoChatApp()
        self.browser_tab = BrowserApp()
        self.file_share = FileTransferTab()
        self.videochat2_tab = VideoCallTab()

        self.tabs.addTab(self.videochat_tab, "Videochat")
        self.tabs.addTab(self.browser_tab, "Browser")
        self.tabs.addTab(self.file_share, "File Share")
        self.tabs.addTab(self.videochat2_tab, "chat2")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())