from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QTextEdit


class MessageTextEdit(QTextEdit):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setReadOnly(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            # For this demo, use only the first file dropped
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)
        else:
            super().dropEvent(event)