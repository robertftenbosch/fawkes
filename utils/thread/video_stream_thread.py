import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


class VideoStreamThread(QThread):
    frame_received = pyqtSignal(QImage)

    def __init__(self, webcam, face_detection, parent=None):
        super().__init__(parent)
        self.webcam = webcam
        self.face_detection = face_detection
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.webcam.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.face_detection.get_face_mesh_on_black_background(frame)
                h, w, ch = faces.shape
                bytes_per_line = ch * w
                qimg = QImage(faces.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.frame_received.emit(qimg)

    def stop(self):
        self.running = False
        self.wait()