# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fawkes is a Python desktop application for privacy-preserving video communication. It captures webcam feeds, detects faces using MediaPipe, and streams only the face mesh landmarks (on a black background) over UDP—allowing real-time video calls without revealing the actual video feed.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run main application (webcam + UDP streaming with face mesh)
python main.py

# Run RTSP stream viewer (for viewing RTSP camera feeds)
python RTSPStreamApp.py
```

## Architecture

**Core Components:**

- `main.py` - `WebcamUDPApp`: PyQt6 GUI that captures local webcam, extracts face mesh via MediaPipe, and streams it over UDP. Also receives incoming UDP video/audio streams. Uses daemon threads for UDP/audio receive loops and a QTimer for frame updates.

- `face_detection.py` - `FaceDetection`: Wraps MediaPipe FaceMesh to extract face landmarks and render them on a black background. This is the privacy layer—only mesh data is transmitted, not the actual video.

- `RTSPStreamApp.py` - `RTSPStreamApp`: Standalone RTSP stream viewer for connecting to IP cameras.

- `example_cv.py` - Reference implementation showing MediaPipe face mesh and hand detection.

**Network Protocol:**
- Video: UDP on port 5005 (configurable), JPEG-encoded frames
- Audio: UDP on port 5006 (video port + 1), raw PCM data via PyAudio
- Both send and receive on the same ports (bidirectional communication)

**Key Technologies:**
- PyQt6 for GUI
- OpenCV for video capture/encoding
- MediaPipe for face mesh detection
- PyAudio for audio I/O
- Socket/UDP for network streaming
