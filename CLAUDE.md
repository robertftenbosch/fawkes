# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fawkes is a Python desktop application for privacy-preserving video communication. It captures webcam feeds, detects faces using MediaPipe, and streams only the face mesh landmarks (on a black background) over UDP—allowing real-time video calls without revealing the actual video feed.

## Running the Application

```bash
# Install with uv
uv sync

# Run main application (webcam + UDP streaming with face mesh)
uv run fawkes

# Run RTSP stream viewer (for viewing RTSP camera feeds)
uv run fawkes-rtsp
```

## Architecture

**Package Structure:**
```
fawkes/
├── __init__.py
├── main.py          # WebcamUDPApp - main application
├── rtsp.py          # RTSPStreamApp - RTSP viewer
└── face_detection.py # FaceDetection - MediaPipe wrapper
```

**Core Components:**

- `fawkes/main.py` - `WebcamUDPApp`: PyQt6 GUI that captures local webcam, extracts face mesh via MediaPipe, and streams it over UDP. Also receives incoming UDP video/audio streams.

- `fawkes/face_detection.py` - `FaceDetection`: Wraps MediaPipe FaceMesh to extract face landmarks and render them on a black background. This is the privacy layer—only mesh data is transmitted, not the actual video.

- `fawkes/rtsp.py` - `RTSPStreamApp`: Standalone RTSP stream viewer for connecting to IP cameras.

**Network Protocol:**
- Video: UDP on port 5005 (configurable), JPEG-encoded frames
- Audio: UDP on port 5006 (video port + 1), raw PCM via sounddevice
- Both send and receive on the same ports (bidirectional communication)

**Key Technologies:**
- PyQt6 for GUI
- OpenCV for video capture/encoding
- MediaPipe for face mesh detection
- sounddevice for cross-platform audio I/O
- Socket/UDP for network streaming

## Git Workflow

Always create a feature branch for changes, then open a pull request to main:

```bash
git checkout -b feature/my-change
# make changes and commit
git push -u origin feature/my-change
gh pr create
```
