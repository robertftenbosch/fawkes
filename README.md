# Fawkes

Privacy-preserving video communication using face mesh. Stream your face as landmarks only—no actual video is transmitted.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.10+.

```bash
# Clone the repository
git clone git@github.com:robertftenbosch/fawkes.git
cd fawkes

# Install dependencies
uv sync

# Run the application
uv run fawkes
```

## Usage

**Main application** - Webcam capture with face mesh streaming over UDP:
```bash
uv run fawkes
```

**RTSP viewer** - Connect to IP cameras:
```bash
uv run fawkes-rtsp
```

## Building Standalone Binary

```bash
# Install build dependencies
uv sync --extra build

# Build the executable
uv run pyinstaller fawkes.spec

# Binary will be in dist/fawkes
./dist/fawkes
```

## How it works

Fawkes captures your webcam feed, detects your face using MediaPipe, and transmits only the face mesh landmarks on a black background. The recipient sees your facial expressions without seeing your actual face.

- Video: UDP port 5005 (configurable)
- Audio: UDP port 5006

## License

MIT
