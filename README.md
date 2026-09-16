# Traffic Analytics — Computer Vision Pipeline

A real-time traffic analytics system built on **YOLOv8** that detects, **tracks**, and counts vehicles and pedestrians from traffic camera footage — with unique-object counting, directional line-crossing counts, and exportable results.

## What It Does

- Detects **cars, trucks, buses, motorcycles, bicycles, and pedestrians** in real time
- **Tracks** each object across frames with a persistent ID (ByteTrack / BoT-SORT), so vehicles aren't recounted every frame
- Reports both **live on-screen counts** and **unique total counts** per class
- Optional **line-crossing counter** for directional flow (in / out)
- Works on video files, webcams, or headless servers
- Can **export the annotated video** and a **JSON summary report**

## Screenshots

![Detection Example 1](image%201.PNG)

![Detection Example 2](image%202.PNG)

## Tech Stack

- **YOLOv8** (Ultralytics) — object detection
- **ByteTrack / BoT-SORT** (via Ultralytics) — multi-object tracking
- **OpenCV** — video I/O, drawing, and display
- **Python 3.10+**

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/traffic-analytics-cv.git
cd traffic-analytics-cv

# Create a virtual environment and install dependencies
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

`requirements.txt`:
```
ultralytics
opencv-python-headless   # use opencv-python instead if you need a display window and it's unavailable headless
lap
```

## Usage

Basic run, same as before — just point it at a video:

```bash
python traffic_analytics.py --source test_video.mp4
```

Press **q** in the video window, or **Ctrl+C** in the terminal, to stop early.

### Common options

| Flag | Description | Default |
|---|---|---|
| `--source` | Video file path, or a number for a webcam index (`0`, `1`, ...) | `test_video.mp4` |
| `--model` | YOLO weights file | `yolov8m.pt` |
| `--tracker` | Tracker config: `bytetrack.yaml` or `botsort.yaml` | `bytetrack.yaml` |
| `--conf` | Confidence threshold | `0.3` |
| `--imgsz` | Inference resolution | `640` |
| `--classes` | Classes to count (space-separated), or `all` | car, truck, bus, motorcycle, person, bicycle |
| `--line x1,y1,x2,y2` | Draw a virtual line and count in/out crossings | off |
| `--save out.mp4` | Save the annotated video to disk | off |
| `--report summary.json` | Save a JSON summary at the end of the run | off |
| `--headless` | Run without opening a display window (for servers) | off |
| `--max-frames N` | Stop after N frames (useful for quick tests) | off |

### Examples

```bash
# Webcam, restrict to vehicles only
python traffic_analytics.py --source 0 --classes car truck bus motorcycle

# Directional counting across a line at y=540 on a 1280-wide frame
python traffic_analytics.py --source test_video.mp4 --line 0,540,1280,540

# Headless batch processing on a server, saving video + report
python traffic_analytics.py --source cam_feed.mp4 --headless --save output.mp4 --report summary.json
```

## How It Works

1. Each video frame is fed into a pre-trained YOLOv8 model running in **tracking mode** (`model.track(...)`), which assigns a persistent ID to every detected object across frames.
2. Detections are filtered by class (car, truck, bus, motorcycle, person, bicycle, or a custom list).
3. Each track ID is added to a per-class **seen-IDs set**, giving an accurate unique total instead of a per-frame count that double-counts objects still in view.
4. If a counting line is configured, each track's midpoint is checked frame-to-frame for a side change, incrementing an in/out counter on crossing.
5. Live counts, unique totals, FPS, and the counting line (if enabled) are drawn on the frame.
6. At the end of the run, a summary (frame count, elapsed time, average FPS, unique counts, line-crossing totals) is logged and optionally saved as JSON.

## Future Improvements

- [x] ~~Object tracking with unique IDs (ByteTrack / DeepSORT)~~ — done via Ultralytics' built-in ByteTrack/BoT-SORT
- [x] ~~Line-crossing counter for directional traffic flow~~ — done, configurable via `--line`
- [ ] **CSV/DB data logging with timestamps** — log each crossing/detection event with a wall-clock timestamp for downstream analysis (e.g. per-hour traffic volume)
- [ ] **Speed estimation using homography** — map pixel coordinates to real-world distance and estimate per-vehicle speed
- [ ] **Live RTSP / IP camera stream support** — accept `rtsp://` URLs as `--source` with reconnect-on-drop handling
- [ ] **Multiple counting lines / zones** — support several lines or polygonal zones (e.g. per-lane counts, wrong-way detection)
- [ ] **Web dashboard** — a lightweight Flask/FastAPI + WebSocket dashboard for live counts instead of (or alongside) the OpenCV window
- [ ] **Alerting** — configurable thresholds (e.g. congestion, wrong-way movement) that trigger a webhook or notification
- [ ] **Heatmaps** — accumulate detection density over time to visualize congestion hotspots
- [ ] **Model/tracker benchmarking mode** — a `--benchmark` flag to compare FPS/accuracy across YOLO model sizes and trackers on the same clip
- [ ] **Dockerfile** — containerized setup for consistent deployment on servers/edge devices
- [ ] **Config file support** — load all CLI options from a `config.yaml` for reproducible runs
- [ ] **Unit tests** — cover `LineCounter` crossing logic and CLI argument parsing
