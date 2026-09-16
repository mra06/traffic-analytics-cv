#!/usr/bin/env python3
"""
Traffic Analytics with YOLO
============================

Detects, tracks, and counts vehicles/pedestrians in a video using YOLOv8's
built-in multi-object tracker (ByteTrack/BoT-SORT). Supports:

  * Unique object counting via persistent track IDs (not just per-frame counts)
  * Optional line-crossing counter (in/out counts across a virtual line)
  * Saving the annotated video to disk
  * Headless mode (no display window) for servers / batch jobs
  * Live FPS overlay and end-of-run summary (optionally exported as JSON)

Usage
-----
    python traffic_analytics.py --source test_video.mp4
    python traffic_analytics.py --source 0                     # webcam
    python traffic_analytics.py --source test_video.mp4 --save out.mp4 --headless
    python traffic_analytics.py --source test_video.mp4 --line 0,540,1280,540
    python traffic_analytics.py --source test_video.mp4 --classes car truck bus --conf 0.4

Press 'q' (in the display window) or Ctrl+C to stop early.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("traffic_analytics")

DEFAULT_CLASSES = ["car", "truck", "bus", "motorcycle", "person", "bicycle"]


@dataclass
class LineCounter:
    """Counts track crossings of a virtual line (simple midpoint side-check)."""

    p1: tuple[int, int]
    p2: tuple[int, int]
    in_count: int = 0
    out_count: int = 0
    _last_side: dict[int, float] = field(default_factory=dict)

    def _side(self, point: tuple[float, float]) -> float:
        (x1, y1), (x2, y2) = self.p1, self.p2
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def update(self, track_id: int, center: tuple[float, float]) -> None:
        side = self._side(center)
        prev = self._last_side.get(track_id)
        if prev is not None and prev * side < 0:  # sign changed -> crossed
            if side > 0:
                self.in_count += 1
            else:
                self.out_count += 1
        self._last_side[track_id] = side

    def draw(self, frame) -> None:
        cv2.line(frame, self.p1, self.p2, (0, 0, 255), 2)
        cv2.putText(
            frame, f"IN: {self.in_count}  OUT: {self.out_count}",
            (self.p1[0] + 10, self.p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )


class TrafficAnalyzer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = YOLO(args.model)
        self.class_filter = set(args.classes) if args.classes else None
        self.seen_ids: dict[str, set[int]] = {}
        self.current_counts: dict[str, int] = {}
        self.line_counter: Optional[LineCounter] = None
        if args.line:
            x1, y1, x2, y2 = args.line
            self.line_counter = LineCounter((x1, y1), (x2, y2))

        self.cap = self._open_source(args.source)
        self.writer: Optional[cv2.VideoWriter] = None
        self._stop = False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, *_):
        log.info("Interrupt received, shutting down gracefully...")
        self._stop = True

    @staticmethod
    def _open_source(source: str) -> cv2.VideoCapture:
        # Allow numeric strings to mean a webcam index
        src: str | int = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source '{source}'.")
        return cap

    def _init_writer(self, frame_shape) -> None:
        if not self.args.save:
            return
        h, w = frame_shape[:2]
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = str(self.args.save)
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        log.info(f"Saving annotated video to '{out_path}'")

    def _class_allowed(self, name: str) -> bool:
        return self.class_filter is None or name in self.class_filter

    def _process_frame(self, frame):
        results = self.model.track(
            frame,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            persist=True,
            tracker=self.args.tracker,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()

        self.current_counts = {}
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            clss = boxes.cls.int().tolist()
            xyxy = boxes.xyxy.tolist()

            for track_id, cls_idx, box in zip(ids, clss, xyxy):
                name = self.model.names[cls_idx]
                if not self._class_allowed(name):
                    continue

                self.current_counts[name] = self.current_counts.get(name, 0) + 1
                self.seen_ids.setdefault(name, set()).add(track_id)

                if self.line_counter:
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    self.line_counter.update(track_id, (cx, cy))

        return annotated

    def _draw_overlay(self, frame, fps: float) -> None:
        y = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35
        for name, count in sorted(self.current_counts.items()):
            total = len(self.seen_ids.get(name, ()))
            text = f"{name.capitalize()}: now={count}  total={total}"
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 30

        if self.line_counter:
            self.line_counter.draw(frame)

    def run(self) -> None:
        window = "Traffic Analytics - YOLO"
        frame_count = 0
        t_start = time.time()
        fps_smooth = 0.0

        log.info("Starting video processing. Press 'q' to quit (display mode) or Ctrl+C.")

        while self.cap.isOpened() and not self._stop:
            t0 = time.time()
            success, frame = self.cap.read()
            if not success:
                log.info("End of stream reached.")
                break

            annotated = self._process_frame(frame)

            dt = time.time() - t0
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst_fps if frame_count == 0 else 0.9 * fps_smooth + 0.1 * inst_fps
            self._draw_overlay(annotated, fps_smooth)

            if self.writer is None and self.args.save:
                self._init_writer(annotated.shape)
            if self.writer is not None:
                self.writer.write(annotated)

            if not self.args.headless:
                cv2.imshow(window, annotated)
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Quit key pressed.")
                    break

            frame_count += 1
            if self.args.max_frames and frame_count >= self.args.max_frames:
                log.info(f"Reached max_frames={self.args.max_frames}, stopping.")
                break

        elapsed = time.time() - t_start
        self._cleanup()
        self._report(frame_count, elapsed)

    def _cleanup(self) -> None:
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()

    def _report(self, frame_count: int, elapsed: float) -> None:
        summary = {
            "frames_processed": frame_count,
            "elapsed_seconds": round(elapsed, 2),
            "avg_fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
            "unique_counts": {name: len(ids) for name, ids in self.seen_ids.items()},
        }
        if self.line_counter:
            summary["line_crossing"] = {
                "in": self.line_counter.in_count,
                "out": self.line_counter.out_count,
            }

        log.info("----- Run Summary -----")
        for k, v in summary.items():
            log.info(f"{k}: {v}")

        if self.args.report:
            Path(self.args.report).write_text(json.dumps(summary, indent=2))
            log.info(f"Summary saved to '{self.args.report}'")


def parse_line(value: str) -> tuple[int, int, int, int]:
    try:
        x1, y1, x2, y2 = (int(v) for v in value.split(","))
        return x1, y1, x2, y2
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "Line must be 'x1,y1,x2,y2', e.g. 0,540,1280,540"
        ) from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO-based traffic analytics with tracking.")
    p.add_argument("--source", default="test_video.mp4",
                   help="Video file path, or a number for a webcam index (default: test_video.mp4)")
    p.add_argument("--model", default="yolov8m.pt", help="YOLO model weights (default: yolov8m.pt)")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="Ultralytics tracker config, e.g. bytetrack.yaml or botsort.yaml")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    p.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    p.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES,
                   help=f"Class names to count (default: {DEFAULT_CLASSES}). Use --classes all to disable filtering.")
    p.add_argument("--line", type=parse_line, default=None,
                   help="Virtual counting line as 'x1,y1,x2,y2' for in/out crossing counts")
    p.add_argument("--save", type=str, default=None, help="Path to save annotated output video (e.g. out.mp4)")
    p.add_argument("--report", type=str, default=None, help="Path to save a JSON summary report")
    p.add_argument("--headless", action="store_true", help="Run without opening a display window")
    p.add_argument("--max-frames", dest="max_frames", type=int, default=None,
                   help="Stop after N frames (useful for testing)")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.classes == ["all"]:
        args.classes = None

    try:
        analyzer = TrafficAnalyzer(args)
    except FileNotFoundError as e:
        log.error(str(e))
        return 1

    try:
        analyzer.run()
    except Exception:
        log.exception("Unexpected error during processing")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
