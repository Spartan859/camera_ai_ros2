#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from typing import Optional, Tuple, List, Any

import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    rs = None  # allow import on systems without RealSense

from .mslite_yolo_predictor import MSLiteYOLODetector


CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
PERSON_CLASS_ID = CLASS_NAMES.index('person')


class CameraAI:
    def __init__(
        self,
        mindir_path: str = './yolov8x.mindir',
        visualize: bool = False,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        nms_time_limit: float = 60.0,
        conf_free: bool = True,
        device_target: str = 'Ascend',
        detection_interval: int = 1,
        logger: Any | None = None,
    ) -> None:
        # logger: expect rclpy Logger or std logging-like with info/warn/error
        self._log = logger
        self._visualize = visualize
        self._mindir_path = mindir_path
        self._detector: Optional[MSLiteYOLODetector] = None
        self._detector_args = dict(
            mindir_path=self._mindir_path,
            img_size=640,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            conf_free=conf_free,
            nms_time_limit=nms_time_limit,
            device_target=device_target,
        )

        self._pipeline = None
        self._align = None
        self._colorizer = None

        self._detection_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Shared data
        self._latest_detections: List[dict] = []
        self._latest_color_frame = None
        self._latest_depth_frame = None
        self._latest_depth_heatmap = None
        self.is_running = False
        self._detection_interval = max(1, int(detection_interval))

    def start(self) -> bool:
        if self.is_running:
            if self._log:
                self._log.info('CameraAI is already running.')
            return True
        if rs is None:
            if self._log:
                self._log.error('pyrealsense2 not available')
            else:
                print('pyrealsense2 not available')
            return False
        if not self._initialize_camera():
            return False
        self._stop_event.clear()
        self._detection_thread = threading.Thread(target=self._run_detection_loop, daemon=True)
        self._detection_thread.start()
        self.is_running = True
        if self._log:
            self._log.info('✅ CameraAI service started successfully (NPU initializing in background).')
        return True

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        if self._detection_thread:
            self._detection_thread.join(timeout=5)
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self.is_running = False
        if self._log:
            self._log.info('🧹 Resources cleaned up.')

    def get_latest_person_detections(self) -> List[dict]:
        with self._lock:
            return list(self._latest_detections)

    def is_safe(self, person_safe_dist: float = 1.5, obstacle_safe_dist: float = 1.0, obstacle_threshold_ratio: float = 0.05):
        with self._lock:
            if self._latest_depth_frame is None:
                return False, 'WARNING: No depth data available'
            for person in self._latest_detections:
                dist = float(person.get('distance_m', 0.0))
                if 0.01 < dist < person_safe_dist:
                    return False, f'STOP: Person too close at {dist:.2f}m'
            depth_frame = self._latest_depth_frame
            h, w = depth_frame.get_height(), depth_frame.get_width()
            roi_x_start, roi_x_end = w // 4, w * 3 // 4
            roi_y_start, roi_y_end = h // 2, h
            import numpy as np
            roi = np.asanyarray(depth_frame.get_data())[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            close_obstacle_pixels = roi[(roi < obstacle_safe_dist * 1000) & (roi > 10)]
            roi_area = (roi_x_end - roi_x_start) * (roi_y_end - roi_y_start)
            if len(close_obstacle_pixels) > roi_area * obstacle_threshold_ratio:
                return False, 'STOP: Obstacle detected ahead'
            return True, 'Path Clear'

    def _initialize_detector(self) -> bool:
        try:
            if self._log:
                self._log.info('🧠 Initializing MindSpore Lite YOLO detector...')
            self._detector = MSLiteYOLODetector(**self._detector_args, logger=self._log)
            if self._log:
                self._log.info('✅ YOLO mindir model loaded successfully.')
            return True
        except Exception as e:
            if self._log:
                self._log.error(f'❌ Failed to load mindir model: {e}')
            else:
                print(f'Failed to load mindir model: {e}')
            return False

    def _initialize_camera(self) -> bool:
        try:
            if self._log:
                self._log.info('📷 Initializing RealSense D455 camera...')
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
            self._pipeline.start(cfg)
            self._align = rs.align(rs.stream.color)
            self._colorizer = rs.colorizer()
            self._colorizer.set_option(rs.option.color_scheme, 2)
            if self._log:
                self._log.info('✅ RealSense camera initialized successfully.')
            return True
        except Exception as e:
            if self._log:
                self._log.error(f'❌ Failed to initialize RealSense camera: {e}')
            else:
                print(f'Failed to initialize RealSense: {e}')
            return False

    def _run_detection_loop(self) -> None:
        if not self._initialize_detector():
            self.is_running = False
            return
        import cv2
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
                if not frames:
                    time.sleep(0.1)
                    continue
                aligned = self._align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                result = self._detector.predict(color_image)
                boxes = result.get('bbox', [])
                cats = result.get('category_id', [])
                scores = result.get('score', [])
                person_dets = []
                for box, cid, score in zip(boxes, cats, scores):
                    if cid == PERSON_CLASS_ID:
                        dist = self._get_center_distance(depth_frame, box)
                        person_dets.append({'box': box, 'score': float(score), 'distance_m': round(dist, 2)})
                with self._lock:
                    self._latest_detections = person_dets
                    self._latest_color_frame = color_image
                    self._latest_depth_frame = depth_frame
                    self._latest_depth_heatmap = None
            except Exception as e:
                if self._log:
                    self._log.warn(f'Error in detection loop: {e}')
                else:
                    print(f'Error in detection loop: {e}')
                time.sleep(1)
            time.sleep(self._detection_interval)

    @staticmethod
    def _get_center_distance(depth_frame, box: list) -> float:
        try:
            x, y, w, h = box
            cx = int(x + w // 2)
            cy = int(y + h // 2)
            fw, fh = depth_frame.get_width(), depth_frame.get_height()
            cx = max(0, min(cx, fw - 1))
            cy = max(0, min(cy, fh - 1))
            dist = depth_frame.get_distance(cx, cy)
            if 0.01 < dist < 20.0:
                return float(dist)
            return 0.0
        except Exception:
            return 0.0
