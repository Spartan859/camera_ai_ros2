#!/usr/bin/env python3
import os
from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from vision_msgs.msg import BoundingBox2D

from .camera_utils_ms import CameraAI


class CameraAINode(Node):
    def __init__(self) -> None:
        super().__init__('camera_ai_node')

        # Parameters
        self.declare_parameter('mindir_path', os.path.join(os.getcwd(), 'yolov8x.mindir'))
        self.declare_parameter('visualize', False)
        self.declare_parameter('detection_interval', 1.0)
        self.declare_parameter('device_target', 'Ascend')
        self.declare_parameter('person_safe_dist', 1.5)
        self.declare_parameter('obstacle_safe_dist', 1.0)
        self.declare_parameter('obstacle_threshold_ratio', 0.05)
        self.declare_parameter('enabled', True)

        mindir_path = self.get_parameter('mindir_path').get_parameter_value().string_value
        visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        detection_interval = float(self.get_parameter('detection_interval').value)
        device_target = self.get_parameter('device_target').get_parameter_value().string_value

        self.person_safe_dist = float(self.get_parameter('person_safe_dist').value)
        self.obstacle_safe_dist = float(self.get_parameter('obstacle_safe_dist').value)
        self.obstacle_threshold_ratio = float(self.get_parameter('obstacle_threshold_ratio').value)
        enabled = self.get_parameter('enabled').get_parameter_value().bool_value

        # Publishers
        self.status_pub = self.create_publisher(String, 'safety/status', 10)
        self.dets_pub = self.create_publisher(Detection2DArray, 'camera/detections', 10)

        # Camera AI backend
        self.ai = None
        if enabled:
            try:
                self.ai = CameraAI(
                    mindir_path=mindir_path,
                    visualize=visualize,
                    device_target=device_target,
                    detection_interval=float(max(0.1, detection_interval)),
                    logger=self.get_logger()
                )
                self.get_logger().info('Initializing CameraAI')
                self.get_logger().info(f'status:{self.ai.start()}')
                if not self.ai.start():
                    self.get_logger().warn('CameraAI failed to start; node will publish WARNING status')
                self.get_logger().info('CameraAI initialized')
            except Exception as e:
                self.get_logger().error(f'Failed to initialize CameraAI: {e}')
                self.ai = None
        else:
            self.get_logger().warn('CameraAI disabled by parameter enabled=false')

        # Timer to publish status and detections (~same rate as detection_interval)
        pub_period = max(0.1, float(detection_interval))
        self.timer = self.create_timer(pub_period, self._on_timer)

        self.get_logger().info('camera_ai_node started')

    def _on_timer(self) -> None:
        # Publish safety status
        status_msg = String()
        if self.ai and self.ai.is_running:
            safe, reason = self.ai.is_safe(
                person_safe_dist=self.person_safe_dist,
                obstacle_safe_dist=self.obstacle_safe_dist,
                obstacle_threshold_ratio=self.obstacle_threshold_ratio,
            )
            status_msg.data = 'OK' if safe else reason
        else:
            status_msg.data = 'WARNING: CameraAI not running'
        self.status_pub.publish(status_msg)

        # Publish detections (persons only from CameraAI)
        dets_msg = Detection2DArray()
        dets_msg.header.stamp = self.get_clock().now().to_msg()
        dets_msg.header.frame_id = 'camera_color_optical_frame'

        if self.ai and self.ai.is_running:
            persons = self.ai.get_latest_person_detections()
            for p in persons:
                box = p.get('box', [0, 0, 0, 0])
                score = float(p.get('score', 0.0))
                # Build Detection2D
                det = Detection2D()
                det.header = dets_msg.header
                det.results = [ObjectHypothesisWithPose()]
                det.results[0].hypothesis.class_id = 'person'
                det.results[0].hypothesis.score = score
                x, y, w, h = [int(v) for v in box]
                det.bbox = BoundingBox2D()
                det.bbox.center.position.x = float(x + w / 2.0)
                det.bbox.center.position.y = float(y + h / 2.0)
                det.bbox.size_x = float(w)
                det.bbox.size_y = float(h)
                dets_msg.detections.append(det)

        self.dets_pub.publish(dets_msg)

    def destroy_node(self):
        try:
            if self.ai:
                self.ai.stop()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraAINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
