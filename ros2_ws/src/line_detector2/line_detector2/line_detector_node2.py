import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # 웹캠 노드에서 뿌려주는 토픽
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        lanes, roi_start_y = self.detect_lanes(cv_image)

        height = cv_image.shape[0]
        width = cv_image.shape[1]

        lane_centers = []
        for lane in lanes:
            M = cv2.moments(lane)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cy_global = cy + roi_start_y
                lane_centers.append((cx, cy_global, lane))

        lane_centers = sorted(lane_centers, key=lambda c: -c[1])
        two_lanes = lane_centers[:2]

        if len(two_lanes) == 2:
            (cx1, cy1, lane1), (cx2, cy2, lane2) = two_lanes
            cv2.circle(cv_image, (cx1, cy1), 8, (255, 0, 0), -1)
            cv2.circle(cv_image, (cx2, cy2), 8, (255, 0, 0), -1)
            lane1_shifted = lane1.copy()
            lane2_shifted = lane2.copy()
            lane1_shifted[:, 0, 1] += roi_start_y
            lane2_shifted[:, 0, 1] += roi_start_y
            cv2.drawContours(cv_image, [lane1_shifted], -1, (0, 255, 0), 3)
            cv2.drawContours(cv_image, [lane2_shifted], -1, (0, 255, 0), 3)
            center_x = (cx1 + cx2) // 2
            center_y = (cy1 + cy2) // 2
            cv2.circle(cv_image, (center_x, center_y), 10, (0, 255, 255), -1)
            cv2.line(cv_image, (center_x, center_y), (center_x, height), (0, 255, 255), 2)
            print(f"차선1 중심: ({cx1},{cy1}), 차선2 중심: ({cx2},{cy2}), 중앙선: ({center_x},{center_y})")
        elif len(two_lanes) == 1:
            cx, cy, lane = two_lanes[0]
            lane_shifted = lane.copy()
            lane_shifted[:, 0, 1] += roi_start_y
            cv2.drawContours(cv_image, [lane_shifted], -1, (0, 255, 0), 3)
            cv2.circle(cv_image, (cx, cy), 8, (255, 0, 0), -1)
            print(f"한 개 차선만 검출: ({cx},{cy})")

        cv2.imshow("Lane Detection", cv_image)
        cv2.waitKey(1)

    def detect_lanes(self, image):
        height, width, _ = image.shape
        roi_ratio = 0.01
        roi_start_y = int(height * roi_ratio)
        roi = image[roi_start_y:, :]
        cv2.imshow('ROI', roi)

        # 노란색 HSV 변환 및 마스킹 (채도, 명도 기준 상향)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_lower = (13, 90, 90)  # S, V값을 140 이상으로 상향
        yellow_upper = (35, 255, 255)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        cv2.imshow('Yellow Mask', yellow_mask)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lane_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 800]
        return lane_contours, roi_start_y  # roi_start_y를 같이 반환

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
