import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from piracer.vehicles import PiRacerPro

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.piracer = PiRacerPro()

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = cv_image.shape[:2]
        roi_ratio = 0.01
        roi_start_y = int(height * roi_ratio)
        roi = cv_image[roi_start_y:, :]

        # --- HSV 마스킹 ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_lower = (10, 90, 110)
        yellow_upper = (35, 255, 255)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        # --- 라인 중심 검출 ---
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lane_centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00']) + roi_start_y
                    lane_centers.append((cx, cy, cnt))

        lane_centers = sorted(lane_centers, key=lambda c: -c[1])
        two_lanes = lane_centers[:2]

        if len(two_lanes) == 2:
            (cx1, cy1, lane1), (cx2, cy2, lane2) = two_lanes
            center_x = (cx1 + cx2) // 2
            center_y = (cy1 + cy2) // 2

            # --- 시각화 ---
            cv2.circle(cv_image, (cx1, cy1), 8, (255, 0, 0), -1)
            cv2.circle(cv_image, (cx2, cy2), 8, (255, 0, 0), -1)
            lane1_shifted = lane1.copy()
            lane2_shifted = lane2.copy()
            lane1_shifted[:, 0, 1] += roi_start_y
            lane2_shifted[:, 0, 1] += roi_start_y
            cv2.drawContours(cv_image, [lane1_shifted], -1, (0, 255, 0), 3)
            cv2.drawContours(cv_image, [lane2_shifted], -1, (0, 255, 0), 3)
            cv2.circle(cv_image, (center_x, center_y), 10, (0, 255, 255), -1)
            cv2.line(cv_image, (center_x, center_y), (center_x, height), (0, 255, 255), 2)
            print(f"차선1 중심: ({cx1},{cy1}), 차선2 중심: ({cx2},{cy2}), 중앙선: ({center_x},{center_y})")

            # --- 주행 제어 ---
            error = (center_x - (width // 2)) / (width // 2)
            steering = np.clip(error * 0.6, -1, 1)
            throttle = 0.18
            self.piracer.set_steering_percent(float(steering))
            self.piracer.set_throttle_percent(float(throttle))
        else:
            print("차선 2개 미검출 → 정지")
            self.piracer.set_throttle_percent(0)

        # --- 여러 창 시각화 ---
        cv2.imshow("Lane Detection", cv_image)   # 전체 시각화
        cv2.imshow("ROI", roi)                   # 관심영역
        cv2.imshow("Yellow Mask", yellow_mask)   # 노란색 라인 마스크
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    finally:
        node.piracer.set_throttle_percent(0)
        node.piracer.set_steering_percent(0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
