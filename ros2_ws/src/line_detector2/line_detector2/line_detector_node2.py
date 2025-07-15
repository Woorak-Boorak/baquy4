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
        self.last_detected = False

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = cv_image.shape[:2]
        roi_ratio = 0.01
        roi_start_y = int(height * roi_ratio)
        roi = cv_image[roi_start_y:, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_lower = (10, 90, 90)
        yellow_upper = (35, 255, 255)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lane_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 800]

        # 디폴트: 차선 미검출
        cx = None
        if lane_contours:
            biggest = max(lane_contours, key=cv2.contourArea)
            M = cv2.moments(biggest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00']) + roi_start_y
                # 시각화
                cv2.circle(cv_image, (cx, cy), 8, (255, 0, 0), -1)
                cv2.drawContours(cv_image, [biggest + np.array([0, roi_start_y])], -1, (0, 255, 0), 3)
                print(f"차선 중점: ({cx},{cy})")
        
        # 제어 (중점 기준 조향/속도)
        center = width // 2
        if cx is not None:
            error = (cx - center) / (width // 2)  # -1 ~ +1 스케일
            steering = np.clip(error * 0.6, -1, 1)  # 0.6은 조향 계수(환경에 따라 조정)
            throttle = 0.18  # 기본 주행 속도
            print(f"Steering: {steering:.2f}, Throttle: {throttle:.2f}")
            self.piracer.set_steering_percent(float(steering))
            self.piracer.set_throttle_percent(float(throttle))
            self.last_detected = True
        else:
            print("차선 미검출! -> 정지")
            self.piracer.set_throttle_percent(0)
            self.last_detected = False

        cv2.imshow("Lane Detection", cv_image)
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
