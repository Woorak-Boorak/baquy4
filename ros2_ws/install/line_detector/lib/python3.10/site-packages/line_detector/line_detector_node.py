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
        lanes = self.detect_lanes(cv_image)

        # 차선 표시
        for lane in lanes:
            cv2.drawContours(cv_image, [lane], -1, (0,255,0), 3)
        cv2.imshow("Lane Detection", cv_image)
        cv2.waitKey(1)

    def detect_lanes(self, image):
        height, width, _ = image.shape
        roi = image[int(height*0.6):, :]
        cv2.imshow('ROI', roi)

        # HSV 변환 (노란색만 추출)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 노란색 범위 (필요에 따라 아래 값 조절 가능!)
        yellow_lower = (15, 80, 80)
        yellow_upper = (35, 255, 255)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        cv2.imshow('Yellow Mask', yellow_mask)

        # 외곽선 추출은 노란색 마스크만으로!
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lane_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 800]
        return lane_contours



def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
