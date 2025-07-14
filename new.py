import cv2
import numpy as np
import time

def get_lane_masks(hsv):
    # (튜닝값: 현장 환경에 따라 추가 조정)
    lower_yellow = np.array([20, 90, 120])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([179, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((3, 3), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    return mask_white, mask_yellow, mask_red

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

interval = 0.3  # 0.3초마다 처리
frame_count = 0

try:
    while True:
        time.sleep(interval)
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임 읽기 실패!")
            break

        height, width = frame.shape[:2]
        roi = frame[int(height*0.6):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_white, mask_yellow, mask_red = get_lane_masks(hsv)

        lane_mask = cv2.bitwise_or(mask_white, mask_yellow)
        blurred = cv2.GaussianBlur(lane_mask, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=50)

        message = ""

        if lines is not None:
            message += "차선 OK  "
        else:
            message += "차선 X  "

        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stop_detected = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                cy = y + h//2
                if cy > roi.shape[0]*0.7:
                    message += "정지선!  "
                    stop_detected = True
                else:
                    message += "보호구역!  "
        if not stop_detected:
            message += "정지선/보호구역 X  "

        contours_w, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crosswalk_cnt = 0
        for cnt in contours_w:
            area = cv2.contourArea(cnt)
            if 100 < area < 3000:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > h:
                    crosswalk_cnt += 1
        if crosswalk_cnt >= 3:
            message += "횡단보도!  "
        else:
            message += "횡단보도 X  "

        print(f"[Frame {frame_count:03d}] {message}")
        # cv2.imwrite(f'lane_result_{frame_count:03d}.jpg', roi)
        frame_count += 1

except KeyboardInterrupt:
    print("중단됨. 종료합니다.")

cap.release()
