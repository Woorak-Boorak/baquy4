from piracer.vehicles import PiRacerPro

piracer = PiRacerPro()

# 스티어링/스로틀 직접 제어 예시
piracer.set_steering_percent(0.8)
piracer.set_throttle_percent(0.3)
piracer.set_throttle_percent(0)  # 정지

from piracer.gamepads import ShanWanGamepad
shanwan_gamepad = ShanWanGamepad()
mode_count = 0

while True:
    gamepad_input = shanwan_gamepad.read_data()

    # 오른쪽 아날로그 스틱의 y축 값(스로틀), 왼쪽 스틱의 x축 값(조향)
    throttle = gamepad_input.analog_stick_right.y * 0.5
    steering = gamepad_input.analog_stick_left.x
    buf = gamepad_input.button_start
    
    mode_count += buf
    if mode_count % 2 == 1:
        mode = 1
    else:
        mode = 0


    print(f'throttle={throttle}, steering={steering}, mode={mode}')

    piracer.set_throttle_percent(throttle)
    piracer.set_steering_percent(steering)