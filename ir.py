import RPi.GPIO as GPIO

ir_sensor = 16
buzzer = 18
GPIO.setmode(GPIO.BOARD)
GPIO.setup(ir_sensor,GPIO.IN)
GPIO.setup(buzzer,GPIO.OUT)

try:
    while True:
        print(GPIO.input(ir_sensor))
        GPIO.output(buzzer, GPIO.input(ir_sensor))

except KeyboardInterrupt:
    GPIO.cleanup()