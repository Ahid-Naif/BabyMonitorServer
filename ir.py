import RPi.GPIO as GPIO

ir_sensor = 16
buzzer = 18

GPIO.setmode(GPIO.BOARD)


GPIO.setup(ir_sensor,GPIO.IN)
GPIO.setup(buzzer,GPIO.OUT)

try:
    while True:
        print(GPIO.input(ir_sensor))
        if GPIO.input(ir_sensor):
            GPIO.output(buzzer, False)
        else:
            GPIO.output(buzzer, True)

except KeyboardInterrupt:
    GPIO.cleanup()